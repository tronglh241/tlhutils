from abc import ABC, abstractmethod
from typing import Any, Tuple, cast

import numpy as np
import numpy.typing as npt


class Camera(ABC):
    def __init__(
        self,
        width: int,
        height: int,
        extrinsic: npt.NDArray[Any],
        fov: npt.NDArray[Any] = None,
    ):
        yx = np.mgrid[:height, :width]
        xy = yx[::-1]

        assert xy.shape == (2, height, width), xy.shape
        pix_coords = xy.reshape(2, -1).T

        assert extrinsic.shape[-2:] == (4, 4), extrinsic.shape[-2:]

        inv_extrinsic = np.linalg.inv(extrinsic)

        if fov is None:
            self.fov = np.array(
                [
                    [-np.pi, np.pi],
                    [-np.pi, np.pi],
                ],
                dtype=np.float32,
            )
        else:
            self.fov = fov

        self.width = width
        self.height = height
        self.extrinsic = extrinsic
        self.inv_extrinsic = inv_extrinsic
        self.pix_coords = pix_coords

    def world_to_im(
        self,
        points_3d: npt.NDArray[Any],
        normalize: bool = True,
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        # points_3d (..., 3)
        pre_dims = points_3d.shape[:-1]
        assert points_3d.shape[-1] == 3, points_3d.shape

        points_3d = self.world_to_cam(points_3d)
        assert points_3d.shape == (*pre_dims, 3), (points_3d.shape, (*pre_dims, 3))

        is_in_fov = self.in_fov(points_3d)
        assert pre_dims == is_in_fov.shape

        points_depth = points_3d[..., 2]
        assert pre_dims == points_depth.shape

        points_2d = self.cam_to_im(points_3d)
        assert points_2d.shape == (*pre_dims, 2), (points_2d.shape, (*pre_dims, 2))

        is_in_image = self.is_in_image(points_2d)
        assert pre_dims == is_in_image.shape

        valid_points = np.logical_and(
            is_in_fov,
            is_in_image,
        )
        assert valid_points.shape == pre_dims

        if normalize:
            points_2d = points_2d / np.array([self.width - 1, self.height - 1], dtype=np.float32)
            points_2d = (points_2d - 0.5) * 2

        assert points_2d.shape == (*pre_dims, 2), (points_2d.shape, (*pre_dims, 2))
        assert valid_points.shape == pre_dims, (valid_points.shape, pre_dims)
        assert points_depth.shape == pre_dims, (points_depth.shape, pre_dims)

        # points_2d (..., 2)
        # valid_points (..., )
        # points_depth (..., )
        return points_2d, valid_points, points_depth

    @abstractmethod
    def cam_to_im(self, points_3d: npt.NDArray[Any]) -> npt.NDArray[Any]:
        pass

    @abstractmethod
    def im_to_cam(
        self,
        points_2d: npt.NDArray[Any],
        depth: npt.NDArray[Any],
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        pass

    def cam_to_world(self, points_3d: npt.NDArray[Any]) -> npt.NDArray[Any]:
        # points_3d (..., 3)
        assert points_3d.shape[-1] == 3, points_3d.shape
        org_shape = points_3d.shape

        points_3d = np.concatenate((points_3d, np.ones((*org_shape[:-1], 1))), axis=-1)
        points_3d = points_3d @ self.inv_extrinsic[:3].T
        assert points_3d.shape == org_shape, (points_3d.shape, org_shape)

        # points_3d (..., 3)
        return points_3d

    def world_to_cam(self, points_3d: npt.NDArray[Any]) -> npt.NDArray[Any]:
        # points_3d (..., 3)
        assert points_3d.shape[-1] == 3, points_3d.shape
        org_shape = points_3d.shape

        points_3d = np.concatenate((points_3d, np.ones((*org_shape[:-1], 1))), axis=-1)
        points_3d = points_3d @ self.extrinsic[:3].T
        assert points_3d.shape == org_shape, (points_3d.shape, org_shape)

        # points_3d (..., 3)
        return points_3d

    def im_to_cam_map(
        self,
        depth_map: npt.NDArray[Any],
        mask: npt.NDArray[Any],
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        # depth_map (..., height, width, n)
        # mask (..., height, width)

        *pre_dims, height, width, n = depth_map.shape
        assert (height, width) == (self.height, self.width)
        assert (height, width) == mask.shape[-2:]

        depth_map = depth_map.reshape(*pre_dims, height * width, n)

        points_3d, valid_points = self.im_to_cam(self.pix_coords, depth_map)
        assert points_3d.shape == (*pre_dims, height * width, n, 3)

        points_3d = points_3d.reshape(*pre_dims, height, width, n, 3)
        valid_points = valid_points.reshape(*pre_dims, height, width)
        valid_points = np.where(mask > 0.5, valid_points, False)

        assert valid_points.shape == (*pre_dims, height, width)

        # points_3d (..., height, width, n, 3)
        # valid_points (..., height, width)
        return points_3d, valid_points

    def is_in_image(self, points_2d: npt.NDArray[Any]) -> npt.NDArray[Any]:
        is_point_in_image = np.logical_and.reduce(
            [
                points_2d[..., 0] <= self.width - 1,
                points_2d[..., 0] >= 0,
                points_2d[..., 1] <= self.height - 1,
                points_2d[..., 1] >= 0,
            ]
        )
        return cast(npt.NDArray[Any], is_point_in_image)

    def _in_fov(
        self,
        x1: npt.NDArray[Any],
        x2: npt.NDArray[Any],
        fov: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        return np.logical_and(
            fov[..., 0:1] <= np.arctan2(x1, x2),
            np.arctan2(x1, x2) <= fov[..., 1:2],
        )

    def in_fov(self, points_3d: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.logical_and(
            self._in_fov(points_3d[..., 0], points_3d[..., 2], self.fov[..., 0, :]),  # horizontal
            self._in_fov(points_3d[..., 1], points_3d[..., 2], self.fov[..., 1, :]),  # vertical
        )
