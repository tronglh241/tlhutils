from typing import Any, Tuple, cast

import numpy as np
import numpy.typing as npt

from .camera import Camera


class PinHole(Camera):
    def __init__(
        self,
        width: int,
        height: int,
        intrinsic: npt.NDArray[Any],
        extrinsic: npt.NDArray[Any],
        fov: npt.NDArray[Any] = None,
    ):
        assert intrinsic.shape[-2:] == (3, 3), intrinsic.shape

        self.intrinsic = intrinsic
        self.inv_intrinsic = np.linalg.inv(intrinsic)
        self.eps = 1e-6

        if fov is None:
            fov = np.array(
                [
                    [-np.pi / 2, np.pi / 2],
                    [-np.pi / 2, np.pi / 2],
                ],
                dtype=np.float32,
            )

        super(PinHole, self).__init__(
            width=width,
            height=height,
            extrinsic=extrinsic,
            fov=fov,
        )

    def cam_to_im(
        self,
        points_3d: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        # points_3d (..., 3)

        points_2d = points_3d @ self.intrinsic.swapaxes(-1, -2)
        points_2d = points_2d[..., :2] / (points_2d[..., 2:3] + self.eps)
        points_2d = points_2d[..., :2]

        return cast(npt.NDArray[Any], points_2d)

    def im_to_cam(
        self,
        points_2d: npt.NDArray[Any],
        depth: npt.NDArray[Any],
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        # points_2d (..., 2)
        # depth (..., n)
        assert points_2d.shape[-1] == 2
        points_2d = np.concatenate(
            [
                points_2d,
                np.ones((*points_2d.shape[:-1], 1)),
            ],
            axis=-1,
        )

        points_3d = points_2d @ self.inv_intrinsic.swapaxes(-1, -2)
        points_3d = np.expand_dims(points_3d, axis=-2) * np.expand_dims(depth, -1)

        valid_points = np.ones((points_3d.shape[:-2]), dtype=bool)
        # points_3d (..., n, 3)
        return points_3d, valid_points
