from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from .camera import Camera


class CameraTransformer:
    def __init__(
        self,
        src_cam: Camera,
        dst_cam: Camera,
        rotation: npt.NDArray[Any] = None,
    ):
        self.src_cam = src_cam
        self.dst_cam = dst_cam

        if rotation is not None:
            assert rotation.shape == (3, 3)
        else:
            rotation = np.eye(3, dtype=np.float32)

        inv_rotation = np.linalg.inv(rotation)

        points_3d, valid_points = self.dst_cam.im_to_cam_map(
            depth_map=np.ones((self.dst_cam.height, self.dst_cam.width, 1), dtype=np.float32),
            mask=np.ones((self.dst_cam.height, self.dst_cam.width), dtype=np.float32),
        )
        assert points_3d.shape[-2] == 1, points_3d.shape
        points_3d = points_3d.squeeze(-2)

        points_3d = inv_rotation @ np.expand_dims(points_3d, axis=-1)
        points_3d = points_3d.squeeze(axis=-1)

        points_2d = self.src_cam.cam_to_im(points_3d).astype(np.float32)
        assert points_2d.shape == (self.dst_cam.height, self.dst_cam.width, 2)

        points_2d[~valid_points] = -1.0

        self.xmap = points_2d[..., 0]
        self.ymap = points_2d[..., 1]

    def transform(self, im: npt.NDArray[Any]):
        transformed = cv2.remap(
            src=im,
            map1=self.xmap,
            map2=self.ymap,
            interpolation=cv2.INTER_LINEAR,
        )
        return transformed
