from typing import Any, Tuple

import cv2
import numpy as np
import numpy.typing as npt


class Distorter:
    def __init__(
        self,
        intrinsic: npt.NDArray[Any],
        distortion: npt.NDArray[Any],
        distorted_im_size: Tuple[int, int],
        rotation: npt.NDArray[Any] = None,
        extrinsic: npt.NDArray[Any] = None,
        new_intrinsic: npt.NDArray[Any] = None,
    ):
        if rotation is None:
            rotation = np.eye(3)

        assert intrinsic.shape == (3, 3), intrinsic.shape
        assert distortion.shape == (4,), distortion.shape
        assert rotation.shape == (3, 3), rotation.shape

        if new_intrinsic is None:
            new_intrinsic = intrinsic.copy()

        assert new_intrinsic.shape == (3, 3), new_intrinsic.shape

        if extrinsic is not None:
            pad_rotation = np.pad(rotation, ((0, 1), (0, 1)))
            pad_rotation[-1, -1] = 1.0
            new_extrinsic = pad_rotation @ extrinsic

        self.intrinsic = intrinsic
        self.distortion = distortion
        self.rotation = rotation
        self.extrinsic = extrinsic
        self.new_extrinsic = new_extrinsic
        self.new_intrinsic = new_intrinsic
        self.distorted_im_size = distorted_im_size

        yx = np.mgrid[0:distorted_im_size[1], 0:distorted_im_size[0]]
        xy = yx[::-1]
        points = xy.reshape(2, -1).T.astype(np.float32)
        points = np.expand_dims(points, axis=0)
        undistorted_points = cv2.fisheye.undistortPoints(
            distorted=points,
            K=new_intrinsic,
            D=distortion,
            R=np.linalg.inv(rotation),
            P=intrinsic,
        )
        undistorted_points = undistorted_points.reshape(distorted_im_size[1], distorted_im_size[0], 2)
        self.mapx = undistorted_points[..., 0]
        self.mapy = undistorted_points[..., 1]

    def __call__(self, frame: npt.NDArray[Any]) -> npt.NDArray[Any]:
        frame = cv2.remap(
            frame,
            self.mapx,
            self.mapy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return frame
