from typing import Any, Tuple

import cv2
import numpy as np
import numpy.typing as npt


class Undistorter:
    def __init__(
        self,
        intrinsic: npt.NDArray[Any],
        distortion: npt.NDArray[Any],
        undistorted_im_size: Tuple[int, int],
        rotation: npt.NDArray[Any] = None,
        extrinsic: npt.NDArray[Any] = None,
        f: Tuple[float, float] = None,
        c: Tuple[float, float] = None,
    ):

        if rotation is None:
            rotation = np.eye(3)

        assert intrinsic.shape == (3, 3), intrinsic.shape
        assert distortion.shape == (4,), distortion.shape
        assert rotation.shape == (3, 3), rotation.shape

        if f is None or c is None:
            new_intrinsic = intrinsic.copy()
        else:
            new_intrinsic = np.array([
                [f[0], 0.0, c[0]],
                [0.0, f[1], c[1]],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)

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
        self.undistorted_im_size = undistorted_im_size

        self.mapx, self.mapy = cv2.fisheye.initUndistortRectifyMap(
            K=intrinsic,
            D=distortion,
            R=rotation,
            P=new_intrinsic,
            size=undistorted_im_size,
            m1type=cv2.CV_16SC2,
        )

    def __call__(self, frame: npt.NDArray[Any]) -> npt.NDArray[Any]:
        frame = cv2.remap(
            frame,
            self.mapx,
            self.mapy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return frame
