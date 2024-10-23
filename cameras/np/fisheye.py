from typing import Any, Tuple, cast

import numpy as np
import numpy.typing as npt

from .camera import Camera


class Fisheye(Camera):
    def __init__(
        self,
        width: int,
        height: int,
        intrinsic: npt.NDArray[Any],
        extrinsic: npt.NDArray[Any],
        distortion: npt.NDArray[Any],
        fov: npt.NDArray[Any] = None,
        max_count: int = 10,
    ):
        assert intrinsic.shape[-2:] == (3, 3), intrinsic.shape
        assert distortion.shape[-1:] == (4,), distortion.shape

        self.intrinsic = intrinsic
        self.inv_intrinsic = np.linalg.inv(intrinsic)
        self.distortion = distortion

        self.fx = self.intrinsic[..., 0:1, 0]
        self.fy = self.intrinsic[..., 1:2, 1]
        self.cx = self.intrinsic[..., 0:1, 2]
        self.cy = self.intrinsic[..., 1:2, 2]

        self.eps = 1e-6
        self.max_count = max_count

        if fov is None:
            fov = np.array(
                [
                    [-np.pi / 2, np.pi / 2],
                    [-np.pi / 2, np.pi / 2],
                ],
                dtype=np.float32,
            )

        super(Fisheye, self).__init__(
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
        pre_dims = points_3d.shape[:-1]
        assert points_3d.shape[-1] == 3

        points_2d = points_3d[..., :2] / (points_3d[..., 2:3] + self.eps)
        r = np.sqrt(
            np.expand_dims(points_2d, axis=-2) @ np.expand_dims(points_2d, axis=-1)
        ).squeeze(-1).squeeze(-1)
        assert r.shape == pre_dims, (r.shape, pre_dims)

        theta = np.arctan(r)
        theta_d = theta * (
            1 + self.distortion[..., 0:1] * np.power(theta, 2)
            + self.distortion[..., 1:2] * np.power(theta, 4)
            + self.distortion[..., 2:3] * np.power(theta, 6)
            + self.distortion[..., 3:4] * np.power(theta, 8)
        )
        assert theta_d.shape == pre_dims, (theta_d.shape, pre_dims)
        inv_r = np.where(r > 1e-8, 1.0 / (r + self.eps), 1.0)
        cdist = np.where(r > 1e-8, theta_d * inv_r, 1.0)

        x = cdist * points_2d[..., 0]
        y = cdist * points_2d[..., 1]

        u = self.fx * x + self.cx
        v = self.fy * y + self.cy

        points_2d = np.stack([u, v], axis=-1)
        return cast(npt.NDArray[Any], points_2d)

    def im_to_cam(
        self,
        points_2d: npt.NDArray[Any],
        depth: npt.NDArray[Any],
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        u = points_2d[..., 0]
        v = points_2d[..., 1]

        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy

        theta_d = np.sqrt(x ** 2 + y ** 2)
        valid_points = np.logical_and(
            theta_d < np.pi / 2,
            theta_d > -np.pi / 2,
        )
        theta_d = np.clip(theta_d, -np.pi / 2, np.pi / 2)

        theta = theta_d

        for _ in range(self.max_count):
            theta2 = theta * theta
            theta4 = theta2 * theta2
            theta6 = theta4 * theta2
            theta8 = theta6 * theta2
            k0_theta2 = self.distortion[..., 0:1] * theta2
            k1_theta4 = self.distortion[..., 1:2] * theta4
            k2_theta6 = self.distortion[..., 2:3] * theta6
            k3_theta8 = self.distortion[..., 3:4] * theta8
            theta_fix = (
                (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d)
                / (1 + 3 * k0_theta2 + 5 * k1_theta4 + 7 * k2_theta6 + 9 * k3_theta8)
            )
            theta = theta - theta_fix

            if np.all(abs(theta_fix) < self.eps):
                break

        converged_points = abs(theta_fix) < self.eps
        valid_points = np.logical_and(
            valid_points,
            converged_points,
        )

        scale = np.tan(theta) / theta_d

        theta_flipped = np.logical_or(
            np.logical_and(theta_d < 0, theta > 0),
            np.logical_and(theta_d > 0, theta < 0),
        )
        valid_points = np.logical_and(
            valid_points,
            ~theta_flipped,
        )

        x = x * scale
        y = y * scale
        z = np.ones_like(x)
        z[theta_flipped] = -1e6

        points_3d = np.stack([x, y, z], axis=-1)
        points_3d = np.expand_dims(points_3d, axis=-2) * np.expand_dims(depth, axis=-1)
        return points_3d, valid_points
