from typing import Any, Tuple, cast

import numpy as np
import numpy.typing as npt

from .camera import Camera


class Equirectangular(Camera):
    def __init__(
        self,
        width: int,
        height: int,
        extrinsic: npt.NDArray[Any],
    ):
        """
        Initializes an equirectangular projection model.

        Args:
            width (int): Image width in pixels.
            height (int): Image height in pixels.
            extrinsic (npt.NDArray[Any]): 4x4 camera pose transformation matrix.
        """
        self.eps = 1e-6
        super().__init__(
            width=width,
            height=height,
            extrinsic=extrinsic,
            fov=np.array([[-np.pi, np.pi], [-np.pi / 2, np.pi / 2]], dtype=np.float32),
        )

    def cam_to_im(
        self,
        points_3d: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        """
        Projects 3D points in camera space to 2D equirectangular image coordinates.

        Args:
            points_3d (npt.NDArray[Any]): Array of shape (..., 3) representing 3D points.

        Returns:
            npt.NDArray[Any]: 2D pixel coordinates (..., 2) in the equirectangular image.
        """
        x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]

        # Convert Cartesian to spherical coordinates (θ, φ)
        theta = np.arctan2(x, z)  # Longitude (-π, π)
        phi = np.arcsin(y / (np.linalg.norm(points_3d, axis=-1) + self.eps))  # Latitude (-π/2, π/2)

        # Normalize angles to pixel coordinates
        u = (theta + np.pi) / (2 * np.pi) * self.width
        v = (phi + np.pi / 2) / np.pi * self.height  # Flip to match image convention

        points_2d = np.stack([u, v], axis=-1)
        return cast(npt.NDArray[Any], points_2d)

    def im_to_cam(
        self,
        points_2d: npt.NDArray[Any],
        depth: npt.NDArray[Any],
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """
        Converts 2D image coordinates back to 3D camera space.

        Args:
            points_2d (npt.NDArray[Any]): Image pixel coordinates (..., 2).
            depth (npt.NDArray[Any]): Depth values (..., n).

        Returns:
            Tuple[npt.NDArray[Any], npt.NDArray[Any]]: Reconstructed 3D points (..., n, 3)
            and validity mask (..., n).
        """
        u, v = points_2d[..., 0], points_2d[..., 1]

        # Convert pixel coordinates to spherical angles
        theta = (u / self.width) * (2 * np.pi) - np.pi  # Longitude (-π, π)
        phi = (v / self.height - .5) * np.pi  # Latitude (-π/2, π/2)

        # Convert spherical angles to Cartesian coordinates
        x = np.cos(phi) * np.sin(theta)
        y = np.sin(phi)
        z = np.cos(phi) * np.cos(theta)

        points_3d = np.stack([x, y, z], axis=-1)
        points_3d = np.expand_dims(points_3d, axis=-2) * np.expand_dims(depth, -1)

        valid_points = np.ones(points_3d.shape[:-2], dtype=bool)

        return cast(npt.NDArray[Any], points_3d), cast(npt.NDArray[Any], valid_points)
