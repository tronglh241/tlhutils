from typing import Tuple

import torch

from .camera import Camera


class PinHole(Camera):
    def __init__(
        self,
        width: int,
        height: int,
        intrinsic: torch.Tensor,
        extrinsic: torch.Tensor,
        fov: torch.Tensor = None,
        device: str = 'cpu',
    ):
        assert intrinsic.shape[-2:] == (3, 3), intrinsic.shape

        self.intrinsic = intrinsic.to(device).to(torch.float32)
        self.inv_intrinsic = torch.linalg.inv(intrinsic)
        self.eps = 1e-6

        if fov is None:
            fov = torch.tensor(
                [
                    [-torch.pi / 2, torch.pi / 2],
                    [-torch.pi / 2, torch.pi / 2],
                ],
                dtype=torch.float32,
                device=device,
            )

        super(PinHole, self).__init__(
            width=width,
            height=height,
            extrinsic=extrinsic,
            fov=fov,
            device=device,
        )

    def cam_to_im(
        self,
        points_3d: torch.Tensor,
    ) -> torch.Tensor:
        # points_3d (..., 3)

        points_2d = points_3d @ self.intrinsic.swapaxes(-1, -2)
        points_2d = points_2d[..., :2] / (points_2d[..., 2:3] + self.eps)
        points_2d = points_2d[..., :2]

        return points_2d

    def im_to_cam(
        self,
        points_2d: torch.Tensor,
        depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # points_2d (..., 2)
        # depth (..., n)
        assert points_2d.shape[-1] == 2
        points_2d = torch.cat(
            [
                points_2d,
                torch.ones((*points_2d.shape[:-1], 1), device=self.device),
            ],
            dim=-1,
        )

        points_3d = points_2d @ self.inv_intrinsic.swapaxes(-1, -2)
        points_3d = points_3d.unsqueeze(dim=-2) * depth.unsqueeze(dim=-1)

        valid_points = torch.ones((points_3d.shape[:-2]), dtype=torch.bool, device=self.device)
        # points_3d (..., n, 3)
        return points_3d, valid_points
