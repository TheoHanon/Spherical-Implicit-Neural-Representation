import torch
import torch.nn as nn


from abc import ABC, abstractmethod


class Transform(nn.Module, ABC):

    def __init__(self, input_dim: int) -> None:
        super(Transform, self).__init__()
        self.input_dim = input_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class SphericalToCartesian(Transform):

    def __init__(self, input_dim: int, unit: bool = False) -> None:

        super(SphericalToCartesian, self).__init__(input_dim=input_dim)

        if input_dim not in (2, 3):
            raise ValueError(f"Unsupported dimension: {input_dim}. Must be 2 or 3.")

        self.unit = unit

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not self.unit:
            r = x[..., 0]
            angles = x[..., 1:]
        else:
            r = 1.0
            angles = x

        if self.input_dim == 2:

            theta = angles[0]
            x_coord = r * torch.cos(theta)
            y_coord = r * torch.sin(theta)
            return torch.stack([x_coord, y_coord], dim=-1)

        elif self.input_dim == 3:

            theta, phi = angles
            sin_theta = torch.sin(theta)
            x_coord = r * sin_theta * torch.cos(phi)
            y_coord = r * sin_theta * torch.sin(phi)
            z_coord = r * torch.cos(theta)
            return torch.stack([x_coord, y_coord, z_coord], dim=-1)

    def extra_repr(self):
        return f"input_dim={self.input_dim}, unit={self.unit}"
