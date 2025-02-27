import torch
import torch.nn as nn

import spherical_inr.differentiation as D
from typing import Optional


class SphericalLaplacianLoss(nn.Module):
    def forward(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        lap = D.spherical_laplacian(output, input, track=True)
        loss = (lap).pow(2).mean(dim=0)
        return loss


class CartesianLaplacianLoss(nn.Module):
    def forward(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        lap = D.cartesian_laplacian(output, input, track=True)
        loss = (lap).pow(2).mean(dim=0)
        return loss


class CartesianGradientMSELoss(nn.Module):
    def forward(
        self, target: torch.Tensor, output: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        grad = D.cartesian_gradient(output, input, track=True)
        loss = (grad - target).pow(2).sum(dim=-1).mean(dim=0)
        return loss


class SphericalGradientMSELoss(nn.Module):
    def forward(
        self, target: torch.Tensor, output: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        grad = D.spherical_gradient(output, input, track=True)
        loss = (grad - target).pow(2).sum(dim=-1).mean(dim=0)
        return loss


class CartesianGradientLaplacianMSELoss(nn.Module):
    def __init__(self, alpha_reg: float = 1.0):
        super().__init__()
        self.register_buffer("alpha_reg", torch.tensor(alpha_reg))

    def forward(
        self, target: torch.Tensor, output: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        grad = D.cartesian_gradient(output, input, track=True)
        lap = D.cartesian_divergence(grad, input, track=True)
        loss = (grad - target).pow(2).sum(dim=-1).mean(
            dim=0
        ) + self.alpha_reg * lap.pow(2).mean(dim=0)
        return loss


class SphericalGradientLaplacianMSELoss(nn.Module):
    def __init__(self, alpha_reg: float = 1.0):
        super().__init__()
        self.register_buffer("alpha_reg", torch.tensor(alpha_reg))

    def forward(
        self, target: torch.Tensor, output: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        grad = D.spherical_gradient(output, input, track=True)
        lap = D.spherical_divergence(grad, input, track=True)
        loss = (grad - target).pow(2).sum(dim=-1).mean(
            dim=0
        ) + self.alpha_reg * lap.pow(2).mean(dim=0)
        return loss
