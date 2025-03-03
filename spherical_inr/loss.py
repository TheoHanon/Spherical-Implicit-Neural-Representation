import torch
import torch.nn as nn

import spherical_inr.differentiation as D
from typing import Optional


class SphericalLaplacianLoss(nn.Module):
    r"""Spherical Laplacian Loss.

    Computes a loss based on the spherical Laplacian of a function's output with respect to its input.
    For a function :math:`f`, the spherical Laplacian is computed as

    .. math::
        \Delta_{sph} f = \text{spherical\_laplacian}(f, x)

    and the loss is defined as the mean squared error (MSE) of the Laplacian:

    .. math::
        \ell = \operatorname{mean}\Bigl((\Delta_{sph} f)^2\Bigr).
    """

    def forward(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        lap = D.spherical_laplacian(output, input, track=True)
        loss = (lap).pow(2).mean(dim=0)
        return loss


class CartesianLaplacianLoss(nn.Module):
    r"""Cartesian Laplacian Loss.

    Computes a loss based on the Cartesian Laplacian of a function's output with respect to its input.
    For a function :math:`f`, the Cartesian Laplacian is given by

    .. math::
        \Delta f = \text{cartesian\_laplacian}(f, x)

    and the loss is defined as

    .. math::
        \ell = \operatorname{mean}\Bigl((\Delta f)^2\Bigr).
    """

    def forward(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        lap = D.cartesian_laplacian(output, input, track=True)
        loss = (lap).pow(2).mean(dim=0)
        return loss


class S2LaplacianLoss(nn.Module):
    r"""S2 Laplacian Loss.

    Computes a loss based on the Laplacian computed on the 2-sphere (S²) of a function's output with respect to its input.
    For a function :math:`f`, the S2 Laplacian is computed as

    .. math::
        \Delta_{S^2} f = \text{s2\_laplacian}(f, x)

    and the loss is defined as

    .. math::
        \ell = \operatorname{mean}\Bigl((\Delta_{S^2} f)^2\Bigr).
    """

    def forward(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        lap = D.s2_laplacian(output, input, track=True)
        loss = (lap).pow(2).mean(dim=0)
        return loss


class CartesianGradientMSELoss(nn.Module):
    r"""Cartesian Gradient MSE Loss.

    Computes the mean squared error (MSE) loss between the Cartesian gradient of the output and a target gradient.
    For a function :math:`f`, the Cartesian gradient is computed as

    .. math::
        \nabla f = \text{cartesian\_gradient}(f, x)

    and the loss is defined by

    .. math::
        \ell = \operatorname{mean}\Bigl(\sum_{i}\Bigl(\nabla f_i - t_i\Bigr)^2\Bigr),

    where :math:`t` represents the target gradient and the summation is performed over the gradient components.
    """

    def forward(
        self, target: torch.Tensor, output: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        grad = D.cartesian_gradient(output, input, track=True)
        loss = (grad - target).pow(2).sum(dim=-1).mean(dim=0)
        return loss


class SphericalGradientMSELoss(nn.Module):
    r"""Spherical Gradient MSE Loss.

    Computes the mean squared error (MSE) loss between the spherical gradient of the output and a target gradient.
    For a function :math:`f`, the spherical gradient is computed as

    .. math::
        \nabla_{sph} f = \text{spherical\_gradient}(f, x)

    and the loss is defined as

    .. math::
        \ell = \operatorname{mean}\Bigl(\sum_{i}\Bigl(\nabla_{sph} f_i - t_i\Bigr)^2\Bigr),

    where :math:`t` is the target gradient.
    """

    def forward(
        self, target: torch.Tensor, output: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        grad = D.spherical_gradient(output, input, track=True)
        loss = (grad - target).pow(2).sum(dim=-1).mean(dim=0)
        return loss


class S2GradientMSELoss(nn.Module):
    r"""S2 Gradient MSE Loss.

    Computes the mean squared error (MSE) loss between the gradient on the 2-sphere (S²) and a target gradient.
    For a function :math:`f`, the S2 gradient is computed as

    .. math::
        \nabla_{S^2} f = \text{s2\_gradient}(f, x)

    and the loss is defined by

    .. math::
        \ell = \operatorname{mean}\Bigl(\sum_{i}\Bigl(\nabla_{S^2} f_i - t_i\Bigr)^2\Bigr),

    with the summation performed over the gradient components.
    """

    def forward(
        self, target: torch.Tensor, output: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        grad = D.s2_gradient(output, input, track=True)
        loss = (grad - target).pow(2).sum(dim=-1).mean(dim=0)
        return loss


class CartesianGradientLaplacianMSELoss(nn.Module):
    r"""Cartesian Gradient-Laplacian MSE Loss.

    Computes a composite loss that combines the mean squared error (MSE) between the computed Cartesian gradient
    and a target gradient with a regularization term based on the squared Cartesian Laplacian.
    For a function :math:`f`, let

    .. math::
        \nabla f = \text{cartesian\_gradient}(f, x)
        \quad \text{and} \quad
        \Delta (\nabla f) = \text{cartesian\_divergence}(\nabla f, x).

    The loss is defined as

    .. math::
        \ell = \operatorname{mean}\Bigl(\sum_{i}\Bigl(\nabla f_i - t_i\Bigr)^2\Bigr)
        \;+\; \alpha_{\text{reg}}\,\operatorname{mean}\Bigl((\Delta (\nabla f))^2\Bigr),

    where :math:`t` is the target gradient and :math:`\alpha_{\text{reg}}` is a regularization coefficient.
    """

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
    r"""Spherical Gradient-Laplacian MSE Loss.

    Computes a composite loss consisting of the mean squared error (MSE) between the computed spherical gradient
    and a target gradient, along with a regularization term based on the squared spherical Laplacian of the gradient.
    For a function :math:`f`, define

    .. math::
        \nabla_{sph} f = \text{spherical\_gradient}(f, x)
        \quad \text{and} \quad
        \Delta_{sph} (\nabla f) = \text{spherical\_divergence}(\nabla_{sph} f, x).

    Then, the loss is given by

    .. math::
        \ell = \operatorname{mean}\Bigl(\sum_{i}\Bigl(\nabla_{sph} f_i - t_i\Bigr)^2\Bigr)
        \;+\; \alpha_{\text{reg}}\,\operatorname{mean}\Bigl((\Delta_{sph} (\nabla f))^2\Bigr),

    where :math:`t` is the target gradient and :math:`\alpha_{\text{reg}}` is a regularization coefficient.
    """

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


class S2GradientLaplacianMSELoss(nn.Module):
    r"""S2 Gradient-Laplacian MSE Loss.

    Computes a composite loss for functions defined on the 2-sphere (S²) that combines the mean squared error (MSE)
    between the computed S2 gradient and a target gradient with a regularization term based on the squared divergence
    of the S2 gradient. For a function :math:`f`, let

    .. math::
        \nabla_{S^2} f = \text{s2\_gradient}(f, x)
        \quad \text{and} \quad
        \Delta_{S^2} (\nabla f) = \text{s2\_divergence}(\nabla_{S^2} f, x).

    The loss is then defined as

    .. math::
        \ell = \operatorname{mean}\Bigl(\sum_{i}\Bigl(\nabla_{S^2} f_i - t_i\Bigr)^2\Bigr)
        \;+\; \alpha_{\text{reg}}\,\operatorname{mean}\Bigl((\Delta_{S^2} (\nabla f))^2\Bigr),

    where :math:`t` denotes the target gradient and :math:`\alpha_{\text{reg}}` is a regularization parameter.
    """

    def __init__(self, alpha_reg: float = 1.0):
        super().__init__()
        self.register_buffer("alpha_reg", torch.tensor(alpha_reg))

    def forward(
        self, target: torch.Tensor, output: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        grad = D.s2_gradient(output, input, track=True)
        lap = D.s2_divergence(grad, input, track=True)
        loss = (grad - target).pow(2).sum(dim=-1).mean(
            dim=0
        ) + self.alpha_reg * lap.pow(2).mean(dim=0)
        return loss
