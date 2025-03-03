import torch
import torch.nn as nn

import spherical_inr.differentiation as D
from typing import Optional


class SphericalLaplacianLoss(nn.Module):
    r"""Spherical Laplacian Loss.

    Computes the loss based on the spherical Laplacian of the network output.
    For a scalar function :math:`f` defined in spherical coordinates :math:`(r,\theta,\phi)`, the spherical Laplacian is given by

    .. math::
        \Delta_{sph} f = \frac{1}{r^2}\frac{\partial}{\partial r}\left( r^2\,\frac{\partial f}{\partial r} \right)
        + \frac{1}{r^2 \sin\theta}\frac{\partial}{\partial \theta}\left( \sin\theta\,\frac{\partial f}{\partial \theta} \right)
        + \frac{1}{r^2 \sin^2\theta}\frac{\partial^2 f}{\partial \phi^2}.

    The loss is defined as the mean squared value of the Laplacian:

    .. math::
        \mathcal{L} = \operatorname{mean}\Bigl( (\Delta_{sph} f)^2 \Bigr).
    """

    def forward(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        lap = D.spherical_laplacian(output, input, track=True)
        loss = (lap).pow(2).mean(dim=0)
        return loss


class CartesianLaplacianLoss(nn.Module):
    r"""Cartesian Laplacian Loss.

    Computes the loss based on the Cartesian Laplacian of the network output.
    For a scalar function :math:`f` defined on :math:`\mathbb{R}^n`, the Cartesian Laplacian is

    .. math::
        \Delta f = \sum_{i=1}^{n} \frac{\partial^2 f}{\partial x_i^2}.

    The loss is defined as the mean squared value of the Laplacian:

    .. math::
        \mathcal{L} = \operatorname{mean}\Bigl( (\Delta f)^2 \Bigr).
    """

    def forward(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        lap = D.cartesian_laplacian(output, input, track=True)
        loss = (lap).pow(2).mean(dim=0)
        return loss


class S2LaplacianLoss(nn.Module):
    r"""S2 Laplacian Loss.

    Computes the loss based on the Laplacian of the network output on the 2-sphere.
    For a function :math:`f` defined on the 2-sphere with coordinates :math:`(\theta,\phi)`, the Laplacian is

    .. math::
        \Delta_{S^2} f = \frac{1}{\sin\theta}\frac{\partial}{\partial \theta}\left( \sin\theta\,\frac{\partial f}{\partial \theta} \right)
        + \frac{1}{\sin^2\theta}\frac{\partial^2 f}{\partial \phi^2}.

    The loss is defined as the mean squared value of the Laplacian:

    .. math::
        \mathcal{L} = \operatorname{mean}\Bigl( (\Delta_{S^2} f)^2 \Bigr).
    """

    def forward(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        lap = D.s2_laplacian(output, input, track=True)
        loss = (lap).pow(2).mean(dim=0)
        return loss


class CartesianGradientMSELoss(nn.Module):
    r"""Cartesian Gradient MSE Loss.

    Computes the mean squared error (MSE) loss between the Cartesian gradient of the network output and a target gradient.
    For a function :math:`f` defined on :math:`\mathbb{R}^n`, the Cartesian gradient is

    .. math::
        \nabla f = \left( \frac{\partial f}{\partial x_1},\, \frac{\partial f}{\partial x_2},\, \dots,\, \frac{\partial f}{\partial x_n} \right).

    The loss is defined as

    .. math::
        \mathcal{L} = \operatorname{mean}\Bigl( \sum_{i=1}^{n}\Bigl( \frac{\partial f}{\partial x_i} - t_i \Bigr)^2 \Bigr),

    where :math:`t` denotes the target gradient.
    """

    def forward(
        self, target: torch.Tensor, output: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        grad = D.cartesian_gradient(output, input, track=True)
        loss = (grad - target).pow(2).sum(dim=-1).mean(dim=0)
        return loss


class SphericalGradientMSELoss(nn.Module):
    r"""Spherical Gradient MSE Loss.

    Computes the mean squared error (MSE) loss between the spherical gradient of the network output and a target gradient.
    For a function :math:`f` defined in spherical coordinates :math:`(r,\theta,\phi)`, the spherical gradient is given by

    .. math::
        \nabla_{sph} f = \left( \frac{\partial f}{\partial r},\, \frac{1}{r}\frac{\partial f}{\partial \theta},\, \frac{1}{r\,\sin\theta}\frac{\partial f}{\partial \phi} \right).

    The loss is defined as

    .. math::
        \mathcal{L} = \operatorname{mean}\Bigl( \sum_{i}\Bigl( (\nabla_{sph} f)_i - t_i \Bigr)^2 \Bigr),

    where :math:`t` represents the target gradient.
    """

    def forward(
        self, target: torch.Tensor, output: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        grad = D.spherical_gradient(output, input, track=True)
        loss = (grad - target).pow(2).sum(dim=-1).mean(dim=0)
        return loss


class S2GradientMSELoss(nn.Module):
    r"""S2 Gradient MSE Loss.

    Computes the mean squared error (MSE) loss between the gradient on the 2-sphere and a target gradient.
    For a function :math:`f` defined on the 2-sphere with coordinates :math:`(\theta,\phi)`, the gradient is

    .. math::
        \nabla_{S^2} f = \left( \frac{\partial f}{\partial \theta},\, \frac{1}{\sin\theta}\frac{\partial f}{\partial \phi} \right).

    The loss is defined as

    .. math::
        \mathcal{L} = \operatorname{mean}\Bigl( \sum_{i=1}^{2}\Bigl( (\nabla_{S^2} f)_i - t_i \Bigr)^2 \Bigr),

    where :math:`t` denotes the target gradient.
    """

    def forward(
        self, target: torch.Tensor, output: torch.Tensor, input: torch.Tensor
    ) -> torch.Tensor:
        grad = D.s2_gradient(output, input, track=True)
        loss = (grad - target).pow(2).sum(dim=-1).mean(dim=0)
        return loss


class CartesianGradientLaplacianMSELoss(nn.Module):
    r"""Cartesian Gradient-Laplacian MSE Loss.

    Computes a composite loss that combines the MSE between the Cartesian gradient of the network output and a target gradient with a regularization term based on the squared Cartesian Laplacian.
    For a function :math:`f` defined on :math:`\mathbb{R}^n`, let

    .. math::
        \nabla f = \left( \frac{\partial f}{\partial x_1},\, \dots,\, \frac{\partial f}{\partial x_n} \right)
        \quad \text{and} \quad
        \Delta f = \sum_{i=1}^{n} \frac{\partial^2 f}{\partial x_i^2}.

    The loss is defined as

    .. math::
        \mathcal{L} = \operatorname{mean}\Bigl( \sum_{i=1}^{n}\Bigl( \frac{\partial f}{\partial x_i} - t_i \Bigr)^2 \Bigr)
        \;+\; \alpha\,\operatorname{mean}\Bigl( (\Delta f)^2 \Bigr),

    where :math:`t` is the target gradient and :math:`\alpha` is a regularization parameter.
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

    Computes a composite loss that combines the MSE between the spherical gradient of the network output and a target gradient with a regularization term based on the squared spherical Laplacian.
    For a function :math:`f` defined in spherical coordinates :math:`(r,\theta,\phi)`, let

    .. math::
        \nabla_{sph} f = \left( \frac{\partial f}{\partial r},\, \frac{1}{r}\frac{\partial f}{\partial \theta},\, \frac{1}{r\,\sin\theta}\frac{\partial f}{\partial \phi} \right)

    and the spherical Laplacian is

    .. math::
        \Delta_{sph} f = \frac{1}{r^2}\frac{\partial}{\partial r}\left( r^2\,\frac{\partial f}{\partial r} \right)
        + \frac{1}{r^2 \sin\theta}\frac{\partial}{\partial \theta}\left( \sin\theta\,\frac{\partial f}{\partial \theta} \right)
        + \frac{1}{r^2 \sin^2\theta}\frac{\partial^2 f}{\partial \phi^2}.

    The loss is defined as

    .. math::
        \mathcal{L} = \operatorname{mean}\Bigl( \sum_{i}\Bigl( (\nabla_{sph} f)_i - t_i \Bigr)^2 \Bigr)
        \;+\; \alpha\,\operatorname{mean}\Bigl( (\Delta_{sph} f)^2 \Bigr),

    where :math:`t` is the target gradient and :math:`\alpha` is a regularization coefficient.
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

    Computes a composite loss for functions defined on the 2-sphere that combines the MSE between the gradient on the 2-sphere and a target gradient with a regularization term based on the squared Laplacian on the 2-sphere.
    For a function :math:`f` defined on the 2-sphere with coordinates :math:`(\theta,\phi)`, let

    .. math::
        \nabla_{S^2} f = \left( \frac{\partial f}{\partial \theta},\, \frac{1}{\sin\theta}\frac{\partial f}{\partial \phi} \right)

    and

    .. math::
        \Delta_{S^2} f = \frac{1}{\sin\theta}\frac{\partial}{\partial \theta}\left( \sin\theta\,\frac{\partial f}{\partial \theta} \right)
        + \frac{1}{\sin^2\theta}\frac{\partial^2 f}{\partial \phi^2}.

    The loss is defined as

    .. math::
        \mathcal{L} = \operatorname{mean}\Bigl( \sum_{i=1}^{2}\Bigl( (\nabla_{S^2} f)_i - t_i \Bigr)^2 \Bigr)
        \;+\; \alpha\,\operatorname{mean}\Bigl( (\Delta_{S^2} f)^2 \Bigr),

    where :math:`t` denotes the target gradient and :math:`\alpha` is a regularization parameter.
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
