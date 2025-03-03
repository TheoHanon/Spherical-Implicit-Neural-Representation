import torch
import warnings
from typing import Optional

__all__ = [
    "cartesian_gradient",
    "spherical_gradient",
    "s2_gradient",
    "cartesian_divergence",
    "spherical_divergence",
    "s2_divergence",
    "cartesian_laplacian",
    "spherical_laplacian",
    "s2_laplacian",
]


def _gradient(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    create_graph: bool = False,
    retain_graph: bool = False,
) -> torch.Tensor:
    r"""
    Compute the gradient of a function with respect to its inputs.

    Given a function \( f: \mathbb{R}^n \to \mathbb{R} \), the gradient is defined as
    \[
    \nabla_x f(x) = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n}\right)
    \]
    This function computes the derivative of `outputs` with respect to `inputs` using automatic differentiation.
    If the derivative is not defined (i.e. is None), it returns a zero tensor matching the shape of `inputs`,
    preserving gradient tracking when requested.

    Args:
        outputs (torch.Tensor): Tensor representing the function values \( f(x) \).
        inputs (torch.Tensor): Tensor representing the input variables \( x \).
        create_graph (bool, optional): If True, constructs the gradient graph to enable higher-order derivatives.
            (default: False)
        retain_graph (bool, optional): If True, retains the computational graph for further operations.
            (default: False)

    Returns:
        torch.Tensor: The computed gradient \( \nabla_x f(x) \).
    """

    grad = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )[0]

    if grad is None:
        warnings.warn(
            "Computed _gradient is None; replacing with zeros", RuntimeWarning
        )
        grad = (
            torch.zeros_like(inputs)
            if not create_graph
            else torch.zeros_like(inputs).requires_grad_(True)
        )

    return grad


def cartesian_gradient(
    outputs: torch.Tensor, inputs: torch.Tensor, track: bool = False
) -> torch.Tensor:
    r"""Compute the gradient of a scalar function in Cartesian coordinates.

    For a scalar function :math:`f: \mathbb{R}^n \rightarrow \mathbb{R}`, the gradient is given by

    .. math::
        \nabla f(x) = \left[ \frac{\partial f}{\partial x_1},\, \frac{\partial f}{\partial x_2},\, \dots,\, \frac{\partial f}{\partial x_n} \right].

    This function computes :math:`\nabla f(x)` with respect to the Cartesian coordinates provided in ``inputs``.
    Enabling the ``track`` parameter allows for the construction of higher-order derivative graphs.

    Args:
        outputs (torch.Tensor): Tensor representing the function values :math:`f(x)`.
        inputs (torch.Tensor): Tensor representing the Cartesian coordinates :math:`x`.
        track (bool, optional): If True, enables gradient tracking for higher-order derivatives.
            (default: False)

    Returns:
        torch.Tensor: The gradient :math:`\nabla f(x)` in Cartesian coordinates.
    """

    return _gradient(outputs, inputs, create_graph=track, retain_graph=track)


def spherical_gradient(
    outputs: torch.Tensor, inputs: torch.Tensor, track: bool = False
) -> torch.Tensor:
    r"""Compute the gradient of a function defined in spherical coordinates :math:`(r, \theta, \phi)`.

    For a function :math:`f(r, \theta, \phi)`, the spherical gradient is defined as

    .. math::
        \nabla f = \hat{r}\,\frac{\partial f}{\partial r} \;+\; \hat{\theta}\,\frac{1}{r}\frac{\partial f}{\partial \theta} \;+\; \hat{\phi}\,\frac{1}{r\,\sin\theta}\frac{\partial f}{\partial \phi}.

    This function computes the gradient of :math:`f(r,\theta,\phi)` with respect to the spherical coordinates.
    The input tensor ``inputs`` must have three components representing :math:`[r, \theta, \phi]`.

    Args:
        outputs (torch.Tensor): Tensor representing the function values :math:`f(r, \theta, \phi)`.
        inputs (torch.Tensor): Tensor of shape (..., 3) representing spherical coordinates :math:`[r, \theta, \phi]`.
        track (bool, optional): If True, enables gradient tracking for higher-order derivatives.
            (default: False)

    Returns:
        torch.Tensor: The spherical gradient with components scaled as above.

    Raises:
        ValueError: If ``inputs`` does not have three components.
    """

    if inputs.size(-1) != 3:
        raise ValueError(
            "Spherical gradient is only defined for 3D spherical (r, θ, φ) coordinates"
        )

    grad = _gradient(outputs, inputs, create_graph=track, retain_graph=track)
    r = inputs[..., 0]
    theta = inputs[..., 1]

    with torch.set_grad_enabled(track):
        grad = torch.stack(
            [
                grad[..., 0],
                grad[..., 1] / r,
                grad[..., 2] / (r * torch.sin(theta)),
            ],
            dim=-1,
        )

    return grad


def s2_gradient(
    outputs: torch.Tensor, inputs: torch.Tensor, track: bool = False
) -> torch.Tensor:
    r"""Compute the gradient of a function defined on the 2-sphere (S²) with respect to the angles :math:`(\theta, \phi)`.

    For a function :math:`f(\theta, \phi)` defined on S², the gradient is expressed as

    .. math::
        \nabla_{S^2} f = \hat{\theta}\,\frac{\partial f}{\partial \theta} \;+\; \hat{\phi}\,\frac{1}{\sin\theta}\frac{\partial f}{\partial \phi}.

    This function computes the gradient of :math:`f(\theta, \phi)` on S². The input tensor ``inputs`` must have
    two components corresponding to :math:`(\theta, \phi)`.

    Args:
        outputs (torch.Tensor): Tensor representing the function values :math:`f(\theta, \phi)`.
        inputs (torch.Tensor): Tensor of shape (..., 2) representing spherical coordinates :math:`(\theta, \phi)`.
        track (bool, optional): If True, enables gradient tracking for higher-order derivatives.
            (default: False)

    Returns:
        torch.Tensor: The gradient on S², :math:`\nabla_{S^2} f`.

    Raises:
        ValueError: If ``inputs`` does not have two components.
    """

    if inputs.size(-1) != 2:
        raise ValueError(
            "S2 gradient is only defined for 2D spherical (θ, φ) coordinates"
        )

    grad = _gradient(outputs, inputs, create_graph=track, retain_graph=track)
    theta = inputs[..., 0]

    with torch.set_grad_enabled(track):
        grad = torch.stack(
            [
                grad[..., 0],
                grad[..., 1] / (torch.sin(theta)),
            ],
            dim=-1,
        )

    return grad


def cartesian_divergence(
    outputs: torch.Tensor, inputs: torch.Tensor, track: bool = False
) -> torch.Tensor:
    r"""Compute the divergence of a vector field in Cartesian coordinates.

    For a vector field :math:`\mathbf{F}: \mathbb{R}^n \rightarrow \mathbb{R}^n` with components
    :math:`\mathbf{F}(x) = \left[F_1(x),\, F_2(x),\, \dots,\, F_n(x)\right]`, the divergence is defined as

    .. math::
        \nabla \cdot \mathbf{F} = \sum_{i=1}^{n} \frac{\partial F_i}{\partial x_i}.

    This function computes the divergence by summing the partial derivatives of each component of the vector field
    with respect to its corresponding Cartesian coordinate.

    Args:
        outputs (torch.Tensor): Tensor representing the vector field, where the last dimension contains the components
            of :math:`\mathbf{F}`.
        inputs (torch.Tensor): Tensor representing the Cartesian coordinates :math:`x`.
        track (bool, optional): If True, enables gradient tracking for higher-order derivatives.
            (default: False)

    Returns:
        torch.Tensor: The divergence :math:`\nabla \cdot \mathbf{F}`.
    """

    outputs_to_grad = [outputs[..., i] for i in range(outputs.size(-1))]

    div = torch.zeros_like(outputs[..., 0])
    for i, out in enumerate(outputs_to_grad):

        div += _gradient(
            out,
            inputs,
            create_graph=track,
            retain_graph=True if i < outputs.size(-1) - 1 else track,
        )[..., i]

    return div


def spherical_divergence(
    outputs: torch.Tensor, inputs: torch.Tensor, track: bool = False
) -> torch.Tensor:
    r"""Compute the divergence of a vector field in spherical coordinates.

    For a vector field :math:`\mathbf{F}(r,\theta,\phi) = \left[F_r,\, F_\theta,\, F_\phi\right]`, the divergence is given by

    .. math::
        \nabla \cdot \mathbf{F} = \frac{1}{r^2}\frac{\partial}{\partial r}\Bigl(r^2 F_r\Bigr)
        \;+\; \frac{1}{r \sin\theta}\frac{\partial}{\partial \theta}\Bigl(\sin\theta F_\theta\Bigr)
        \;+\; \frac{1}{r \sin\theta}\frac{\partial F_\phi}{\partial \phi}.

    This function computes the divergence of the vector field in spherical coordinates.
    The tensor ``outputs`` must have three components.

    Args:
        outputs (torch.Tensor): Tensor of shape (..., 3) representing the vector field in spherical coordinates.
        inputs (torch.Tensor): Tensor of shape (..., 3) representing spherical coordinates :math:`[r, \theta, \phi]`.
        track (bool, optional): If True, enables gradient tracking for higher-order derivatives.
            (default: False)

    Returns:
        torch.Tensor: The divergence :math:`\nabla \cdot \mathbf{F}` in spherical coordinates.

    Raises:
        ValueError: If ``outputs`` does not have three components.
    """

    if outputs.size(-1) != 3:
        raise ValueError(
            "Spherical divergence is only defined for (r_hat, θ_hat, φ_hat) vector fields."
        )

    r = inputs[..., 0]
    theta = inputs[..., 1]

    sin_theta = torch.sin(theta)
    r_sin_theta = r * sin_theta
    r2 = r**2

    # Combine gradient computations
    outputs_to_grad = [
        r2 * outputs[..., 0],
        sin_theta * outputs[..., 1],
        outputs[..., 2],
    ]

    scaling_factors = [1 / r2, 1 / r_sin_theta, 1 / r_sin_theta]

    div = torch.zeros_like(outputs[..., 0])

    for i, (out, scaling_factors) in enumerate(zip(outputs_to_grad, scaling_factors)):

        grad = _gradient(
            out,
            inputs,
            create_graph=track,
            retain_graph=True if i < outputs.size(-1) - 1 else track,
        )[..., i]
        with torch.set_grad_enabled(track):
            div += grad * scaling_factors

    return div


def s2_divergence(
    outputs: torch.Tensor, inputs: torch.Tensor, track: bool = False
) -> torch.Tensor:
    r"""Compute the divergence of a vector field defined on the 2-sphere (S²).

    For a vector field on S², :math:`\mathbf{F}(\theta,\phi) = \left[F_\theta,\, F_\phi\right]`, the divergence is defined as

    .. math::
        \nabla_{S^2} \cdot \mathbf{F} = \frac{1}{\sin\theta}\frac{\partial}{\partial \theta}\Bigl(\sin\theta F_\theta\Bigr)
        \;+\; \frac{1}{\sin\theta}\frac{\partial F_\phi}{\partial \phi}.

    This function computes the divergence of the vector field on S².
    The tensor ``outputs`` must have two components.

    Args:
        outputs (torch.Tensor): Tensor representing the vector field on S² with two components.
        inputs (torch.Tensor): Tensor of shape (..., 2) representing spherical coordinates :math:`(\theta, \phi)`.
        track (bool, optional): If True, enables gradient tracking for higher-order derivatives.
            (default: False)

    Returns:
        torch.Tensor: The divergence :math:`\nabla_{S^2} \cdot \mathbf{F}` on the 2-sphere.

    Raises:
        ValueError: If ``outputs`` does not have two components.
    """

    if outputs.size(-1) != 2:
        raise ValueError(
            "Spherical divergence is only defined for s2 (θ_hat, φ_hat) vector fields."
        )

    theta = inputs[..., 0]
    sin_theta = torch.sin(theta)

    # Combine gradient computations
    outputs_to_grad = [
        sin_theta * outputs[..., 0],
        outputs[..., 1],
    ]

    scaling_factors = [1 / sin_theta, 1 / sin_theta]

    div = torch.zeros_like(outputs[..., 0])

    for i, (out, scaling_factors) in enumerate(zip(outputs_to_grad, scaling_factors)):

        grad = _gradient(
            out,
            inputs,
            create_graph=track,
            retain_graph=True if i < outputs.size(-1) - 1 else track,
        )[..., i]
        with torch.set_grad_enabled(track):
            div += grad * scaling_factors

    return div


def cartesian_laplacian(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    track: bool = False,
) -> torch.Tensor:
    r"""Compute the Laplacian of a scalar function in Cartesian coordinates.

    The Laplacian is defined as the divergence of the gradient:

    .. math::
        \Delta f = \nabla \cdot \left(\nabla f\right)
        \;=\; \sum_{i=1}^{n} \frac{\partial^2 f}{\partial x_i^2}.

    This function computes the gradient of :math:`f(x)` with respect to the Cartesian coordinates ``x``
    and then evaluates its divergence to obtain the Laplacian. The ``track`` parameter enables higher-order derivative tracking.

    Args:
        outputs (torch.Tensor): Tensor representing the function values :math:`f(x)`.
        inputs (torch.Tensor): Tensor representing Cartesian coordinates :math:`x`.
        track (bool, optional): If True, enables gradient tracking for higher-order derivatives.
            (default: False)

    Returns:
        torch.Tensor: The Laplacian :math:`\Delta f` of the function.
    """

    grad = cartesian_gradient(outputs, inputs, track=True)

    laplacian = cartesian_divergence(grad, inputs, track=track)
    return laplacian


def spherical_laplacian(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    track: bool = False,
) -> torch.Tensor:
    r"""Compute the Laplacian of a function defined in spherical coordinates :math:`(r, \theta, \phi)`.

    The Laplacian is computed as the divergence of the gradient:

    .. math::
        \Delta f = \nabla \cdot \left(\nabla f\right).

    In spherical coordinates, the Laplacian of :math:`f(r,\theta,\phi)` is expressed as

    .. math::
        \Delta f = \frac{1}{r^2}\frac{\partial}{\partial r}\Bigl(r^2 \frac{\partial f}{\partial r}\Bigr)
        \;+\; \frac{1}{r^2 \sin\theta}\frac{\partial}{\partial \theta}\Bigl(\sin\theta \frac{\partial f}{\partial \theta}\Bigr)
        \;+\; \frac{1}{r^2 \sin^2\theta}\frac{\partial^2 f}{\partial \phi^2}.

    This function computes the gradient of :math:`f(r,\theta,\phi)` in spherical coordinates and then
    its divergence to obtain the Laplacian.

    Args:
        outputs (torch.Tensor): Tensor representing the function values :math:`f(r,\theta,\phi)`.
        inputs (torch.Tensor): Tensor representing spherical coordinates :math:`[r, \theta, \phi]`.
        track (bool, optional): If True, enables gradient tracking for higher-order derivatives.
            (default: False)

    Returns:
        torch.Tensor: The Laplacian :math:`\Delta f` in spherical coordinates.
    """

    grad = spherical_gradient(outputs, inputs, track=True)
    laplacian = spherical_divergence(grad, inputs, track=track)
    return laplacian


def s2_laplacian(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    track: bool = False,
) -> torch.Tensor:
    r"""Compute the Laplacian of a function defined on the 2-sphere (S²).

    The Laplacian on S² is defined as the divergence of the gradient on the sphere:

    .. math::
        \Delta_{S^2} f = \nabla_{S^2} \cdot \left(\nabla_{S^2} f\right).

    For a function :math:`f(\theta,\phi)` on S², the Laplacian expands to

    .. math::
        \Delta_{S^2} f = \frac{1}{\sin\theta}\frac{\partial}{\partial \theta}\Bigl(\sin\theta \frac{\partial f}{\partial \theta}\Bigr)
        \;+\; \frac{1}{\sin^2\theta}\frac{\partial^2 f}{\partial \phi^2}.

    This function computes the gradient of :math:`f(\theta,\phi)` on S² and then evaluates its divergence
    to obtain the Laplacian.

    Args:
        outputs (torch.Tensor): Tensor representing the function values :math:`f(\theta,\phi)`.
        inputs (torch.Tensor): Tensor representing spherical coordinates :math:`(\theta, \phi)` on S².
        track (bool, optional): If True, enables gradient tracking for higher-order derivatives.
            (default: False)

    Returns:
        torch.Tensor: The Laplacian :math:`\Delta_{S^2} f` on the 2-sphere.
    """

    grad = s2_gradient(outputs, inputs, track=True)
    laplacian = s2_divergence(grad, inputs, track=track)
    return laplacian
