import torch
import warnings
from typing import Optional


def _gradient(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    create_graph: bool = False,
    retain_graph: bool = False,
) -> torch.Tensor:

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
    return _gradient(outputs, inputs, create_graph=track, retain_graph=track)


def spherical_gradient(
    outputs: torch.Tensor, inputs: torch.Tensor, track: bool = False
) -> torch.Tensor:
    """
    Compute the spherical gradient for a function defined in (r, θ, φ) coordinates.

    Assumes inputs has three components [r, theta, phi].
    If the user only needs the tp or rt components, they can provide a dummy r value (e.g., r=1)
    or ignore the corresponding component of the result.

    The gradient components are adjusted as:
        - d/dr remains unchanged,
        - d/dθ is divided by r,
        - d/dφ is divided by (r * sin(theta)).
    """

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


def spherical_divergence(
    outputs: torch.Tensor, inputs: torch.Tensor, track: bool = False
) -> torch.Tensor:

    if outputs.size(-1) != 3:
        raise ValueError(
            "Spherical divergence is only defined for spherical vector fields."
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

    scaling_factors = [1 / r2, r_sin_theta, r_sin_theta]

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


def cartesian_divergence(
    outputs: torch.Tensor, inputs: torch.Tensor, track: bool = False
) -> torch.Tensor:

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


def spherical_laplacian(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    track: bool = False,
) -> torch.Tensor:
    """
    Compute the spherical laplacian of a function defined in (r,θ,φ) coordinates.

    If a precomputed gradient is provided via the `grad` parameter, it is used directly
    to compute the divergence. Otherwise, the gradient is computed from `outputs`.

    Args:
        outputs: Tensor representing function values. Ignored if `grad` is provided.
        inputs: Coordinates in (r, θ, φ) space.
        track: Whether to track gradients for higher-order derivatives.
        grad: Optional precomputed gradient of the function.

    Returns:
        Tensor representing the laplacian.
    """

    grad = spherical_gradient(outputs, inputs, track=True)
    laplacian = spherical_divergence(grad, inputs, track=track)
    return laplacian


def cartesian_laplacian(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    track: bool = False,
) -> torch.Tensor:
    """
    Compute the Cartesian laplacian of a function.

    If a precomputed gradient is provided via the `grad` parameter, it is used directly
    to compute the divergence. Otherwise, the gradient is computed from `outputs`.

    Args:
        outputs: Tensor representing function values. Ignored if `grad` is provided.
        inputs: Coordinates in Cartesian space.
        track: Whether to track gradients for higher-order derivatives.
        grad: Optional precomputed gradient of the function.

    Returns:
        Tensor representing the laplacian.
    """

    grad = cartesian_gradient(outputs, inputs, track=True)

    laplacian = cartesian_divergence(grad, inputs, track=track)
    return laplacian
