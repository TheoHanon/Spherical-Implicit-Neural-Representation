r"""Coordinate transforms between Cartesian and spherical/polar parameterizations.

The 3D functions use spherical coordinates :math:`(r, \theta, \phi)` with
:math:`\theta` the polar angle and :math:`\phi` the azimuth. The 2D functions
use polar coordinates :math:`(r, \theta)` or just the angle :math:`\theta` on
the unit circle.
"""

import torch


def rtp_to_r3(rtp_coords: torch.Tensor) -> torch.Tensor:
    r"""Map spherical coordinates :math:`(r, \theta, \phi)` to Cartesian coordinates.

    The conversion follows

    .. math::
        x = r \sin\theta \cos\phi,\quad
        y = r \sin\theta \sin\phi,\quad
        z = r \cos\theta.

    Parameters
    ----------
    rtp_coords: torch.Tensor
        Tensor with shape (..., 3) representing :math:`[r, \theta, \phi]`.

    Returns
    -------
    torch.Tensor
        Tensor with shape (..., 3) containing :math:`[x, y, z]`.

    Raises
    ------
    ValueError
        If the last dimension of ``rtp_coords`` is not 3.
    """
    if rtp_coords.shape[-1] != 3:
        raise ValueError("The last dimension of rtp_coords must be 3.")

    r, theta, phi = rtp_coords.unbind(dim=-1)
    sin_theta = torch.sin(theta)
    x = r * sin_theta * torch.cos(phi)
    y = r * sin_theta * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)


def tp_to_r3(tp_coords: torch.Tensor) -> torch.Tensor:
    r"""Map unit-sphere angles :math:`(\theta, \phi)` to Cartesian coordinates.

    Parameters
    ----------
    tp_coords: torch.Tensor
        Tensor with shape (..., 2) representing :math:`[\theta, \phi]`.

    Returns
    -------
    torch.Tensor
        Tensor with shape (..., 3) containing :math:`[x, y, z]` on the unit sphere.

    Raises
    ------
    ValueError
        If the last dimension of ``tp_coords`` is not 2.
    """
    if tp_coords.shape[-1] != 2:
        raise ValueError("The last dimension of tp_coords must be 2.")

    theta, phi = tp_coords.unbind(dim=-1)
    sin_theta = torch.sin(theta)
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)


def r3_to_rtp(r3_coords: torch.Tensor) -> torch.Tensor:
    r"""Convert Cartesian coordinates to spherical coordinates :math:`(r, \theta, \phi)`.

    Parameters
    ----------
    r3_coords: torch.Tensor
        Tensor with shape (..., 3) representing :math:`[x, y, z]`.

    Returns
    -------
    torch.Tensor
        Tensor with shape (..., 3) containing :math:`[r, \theta, \phi]`.

    Raises
    ------
    ValueError
        If the last dimension of ``r3_coords`` is not 3.
    """
    if r3_coords.shape[-1] != 3:
        raise ValueError("The last dimension of r3_coords must be 3.")

    x, y, z = r3_coords.unbind(dim=-1)
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(torch.clamp(z / (r + 1e-8), -1.0, 1.0))
    phi = torch.atan2(y, x)
    return torch.stack([r, theta, phi], dim=-1)


def r3_to_tp(r3_coords: torch.Tensor) -> torch.Tensor:
    r"""Project Cartesian coordinates on the unit sphere to angles :math:`(\theta, \phi)`.

    Parameters
    ----------
    r3_coords: torch.Tensor
        Tensor with shape (..., 3) representing :math:`[x, y, z]`.

    Returns
    -------
    torch.Tensor
        Tensor with shape (..., 2) containing :math:`[\theta, \phi]`.

    Raises
    ------
    ValueError
        If the last dimension of ``r3_coords`` is not 3.
    """
    if r3_coords.shape[-1] != 3:
        raise ValueError("The last dimension of r3_coords must be 3.")

    norm = torch.norm(r3_coords, dim=-1, keepdim=True)
    unit_coords = r3_coords / (norm + 1e-8)
    x, y, z = unit_coords.unbind(dim=-1)
    theta = torch.acos(torch.clamp(z, -1.0, 1.0))
    phi = torch.atan2(y, x)
    return torch.stack([theta, phi], dim=-1)


# === 2D Conversion Functions ===


def rt_to_r2(rt_coords: torch.Tensor) -> torch.Tensor:
    r"""Map polar coordinates :math:`(r, \theta)` to Cartesian coordinates in :math:`\mathbb{R}^2`.

    Parameters
    ----------
    rt_coords: torch.Tensor
        Tensor with shape (..., 2) representing :math:`[r, \theta]`.

    Returns
    -------
    torch.Tensor
        Tensor with shape (..., 2) containing :math:`[x, y]`.

    Raises
    ------
    ValueError
        If the last dimension of ``rt_coords`` is not 2.
    """
    if rt_coords.shape[-1] != 2:
        raise ValueError("The last dimension of rt_coords must be 2.")

    r, theta = rt_coords.unbind(dim=-1)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=-1)


def t_to_r2(t_coords: torch.Tensor) -> torch.Tensor:
    r"""Convert an angle on the unit circle to Cartesian coordinates.

    Parameters
    ----------
    t_coords: torch.Tensor
        Tensor with shape (..., 1) containing the angle :math:`\theta`.

    Returns
    -------
    torch.Tensor
        Tensor with shape (..., 2) containing :math:`[x, y]` on the unit circle.

    Raises
    ------
    ValueError
        If the last dimension of ``t_coords`` is not 1.
    """
    if t_coords.shape[-1] != 1:
        raise ValueError("The last dimension of t_coords must be 1.")

    theta = t_coords.squeeze(dim=-1)
    x = torch.cos(theta)
    y = torch.sin(theta)
    return torch.stack([x, y], dim=-1)


def r2_to_rt(r2_coords: torch.Tensor) -> torch.Tensor:
    r"""Convert Cartesian coordinates in :math:`\mathbb{R}^2` to polar form :math:`(r, \theta)`.

    Parameters
    ----------
    r2_coords: torch.Tensor
        Tensor with shape (..., 2) representing :math:`[x, y]`.

    Returns
    -------
    torch.Tensor
        Tensor with shape (..., 2) containing :math:`[r, \theta]`.

    Raises
    ------
    ValueError
        If the last dimension of ``r2_coords`` is not 2.
    """
    if r2_coords.shape[-1] != 2:
        raise ValueError("The last dimension of r2_coords must be 2.")

    x, y = r2_coords.unbind(dim=-1)
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return torch.stack([r, theta], dim=-1)


def r2_to_t(r2_coords: torch.Tensor) -> torch.Tensor:
    r"""Project Cartesian coordinates on the unit circle to their angle :math:`\theta`.

    Parameters
    ----------
    r2_coords: torch.Tensor
        Tensor with shape (..., 2) representing :math:`[x, y]`.

    Returns
    -------
    torch.Tensor
        Tensor with shape (..., 1) containing :math:`[\theta]`.

    Raises
    ------
    ValueError
        If the last dimension of ``r2_coords`` is not 2.
    """
    if r2_coords.shape[-1] != 2:
        raise ValueError("The last dimension of r2_coords must be 2.")

    norm = torch.norm(r2_coords, dim=-1, keepdim=True)
    unit_coords = r2_coords / (norm + 1e-8)
    x, y = unit_coords.unbind(dim=-1)
    theta = torch.atan2(y, x)
    return theta.unsqueeze(dim=-1)
