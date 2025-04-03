import torch


def rtp_to_r3(rtp_coords: torch.Tensor) -> torch.Tensor:
    """
    Converts full spherical coordinates (rtp) [r, θ, ϕ] to R^3 Cartesian coordinates [x, y, z].

    Parameters:
      rtp_coords (torch.Tensor): Tensor with shape [..., 3] representing [r, θ, ϕ],
        where r is the radius, θ is the polar angle, and ϕ is the azimuthal angle.

    Returns:
      torch.Tensor: Tensor with shape [..., 3] representing [x, y, z].

    Raises:
      ValueError: If the last dimension of rtp_coords is not 3.
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
    """
    Converts spherical coordinates on the unit sphere (tp) [θ, ϕ] to R^3 Cartesian coordinates [x, y, z].

    Parameters:
      tp_coords (torch.Tensor): Tensor with shape [..., 2] representing [θ, ϕ].

    Returns:
      torch.Tensor: Tensor with shape [..., 3] representing [x, y, z] on the unit sphere.

    Raises:
      ValueError: If the last dimension of tp_coords is not 2.
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
    """
    Converts R^3 Cartesian coordinates [x, y, z] to full spherical coordinates (rtp) [r, θ, ϕ].

    Parameters:
      r3_coords (torch.Tensor): Tensor with shape [..., 3] representing [x, y, z].

    Returns:
      torch.Tensor: Tensor with shape [..., 3] representing [r, θ, ϕ],
        where θ is the polar angle and ϕ is the azimuthal angle.

    Raises:
      ValueError: If the last dimension of r3_coords is not 3.
    """
    if r3_coords.shape[-1] != 3:
        raise ValueError("The last dimension of r3_coords must be 3.")
    
    x, y, z = r3_coords.unbind(dim=-1)
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(torch.clamp(z / (r + 1e-8), -1.0, 1.0))
    phi = torch.atan2(y, x)
    return torch.stack([r, theta, phi], dim=-1)


def r3_to_tp(r3_coords: torch.Tensor) -> torch.Tensor:
    """
    Converts R^3 Cartesian coordinates [x, y, z] (assumed to lie on the unit sphere)
    to spherical coordinates on the unit sphere (tp) [θ, ϕ].

    Parameters:
      r3_coords (torch.Tensor): Tensor with shape [..., 3] representing [x, y, z].

    Returns:
      torch.Tensor: Tensor with shape [..., 2] representing [θ, ϕ].

    Raises:
      ValueError: If the last dimension of r3_coords is not 3.
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
    """
    Converts full polar coordinates (rt) [r, θ] to R^2 Cartesian coordinates [x, y].

    Parameters:
      rt_coords (torch.Tensor): Tensor with shape [..., 2] representing [r, θ].

    Returns:
      torch.Tensor: Tensor with shape [..., 2] representing [x, y].

    Raises:
      ValueError: If the last dimension of rt_coords is not 2.
    """
    if rt_coords.shape[-1] != 2:
        raise ValueError("The last dimension of rt_coords must be 2.")
    
    r, theta = rt_coords.unbind(dim=-1)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=-1)


def t_to_r2(t_coords: torch.Tensor) -> torch.Tensor:
    """
    Converts angle-only representation (t) [θ] on the unit circle to R^2 Cartesian coordinates [x, y].

    Parameters:
      t_coords (torch.Tensor): Tensor with shape [..., 1] representing [θ].

    Returns:
      torch.Tensor: Tensor with shape [..., 2] representing [x, y] on the unit circle.

    Raises:
      ValueError: If the last dimension of t_coords is not 1.
    """
    if t_coords.shape[-1] != 1:
        raise ValueError("The last dimension of t_coords must be 1.")
    
    theta = t_coords.squeeze(dim=-1)
    x = torch.cos(theta)
    y = torch.sin(theta)
    return torch.stack([x, y], dim=-1)


def r2_to_rt(r2_coords: torch.Tensor) -> torch.Tensor:
    """
    Converts R^2 Cartesian coordinates [x, y] to full polar coordinates (rt) [r, θ].

    Parameters:
      r2_coords (torch.Tensor): Tensor with shape [..., 2] representing [x, y].

    Returns:
      torch.Tensor: Tensor with shape [..., 2] representing [r, θ].

    Raises:
      ValueError: If the last dimension of r2_coords is not 2.
    """
    if r2_coords.shape[-1] != 2:
        raise ValueError("The last dimension of r2_coords must be 2.")
    
    x, y = r2_coords.unbind(dim=-1)
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    return torch.stack([r, theta], dim=-1)


def r2_to_t(r2_coords: torch.Tensor) -> torch.Tensor:
    """
    Converts R^2 Cartesian coordinates [x, y] (assumed to lie on the unit circle)
    to angle-only representation (t) [θ].

    Parameters:
      r2_coords (torch.Tensor): Tensor with shape [..., 2] representing [x, y].

    Returns:
      torch.Tensor: Tensor with shape [..., 1] representing [θ].

    Raises:
      ValueError: If the last dimension of r2_coords is not 2.
    """
    if r2_coords.shape[-1] != 2:
        raise ValueError("The last dimension of r2_coords must be 2.")
    
    norm = torch.norm(r2_coords, dim=-1, keepdim=True)
    unit_coords = r2_coords / (norm + 1e-8)
    x, y = unit_coords.unbind(dim=-1)
    theta = torch.atan2(y, x)
    return theta.unsqueeze(dim=-1)
