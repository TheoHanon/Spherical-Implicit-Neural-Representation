import torch


def rsph2_to_cart3(rsph2_coords: torch.Tensor) -> torch.Tensor:
    """
    Converts spherical coordinates [r, theta, phi] to 3d Cartesian coordinates [x, y, z].

    Parameters:
      spherical_coords (torch.Tensor): A tensor with shape [..., 3] where the last dimension
        represents [theta, phi].
        - theta is the polar angle (angle from the positive z-axis).
        - phi is the azimuthal angle (angle in the x-y plane from the positive x-axis).

    Returns:
      torch.Tensor: A tensor of shape [..., 3] representing the Cartesian coordinates [x, y, z].

    Raises:
      ValueError: If the last dimension of spherical_coords is not 2.
    """

    if rsph2_coords.shape[-1] != 3:
        raise ValueError("The last dimension of spherical_coords must be 3.")

    r, theta, phi = rsph2_coords.unbind(dim=-1)

    sin_theta = torch.sin(theta)
    x = r * sin_theta * torch.cos(phi)
    y = r * sin_theta * torch.sin(phi)
    z = r * torch.cos(theta)

    return torch.stack([x, y, z], dim=-1)


def sph2_to_cart3(sph2_coords: torch.Tensor) -> torch.Tensor:
    """
    Converts spherical coordinates [theta, phi] to 3d Cartesian coordinates [x, y, z].

    Parameters:
      spherical_coords (torch.Tensor): A tensor with shape [..., 2] where the last dimension
        represents [theta, phi].
        - theta is the polar angle (angle from the positive z-axis).
        - phi is the azimuthal angle (angle in the x-y plane from the positive x-axis).

    Returns:
      torch.Tensor: A tensor of shape [..., 3] representing the Cartesian coordinates [x, y, z].

    Raises:
      ValueError: If the last dimension of spherical_coords is not 2.
    """

    if sph2_coords.shape[-1] != 2:
        raise ValueError("The last dimension of spherical_coords must be 2.")

    theta, phi = sph2_coords.unbind(dim=-1)

    sin_theta = torch.sin(theta)
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = torch.cos(theta)

    return torch.stack([x, y, z], dim=-1)


def rsph1_to_cart2(rsph1_coords: torch.Tensor) -> torch.Tensor:
    """
    Converts angular coordinates [r, theta] to 2d Cartesian coordinates [x, y].

    Parameters:
      spherical_coords (torch.Tensor): A tensor with shape [..., 2] where the last dimension
        represents [theta].
        - theta is the angular coordinate

    Returns:
      torch.Tensor: A tensor of shape [..., 2] representing the 2d Cartesian coordinates [x, y].

    Raises:
      ValueError: If the last dimension of spherical_coords is not 1.
    """

    if rsph1_coords.shape[-1] != 2:
        raise ValueError("The last dimension of spherical_coords must be 2.")

    r, theta = rsph1_coords.unbind(dim=-1)

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    return torch.stack([x, y], dim=-1)


def sph1_to_cart2(sph1_coords: torch.Tensor) -> torch.Tensor:
    """
    Converts angular coordinates [theta] to 2d Cartesian coordinates [x, y].

    Parameters:
      spherical_coords (torch.Tensor): A tensor with shape [..., 1] where the last dimension
        represents [theta].
        - theta is the angular coordinate

    Returns:
      torch.Tensor: A tensor of shape [..., 2] representing the 2d Cartesian coordinates [x, y].

    Raises:
      ValueError: If the last dimension of spherical_coords is not 1.
    """

    if sph1_coords.shape[-1] != 1:
        raise ValueError("The last dimension of spherical_coords must be 1.")

    theta = sph1_coords.squeeze(dim=-1)

    x = torch.cos(theta)
    y = torch.sin(theta)

    return torch.stack([x, y], dim=-1)
