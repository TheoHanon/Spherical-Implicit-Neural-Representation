r"""Positional encodings for spherical data via harmonics, Fourier features, and Herglotz kernels."""

from __future__ import annotations

import torch
import torch.nn as nn
import math

from . import _kernels as PE

from typing import Tuple


__all__ = [
    "HerglotzPE",
    "FourierPE",
    "SphericalHarmonicsPE",
]


class SphericalHarmonicsPE(nn.Module):
    r"""
    Real spherical harmonics positional encoding.

    This module maps spherical angles
    :math:`x = (\theta, \phi) \in [0,\pi] \times [-\pi,\pi]`
    to a vector of real spherical harmonics

    .. math::
        \psi^{\mathrm{SH}}(x) =
        \bigl(
            Y_{\ell_1}^{m_1}(\theta,\phi), \dots,
            Y_{\ell_N}^{m_N}(\theta,\phi)
        \bigr),

    where the index pairs :math:`(\ell_k, m_k)` follow the standard ordering

    .. math::
        (0,0), (1,-1),(1,0),(1,1),(2,-2),\dots

    and only the first ``num_atoms = N`` basis functions are retained.

    The real spherical harmonics are defined as

    .. math::
        Y_\ell^m(\theta,\phi)
        = N_{\ell m}\,P_\ell^{|m|}(\cos\theta)
        \begin{cases}
            \cos(m\phi), & m \ge 0, \\
            \sin(|m|\phi), & m < 0,
        \end{cases}

    where :math:`P_\ell^m` are the associated Legendre polynomials and
    :math:`N_{\ell m}` is a normalization constant.

    Parameters
    ----------
    num_atoms: int
        Number of spherical harmonic basis functions returned.

    Attributes
    ----------
    num_atoms : int
    """

    def __init__(
        self,
        num_atoms: int,
    ) -> None:

        super().__init__()
        self.num_atoms = num_atoms

        L_upper = math.ceil(math.sqrt(num_atoms)) - 1
        ms = [m for l in range(L_upper + 1) for m in range(-l, l + 1)][: self.num_atoms]
        ls = [l for l in range(L_upper + 1) for _ in range(-l, l + 1)][: self.num_atoms]

        # store as buffers for device moves
        self.register_buffer(
            "l_list", torch.tensor(ls, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "m_list", torch.tensor(ms, dtype=torch.int64), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the spherical harmonics encoding.

        This method evaluates the preselected real spherical harmonic basis
        functions at the input angular coordinates and returns the first
        ``num_atoms`` coefficients in standard ordering.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(..., 2)`` containing spherical angles
            ``(theta, phi)`` in radians.

        Returns
        -------
        torch.Tensor
            Real spherical harmonics evaluated at the input angles,
            with shape ``(..., num_atoms)``.

        Raises
        ------
        ValueError
            If ``x.shape[-1] != 2``.
        """
        assert x.size(-1) == 2, "Input dim must be (theta, phi)"
        return PE.sph_harm(x, self.l_list, self.m_list)


class HerglotzPE(nn.Module):
    r"""
    Herglotz positional encoding with learnable phase and magnitude.

    This module implements a real-valued Herglotz-type feature map defined on
    Cartesian coordinates :math:`x \in \mathbb{R}^3`.

    Each atom :math:`k` is defined by two orthonormal vectors
    :math:`a_{k, \Re}, a_{k, \Im} \in \mathbb{R}^3`, forming an
    implicit complex direction
    :math:`a_k = a_{k, \Re} + i\,a_{k, \Im}`.

    For an input point :math:`x`, we compute the projections

    .. math::
        u_k = \langle x, a_{k, \Re} \rangle, \qquad
        v_k = \langle x, a_{k, \Im} \rangle.

    Each atom is parameterized by learnable parameter :math:`\sigma_k` with :math:`\sigma_k \sim \mathcal{U}(0, L_{\text{init}})`.

    The Herglotz feature associated with atom :math:`k` is defined in closed form as

    .. math::
        \psi^{\mathrm{H}}_k(x)
        = \frac{1}{1 + 2L_{\text{init}}} e^{\rho_k (u_k - 1)}
        \Bigl[
            (1 + 2\rho_k u_k)\cos(\rho_k v_k)
            - (2\rho_k v_k)\sin(\rho_k v_k)
        \Bigr].

    Optionally, a learnable quaternion rotation may be applied to all atoms
    before evaluation, allowing the encoding to learn a global orientation.

    Parameters
    ----------
    num_atoms: int
        Number of Herglotz atoms (output features).
    L_init: int
        Upper bound used to initialize the magnitude parameters
        :math:`\sigma_k \sim \mathcal{U}(0, L_{\mathrm{init}})`.
    rot: bool, optional
        If ``True``, applies a learnable quaternion rotation to all atoms.
        Default = ``False``

    Attributes
    ----------
    num_atoms : int

    L_init : int

    rot : bool

    qrot : Optional[torch.Tensor]
        Learnable quaternions for rotations.
    
    A_real0 : torch.Tensor
        Real part of Herglotz vectors

    A_imag0 : torch.Tensor
        Imaginary part of Herglotz vectors

    inv_const : float
        Constant prefactor

    Notes
    -----
    This module is **Cartesian-only**. If your data is given in spherical coordinates :math:`(\theta,\phi)`,
    use a wrapper to convert inputs to Cartesian coordinates before applying this encoding.
    """

    def __init__(self, num_atoms: int, L_init: int, rot: bool = False) -> None:

        super().__init__()
        self.num_atoms = num_atoms
        self.L_init = L_init
        self.rot = rot

        self.sigmas = nn.Parameter(torch.empty(self.num_atoms))
        self.register_buffer("A_real0", torch.empty(self.num_atoms, 3))
        self.register_buffer("A_imag0", torch.empty(self.num_atoms, 3))

        if rot:
            self.qrot = nn.Parameter(torch.empty(num_atoms, 4))
        else:
            self.register_parameter("qrot", None)

        inv_const = 1.0 / (1.0 + 2 * self.L_init)
        self.register_buffer(
            "inv_const",
            torch.tensor(inv_const),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(
        self,
    ) -> None:
        with torch.no_grad():
            aR, aI = self._generate_atoms(
                self.num_atoms, device=self.A_real0.device, dtype=self.A_real0.dtype
            )
            self.A_real0.copy_(aR)
            self.A_imag0.copy_(aI)
            nn.init.uniform_(self.sigmas, 0, self.L_init)

            if self.qrot is not None:
                self.qrot.zero_()
                self.qrot[:, 0] = 1.0

    @staticmethod
    def _generate_atoms(
        num_atoms: int, device=None, dtype=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        aI = torch.randn(num_atoms, 3, device=device, dtype=dtype)
        aR = torch.randn(num_atoms, 3, device=device, dtype=dtype)

        aR /= torch.norm(aR, dim=1, keepdim=True).clamp(1e-12)
        aI -= torch.sum(aI * aR, dim=1, keepdim=True) * aR
        aI /= torch.norm(aI, dim=1, keepdim=True).clamp(1e-12)

        return aR, aI

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Herglotz positional encoding.

        This method computes the closed-form Herglotz feature map by projecting
        Cartesian input points onto the learned atom directions, applying the
        learned phase and magnitude parameters.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape ``(..., 3)`` containing Cartesian coordinates.

        Returns
        -------
        torch.Tensor
            Herglotz features evaluated at the input points,
            with shape ``(..., num_atoms)``.

        Raises
        ------
        ValueError
            If ``x.shape[-1] != 3``.
        """

        if x.shape[-1] != 3:
            raise ValueError(
                f"HerglotzPE(coord='cartesian') expects x[...,3]=(x,y,z), got {x.shape}"
            )
        return PE.herglotz(
            x,
            self.A_real0,
            self.A_imag0,
            self.sigmas,
            self.inv_const,
            self.qrot,
        )


class FourierPE(nn.Module):
    r"""
    Learned Fourier positional encoding.

    This module implements a learnable sinusoidal feature map of the form

    .. math::
        \psi^{\mathrm{F}}(x) = \sin\bigl(\omega_0 (x \Omega^\top + b)\bigr),

    where:
    - :math:`W \in \mathbb{R}^{N \times d}` is a learnable weight matrix,
    - :math:`b \in \mathbb{R}^N` is an optional learnable bias,
    - :math:`\omega_0 > 0` is a fixed frequency scaling factor.

    This corresponds to a standard Fourier-feature embedding with learned
    frequencies.

    Parameters
    ----------
    num_atoms: int
        Number of output features.
    input_dim: int
        Dimension :math:`d` of the input space.
    bias: bool, optional
        Whether to include a learnable bias term :math:`b`.
        Default = ``True``
    omega0: float, optional
        Frequency scaling factor :math:`\omega_0`.
        Default = ``1.0``



    Attributes
    ----------
    num_atoms : int

    input_dim : int

    omega0 : float 

    Omega : torch.Tensor
        Learnable weights
    
    bias : Optional[torch.Tensor]
        Learnable bias
    """

    def __init__(
        self,
        num_atoms: int,
        input_dim: int = 3,
        bias: bool = True,
        omega0: float = 1.0,
    ) -> None:

        super().__init__()

        self.num_atoms = num_atoms
        self.input_dim = input_dim

        self.omega0 = omega0
        self.Omega = nn.Parameter(torch.empty(num_atoms, input_dim))
        self.bias = nn.Parameter(torch.empty(self.num_atoms)) if bias else None

        self.reset_parameters()

    def reset_parameters(
        self,
    ):
        with torch.no_grad():
            nn.init.uniform_(self.Omega, -1 / self.input_dim, 1 / self.input_dim)
            if self.bias is not None:
                bound = 1 / math.sqrt(self.input_dim)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the learned Fourier positional encoding.

        This method applies a learned linear projection of the input coordinates,
        followed by sinusoidal activation with fixed frequency scaling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(..., input_dim)``.

        Returns
        -------
        torch.Tensor
            Learned Fourier features with shape ``(..., num_atoms)``.
        """
        return PE.fourier(x, self.Omega, self.omega0, self.bias)
