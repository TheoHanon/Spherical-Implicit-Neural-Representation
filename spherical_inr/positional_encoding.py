from __future__ import annotations

import torch
import torch.nn as nn
import math

from . import functional as PE
from .third_party.locationencoder.sh import SH

from typing import Tuple
from enum import Enum


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
        \psi(x) =
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
    num_atoms:
        Number of spherical harmonic basis functions returned.

    Input
    -----
    x:
        Tensor of shape ``(..., 2)`` containing :math:`(\theta, \phi)`.

    Output
    ------
    Tensor of shape ``(..., num_atoms)``.
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
        assert x.size(-1) == 2, "Input dim must be (theta, phi)"
        return PE.sph_harm(x, self.l_list, self.m_list)


class HerglotzPE(nn.Module):
    r"""
    Herglotz positional encoding on Cartesian coordinates.

    This module implements a Cartesian Herglotz-type feature map.
    It expects inputs

    .. math::
        x \in \mathbb{R}^3

    and produces ``num_atoms`` real-valued features.

    Each atom :math:`k` is defined by a complex vector

    .. math::
        a_k = a_k^{\mathrm{R}} + i\,a_k^{\mathrm{I}} \in \mathbb{C}^3,

    where the real and imaginary parts are orthonormal vectors in
    :math:`\mathbb{R}^3`.
    For an input point :math:`x`, we form the complex projection

    .. math::
        z_k(x)
        = \langle x, a_k^{\mathrm{R}} \rangle
        + i\,\langle x, a_k^{\mathrm{I}} \rangle.

    The complex Herglotz feature is then defined as

    .. math::
        h_k(x)
        = C \,\bigl(1 + 2\,\sigma_k\,z_k(x)\bigr)
          \exp\bigl(\sigma_k (z_k(x) - 1)\bigr),

    where:
    - :math:`\sigma_k > 0` is a learnable scalar parameter,
    - :math:`C = \frac{1}{1 + 2L_{\text{init}}}` is a fixed normalization constant.

    The final real-valued encoding is obtained by mixing the real and
    imaginary parts with learnable weights

    .. math::
        \psi_k(x)
        = w_k^{\mathrm{R}}\,\Re(h_k(x))
        + w_k^{\mathrm{I}}\,\Im(h_k(x)),

    where :math:`w_k \in \mathbb{R}^2` is a learned mixing vector.

    Parameters
    ----------
    num_atoms:
        Number of Herglotz atoms (output features).
    L_init:
        Upper bound used to initialize
        :math:`\sigma_k \sim \mathcal{U}(0, L_{\text{init}})`.

    Input
    -----
    x:
        Tensor of shape ``(..., 3)`` containing Cartesian coordinates.

    Output
    ------
    Tensor of shape ``(..., num_atoms)``.

    Notes
    -----
    This module is **Cartesian-only**. If your data is given as spherical
    angles :math:`(\theta,\phi)`, convert it to Cartesian coordinates using
    a separate wrapper before calling this module.
    """

    def __init__(self, num_atoms: int, L_init: int) -> None:

        super().__init__()
        self.num_atoms = num_atoms
        self.L_init = L_init

        self.sigmas_mod = nn.Parameter(torch.empty(self.num_atoms))
        self.sigmas_arg = nn.Parameter(torch.empty(self.num_atoms))

        self.register_buffer("A_real", torch.empty(self.num_atoms, 3))
        self.register_buffer("A_imag", torch.empty(self.num_atoms, 3))

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
                self.num_atoms, device=self.A_real.device, dtype=self.A_real.dtype
            )
            self.A_real.copy_(aR)
            self.A_imag.copy_(aI)
            nn.init.uniform_(self.sigmas_mod, 0, self.L_init)
            nn.init.uniform_(self.sigmas_arg, 0, 2 * math.pi)

    @staticmethod
    def _generate_atoms(
        num_atoms: int, device=None, dtype=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        a_I = torch.randn(num_atoms, 3, device=device, dtype=dtype)
        a_R = torch.randn(num_atoms, 3, device=device, dtype=dtype)

        a_R /= torch.norm(a_R, dim=1, keepdim=True).clamp(1e-12)
        a_I -= torch.sum(a_I * a_R, dim=1, keepdim=True) * a_R
        a_I /= torch.norm(a_I, dim=1, keepdim=True).clamp(1e-12)

        return a_R, a_I

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.shape[-1] != 3:
            raise ValueError(
                f"HerglotzPE(coord='cartesian') expects x[...,3]=(x,y,z), got {x.shape}"
            )
        return PE.herglotz(
            x,
            self.A_real,
            self.A_imag,
            self.sigmas_mod,
            self.sigmas_arg,
            self.inv_const,
        )


class FourierPE(nn.Module):
    r"""
    Learned Fourier positional encoding.

    This module implements a learnable sinusoidal feature map of the form

    .. math::
        \psi(x) = \sin\bigl(\omega_0 (x \Omega^\top + b)\bigr),

    where:
    - :math:`W \in \mathbb{R}^{N \times d}` is a learnable weight matrix,
    - :math:`b \in \mathbb{R}^N` is an optional learnable bias,
    - :math:`\omega_0 > 0` is a fixed frequency scaling factor.

    This corresponds to a standard Fourier-feature embedding with learned
    frequencies.

    Parameters
    ----------
    num_atoms:
        Number of output features.
    input_dim:
        Dimension :math:`d` of the input space.
    bias:
        Whether to include a learnable bias term :math:`b`.
    omega0:
        Frequency scaling factor :math:`\omega_0`.

    Input
    -----
    x:
        Tensor of shape ``(..., input_dim)``.

    Output
    ------
    Tensor of shape ``(..., num_atoms)``.
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
        return PE.fourier(x, self.Omega, self.omega0, self.bias)
