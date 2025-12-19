import torch
import torch.nn as nn

from . import _coords as T

from .positional_encoding import (
    HerglotzPE,
    FourierPE,
    SphericalHarmonicsPE,
)

from .mlp import (
    SineMLP,
)

from ._interfaces import PositionalEncoding, MLP

from typing import List


__all__ = ["INR", "SirenNet", "HerglotzNet", "SphericalSirenNet"]


class INR(nn.Module):

    def __init__(self, positional_encoding: PositionalEncoding, mlp: MLP):
        super().__init__()

        if not isinstance(positional_encoding, PositionalEncoding):
            raise TypeError(
                "`pe` must implement the PositionalEncoding interface: `.out_dim` and `forward/__call__`."
            )
        if not isinstance(mlp, MLP):
            raise TypeError(
                "`mlp` must implement the BackboneMLP interface: `.in_dim`, `.out_dim`, and `forward/__call__`."
            )

        if int(mlp.in_dim) != int(positional_encoding.out_dim):
            raise ValueError(
                f"Incompatible PE/MLP: mlp.in_dim={mlp.in_dim} must equal pe.out_dim={positional_encoding.out_dim}."
            )

        self.pe = positional_encoding
        self.mlp = mlp

    def forward(self, x: torch.Tensor):
        return self.mlp(self.pe(x))


class SirenNet(nn.Module):
    r"""
    SIREN on the 2-sphere with learned Fourier positional encoding.

    This network represents a function of spherical angles
    :math:`(\theta,\phi)` by applying a learned Fourier feature map directly
    to the angles, followed by a sine-activated multilayer perceptron:

    .. math::
        f(\theta,\phi) = \mathrm{MLP}_{\sin}\bigl(\psi_{\mathrm{Fourier}}(\theta,\phi)\bigr),

    with

    .. math::
        \psi_{\mathrm{Fourier}}(\theta,\phi)
        = \sin\!\bigl(\omega_0^{\mathrm{PE}}([\theta,\phi] W^\top + b)\bigr).

    No coordinate transformation is applied: the angles are treated as inputs
    in :math:`\mathbb{R}^2`.

    Parameters
    ----------
    num_atoms:
        Number of Fourier features (output channels of the positional encoding).
    mlp_sizes:
        Hidden-layer widths of the sine-activated MLP.
    output_dim:
        Dimensionality of the network output.
    bias:
        Whether to include bias terms in both the positional encoding and the MLP.
    omega0_pe:
        Frequency factor :math:`\omega_0^{\mathrm{PE}}` used in the Fourier encoding.
    omega0_mlp:
        Frequency factor :math:`\omega_0^{\mathrm{MLP}}` used in the sine activations
        of the MLP.
    input_dim:
        Dimensionality of the input space. Must be ``2`` for :math:`(\theta,\phi)`.

    Input
    -----
    x:
        Tensor of shape ``(..., 2)`` containing spherical angles
        :math:`(\theta,\phi)` in radians.

    Output
    ------
    Tensor of shape ``(..., output_dim)``.
    """

    def __init__(
        self,
        num_atoms: int,
        mlp_sizes: List[int],
        output_dim: int,
        *,
        bias: bool = True,
        omega0_pe: float = 30.0,
        omega0_mlp: float = 30.0,
    ):
        super().__init__()
        self.pe = FourierPE(num_atoms, input_dim=2, bias=bias, omega0=omega0_pe)
        self.mlp = SineMLP(
            input_features=num_atoms,
            output_features=output_dim,
            hidden_sizes=mlp_sizes,
            bias=bias,
            omega0=omega0_mlp,
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(self.pe(x))


class HerglotzNet(nn.Module):
    r"""
    Herglotz-Net on the 2-sphere.

    This network represents functions defined on the unit sphere by combining
    a Herglotz positional encoding with a sine-activated multilayer
    perceptron.

    Inputs are provided in spherical coordinates
    :math:`(\theta,\phi)` and internally converted to Cartesian coordinates
    on the unit sphere,

    .. math::
        x(\theta,\phi)
        = (\sin\theta\cos\phi,\; \sin\theta\sin\phi,\; \cos\theta).

    The overall mapping implemented by the network is

    .. math::
        f(\theta,\phi)
        = \mathrm{MLP}_{\sin}
        \Bigl(
            \psi_{\mathrm{H}}\bigl(x(\theta,\phi)\bigr)
        \Bigr),

    where :math:`\psi_{\mathrm{H}}` is the Cartesian Herglotz positional encoding
    defined in :class:`HerglotzPE`.

    Parameters
    ----------
    num_atoms:
        Number of Herglotz atoms (output channels of the positional encoding).
    mlp_sizes:
        Hidden-layer widths of the sine-activated MLP.
    output_dim:
        Dimensionality of the network output.
    bias:
        Whether to include bias terms in the MLP.
    L_init:
        Upper bound used to initialize the Herglotz magnitude parameters
        :math:`\rho_k`.
    omega0_mlp:
        Frequency factor :math:`\omega_0^{\mathrm{MLP}}` used in the sine
        activations of the MLP.
    rot:
        If ``True``, enables a learnable quaternion rotation in the
        Herglotz positional encoding.

    Input
    -----
    x:
        Tensor of shape ``(..., 2)`` containing spherical angles
        :math:`(\theta,\phi)` in radians.

    Output
    ------
    Tensor of shape ``(..., output_dim)``.

    """

    def __init__(
        self,
        num_atoms: int,
        mlp_sizes: List[int],
        output_dim: int,
        *,
        bias: bool = True,
        L_init: int = 15,
        omega0_mlp: float = 1.0,
        rot: bool = False,
    ):

        super().__init__()
        self.pe = HerglotzPE(num_atoms=num_atoms, L_init=L_init, rot=rot)
        self.mlp = SineMLP(
            input_features=num_atoms,
            output_features=output_dim,
            hidden_sizes=mlp_sizes,
            bias=bias,
            omega0=omega0_mlp,
        )

    def forward(self, x: torch.Tensor):
        if x.shape[-1] != 2:
            raise ValueError(
                f"Expected input shape (..., 2) for spherical coordinates (θ, φ), but got {x.shape}."
            )
        x_r3 = T.tp_to_r3(x)
        return self.mlp(self.pe(x_r3))


class SphericalSirenNet(nn.Module):
    r"""
    Spherical-SIREN on the 2-sphere using real spherical harmonics.

    This network represents functions defined on the sphere by first encoding
    angular coordinates :math:`(\theta,\phi)` using real spherical harmonics,
    then applying a sine-activated multilayer perceptron.

    The mapping is

    .. math::
        f(\theta,\phi)
        = \mathrm{MLP}_{\sin}\bigl(\psi_{\mathrm{SH}}(\theta,\phi)\bigr),

    where :math:`\psi_{\mathrm{SH}}` denotes the real spherical harmonics
    positional encoding.

    Parameters
    ----------
    num_atoms:
        Number of spherical harmonic basis functions retained
        (i.e. the first ``num_atoms`` channels in the standard
        :math:`(\ell,m)` ordering).
    mlp_sizes:
        Hidden-layer widths of the sine-activated MLP.
    output_dim:
        Dimensionality of the network output.
    bias:
        Whether to include bias terms in the MLP.
    omega0_mlp:
        Frequency factor :math:`\omega_0^{\mathrm{MLP}}` used in the sine activations
        of the MLP.

    Input
    -----
    x:
        Tensor of shape ``(..., 2)`` containing spherical angles
        :math:`(\theta,\phi)` in radians.

    Output
    ------
    Tensor of shape ``(..., output_dim)``.
    """

    def __init__(
        self,
        num_atoms: int,
        mlp_sizes: List[int],
        output_dim: int,
        *,
        bias: bool = True,
        omega0_mlp: float = 1.0,
    ) -> None:

        super().__init__()

        self.pe = SphericalHarmonicsPE(num_atoms)
        self.mlp = SineMLP(
            input_features=num_atoms,
            output_features=output_dim,
            hidden_sizes=mlp_sizes,
            bias=bias,
            omega0=omega0_mlp,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 2:
            raise ValueError(
                f"Expected input shape (..., 2) for spherical coordinates (θ, φ), but got {x.shape}."
            )
        return self.mlp(self.pe(x))
