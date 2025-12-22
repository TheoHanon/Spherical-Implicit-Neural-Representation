import torch
import torch.nn as nn

from . import coords as T

from .positional_encoding import (
    HerglotzPE,
    FourierPE,
    SphericalHarmonicsPE,
)

from .mlp import (
    SineMLP,
)

from typing import List


__all__ = ["INR", "SirenNet", "HerglotzNet", "SphericalSirenNet"]


class INR(nn.Module):
    r"""
    Composable implicit neural representation.

    This class represents an implicit function as the composition

    .. math::
        f(x) = \mathrm{MLP}(\psi(x)),

    where :math:`\psi` is a positional encoding and the MLP is a pointwise
    neural network.

    Parameters
    ----------
    positional_encoding : PositionalEncoding
        Positional encoding module :math:`\psi`.
        Must expose an ``out_dim`` attribute and be callable on a tensor.
    mlp : MLP
        Backbone network applied to the encoded features.
        Must expose ``in_dim`` and ``out_dim`` attributes and be callable.
    """

    def __init__(self, positional_encoding: nn.Module, mlp: nn.Module):
        super().__init__()
        self.pe = positional_encoding
        self.mlp = mlp

    def forward(self, x: torch.Tensor):
        r"""
        Evaluate the implicit neural representation.

        This method applies the positional encoding followed by the MLP backbone.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor passed to the positional encoding.
            Shape and interpretation depend on the chosen encoding ``pe``.

        Returns
        -------
        torch.Tensor
            Output of the MLP applied to the encoded input.
            Shape ``(..., mlp.out_dim)``.

        Notes
        -----
        The method doesn't check whether the dimensions between the backbone and the positional encodings are consistent.
        """

        return self.mlp(self.pe(x))


class SirenNet(nn.Module):
    r"""
    SIREN on the 2-sphere with learned Fourier positional encoding.

    This network represents a function of spherical angles
    :math:`(\theta,\phi)` by applying a learned Fourier feature map directly
    to the angles, followed by a sine-activated multilayer perceptron:

    .. math::
        f(\theta,\phi) = \operatorname{SineMLP}\bigl(\psi^{\mathrm{F}}(\theta,\phi)\bigr),

    where :math:`\psi^{\mathrm{F}}` is the Fourier positional encoding
    defined in :class:`FourierPE`.

    No coordinate transformation is applied: the angles are treated as inputs
    in :math:`\mathbb{R}^2`.

    Parameters
    ----------
    num_atoms: int
        Number of Fourier features (output channels of the positional encoding).
    mlp_sizes: list[int]
        Hidden-layer widths of the sine-activated MLP.
    output_dim: int
        Dimensionality of the network output.
    bias: bool, optional
        Whether to include bias terms in both the positional encoding and the MLP.
        Default = ``True``
    omega0_pe: float, optional
        Frequency factor :math:`\omega_0^{\mathrm{PE}}` used in the Fourier encoding.
        Default = ``30.0``
    omega0_mlp: float, optional
        Frequency factor :math:`\omega_0^{\mathrm{MLP}}` used in the sine activations
        of the MLP.
        Default = ``30.0``
    input_dim: int, optional
        Dimensionality of the input space. Must be ``2`` for :math:`(\theta,\phi)`.
        Default = ``2``
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
        r"""
        Evaluate the SIREN on spherical angles.

        The input angles :math:`(\theta,\phi)` are encoded using learned Fourier
        features and then processed by a sine-activated MLP.

        Parameters
        ----------
        x: torch.Tensor
            Tensor of shape ``(..., 2)`` containing spherical angles
            :math:`(\theta,\phi)` in radians.

        Returns
        -------
        torch.Tensor
            Network output of shape ``(..., output_dim)``.
        """

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
        = \operatorname{SineMLP}
        \Bigl(
            \psi^{H}\bigl(x(\theta,\phi)\bigr)
        \Bigr),

    where :math:`\psi^{\mathrm{H}}` is the Cartesian Herglotz positional encoding
    defined in :class:`HerglotzPE`.

    Parameters
    ----------
    num_atoms: int
        Number of Herglotz atoms (output channels of the positional encoding).
    mlp_sizes: list[int]
        Hidden-layer widths of the sine-activated MLP.
    output_dim: int
        Dimensionality of the network output.
    bias: bool, optional
        Whether to include bias terms in the MLP.
        Default = ``True``
    L_init: int, optional
        Upper bound used to initialize the Herglotz magnitude parameters
        :math:`\rho_k`.
        Default = ``15``
    omega0_mlp: float, optional
        Frequency factor :math:`\omega_0^{\mathrm{MLP}}` used in the sine
        activations of the MLP.
        Default = ``1.0``
    rot: bool, optional
        If ``True``, enables a learnable quaternion rotation in the
        Herglotz positional encoding.
        Default = ``False``

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
        r"""
        Evaluate the Herglotz-based SIREN on the 2-sphere.

        The input angles :math:`(\theta,\phi)` are first mapped to Cartesian
        coordinates on the unit sphere, then encoded using the Cartesian Herglotz positional encoding and
        processed by a sine-activated MLP.

        Parameters
        ----------
        x: torch.Tensor
            Tensor of shape ``(..., 2)`` containing spherical angles
            :math:`(\theta,\phi)` in radians.

        Returns
        -------
        torch.Tensor
            Network output of shape ``(..., output_dim)``.

        Raises
        ------
        ValueError
            If ``x.shape[-1] != 2``.
        """

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
        = \operatorname{SineMLP}\bigl(\psi^{\mathrm{SH}}(\theta,\phi)\bigr),

    where :math:`\psi^{\mathrm{SH}}` denotes the real spherical harmonics
    positional encoding.

    Parameters
    ----------
    num_atoms: int
        Number of spherical harmonic basis functions retained
        (i.e. the first ``num_atoms`` channels in the standard
        :math:`(\ell,m)` ordering).
    mlp_sizes: list[int]
        Hidden-layer widths of the sine-activated MLP.
    output_dim: int
        Dimensionality of the network output.
    bias: bool, optional
        Whether to include bias terms in the MLP.
    omega0_mlp: float, optional
        Frequency factor :math:`\omega_0^{\mathrm{MLP}}` used in the sine activations
        of the MLP.
        Default : ``1.0``.

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
        r"""
        Evaluate the spherical-harmonics SIREN.

        The input angles :math:`(\theta,\phi)` are encoded using real spherical
        harmonics and then processed by a sine-activated MLP.

        Parameters
        ----------
        x: torch.Tensor
            Tensor of shape ``(..., 2)`` containing spherical angles
            :math:`(\theta,\phi)` in radians.

        Returns
        -------
        torch.Tensor
            Network output of shape ``(..., output_dim)``.

        Raises
        ------
        ValueError
            If ``x.shape[-1] != 2``.
        """

        if x.shape[-1] != 2:
            raise ValueError(
                f"Expected input shape (..., 2) for spherical coordinates (θ, φ), but got {x.shape}."
            )
        return self.mlp(self.pe(x))
