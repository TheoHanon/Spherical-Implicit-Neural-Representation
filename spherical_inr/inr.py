import torch
import torch.nn as nn

from .transforms import *
from .positional_encoding import *
from .mlp import *

from typing import Optional, List


class INR(nn.Module):
    r"""Implicit Neural Representation (INR).

    Implements an implicit neural representation where an input :math:`x \in \mathbb{R}^{d}` is first mapped to a
    high-dimensional feature space via a positional encoding :math:`\psi(x)` and then processed by a multilayer
    perceptron (MLP). In mathematical form, the representation is defined as

    .. math::
        \text{INR}(x) = \text{MLP}\Bigl(\psi(x)\Bigr).

    Parameters:
        input_dim (int): Dimensionality of the input.
        output_dim (int): Dimensionality of the output.
        inr_sizes (List[int]): A list where the first element specifies the number of atoms for the positional encoding
            and subsequent elements define the hidden layer sizes of the MLP.
        pe (str, optional): Identifier for the type of positional encoding (default: "herglotz").
        pe_kwards (Optional[dict], optional): Additional keyword arguments for configuring the positional encoding.
        activation (str, optional): Activation function used in the MLP (default: "relu").
        activation_kwargs (dict, optional): Additional keyword arguments for the activation function.
        bias (bool, optional): If True, includes bias terms in the network layers (default: False).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        inr_sizes: List[int],
        pe: str = "herglotz",
        pe_kwards: Optional[dict] = None,
        activation: str = "relu",
        activation_kwargs: dict = {},
        bias: bool = False,
    ) -> None:

        super(INR, self).__init__()

        self.pe = get_positional_encoding(
            pe,
            **{
                "num_atoms": inr_sizes[0],
                "input_dim": input_dim,
                "bias": bias,
                **(pe_kwards or {}),
            },
        )

        self.mlp = MLP(
            input_features=inr_sizes[0],
            output_features=output_dim,
            hidden_sizes=inr_sizes[1:],
            bias=bias,
            activation=activation,
            activation_kwargs=activation_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pe(x)
        x = self.mlp(x)

        return x


class HerglotzNet(nn.Module):
    r"""HerglotzNet.

    A neural network designed for inputs defined on the 2-sphere. This network first converts the input
    (spherical coordinates) to Cartesian coordinates, then computes a Herglotz positional encoding :math:`\psi(x)`
    and finally processes the result through a sine-activated MLP. In summary, if :math:`x(\theta, \varphi)` denotes the
    Cartesian coordinates derived from :math:`x`, then

    .. math::
        \text{HerglotzNet}(x) = \text{SineMLP}\Bigl(\psi(x(\theta, \varphi))\Bigr).

    Attributes:
        input_dim (int): Dimensionality of the input (typically 1 or 2 for spherical coordinates).
        output_dim (int): Dimensionality of the output.
        num_atoms (int): Number of encoding atoms (derived from the first element of inr_sizes).
        mlp_sizes (List[int]): Hidden layer sizes of the MLP.
        bias (bool): Whether bias terms are included in the network layers.
        omega0 (float): Frequency factor used in the encoding and sine activation.
        seed (Optional[int]): Seed for reproducibility.
    """

    def __init__(
        self,
        output_dim: int,
        inr_sizes: List[int],
        bias: bool = True,
        pe_omega0: float = 1.0,
        hidden_omega0: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:

        super(HerglotzNet, self).__init__()

        self.pe = RegularHerglotzPE(
            num_atoms=inr_sizes[0],
            input_dim=3,
            bias=bias,
            omega0=pe_omega0,
            seed=seed,
        )

        self.mlp = SineMLP(
            input_features=inr_sizes[0],
            output_features=output_dim,
            hidden_sizes=inr_sizes[1:],
            bias=bias,
            omega0=hidden_omega0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = sph2_to_cart3(x)
        x = self.pe(x)
        x = self.mlp(x)

        return x


class SolidHerlotzNet(nn.Module):
    r"""SolidHerlotzNet.

    A neural network that integrates a spherical-to-Cartesian coordinate transformation tailored for solid harmonics,
    a Herglotz positional encoding (which can be either regular or irregular), and a sine-activated MLP.
    The network accepts input in a spherical coordinate system and computes its representation as

    .. math::
        \text{SolidHerlotzNet}(x) = \text{SineMLP}\Bigl(\psi(x(r, \theta, \varphi))\Bigr),

    where :math:`x(r, \theta, \varphi)` denotes the Cartesian coordinates derived from the spherical input.
    The type of positional encoding is chosen via a parameter ("R" for regular, "I" for irregular).

    Parameters:
        output_dim (int): Dimensionality of the output.
        inr_sizes (List[int]): A list where the first element specifies the number of atoms for the positional encoding and
            subsequent elements define the hidden layer sizes of the MLP.
        bias (bool, optional): If True, includes bias terms in the network layers (default: True).
        omega0 (float, optional): Frequency factor applied to both the positional encoding and the MLP (default: 1.0).
        type (str, optional): Specifies the type of Herglotz positional encoding ("R" for regular or "I" for irregular).
        seed (Optional[int], optional): Seed for reproducibility.

    Raises:
        ValueError: If the specified type is not "R" or "I".
    """

    def __init__(
        self,
        output_dim: int,
        inr_sizes: List[int],
        bias: bool = True,
        omega0: float = 1.0,
        type: str = "R",
        seed: Optional[int] = None,
    ) -> None:

        super(SolidHerlotzNet, self).__init__()

        if type not in ["R", "I"]:
            raise ValueError("Invalid type. Must be 'R' or 'I'.")

        self.pe = get_positional_encoding(
            "herglotz" if type == "R" else "irregular_herglotz",
            num_atoms=inr_sizes[0],
            input_dim=3,
            bias=bias,
            omega0=omega0,
            seed=seed,
        )

        self.mlp = SineMLP(
            input_features=inr_sizes[0],
            output_features=output_dim,
            hidden_sizes=inr_sizes[1:],
            bias=bias,
            omega0=omega0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = rsph2_to_cart3(x)
        x = self.pe(x)
        x = self.mlp(x)

        return x


class SirenNet(nn.Module):
    r"""SirenNet.

    A neural network that employs a Fourier-based positional encoding to compute the representation,
    followed by a sine-activated MLP as described in the SIREN architecture. For an input :math:`x`, the
    representation is computed as

    .. math::
        \text{SirenNet}(x) = \text{SineMLP}\Bigl(\psi(x)\Bigr),

    where the positional encoding :math:`\psi(x)` is obtained via a learnable linear mapping and a sinusoidal activation.

    Parameters:
        input_dim (int): Dimensionality of the input.
        output_dim (int): Dimensionality of the output.
        inr_sizes (List[int]): A list where the first element specifies the number of atoms for the positional encoding
            and subsequent elements define the hidden layer sizes of the MLP.
        bias (bool, optional): If True, includes bias terms in the network layers (default: True).
        first_omega0 (float, optional): Frequency factor for the Fourier positional encoding (default: 1.0).
        hidden_omega0 (float, optional): Frequency factor for the sine activation in the MLP (default: 1.0).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        inr_sizes: List[int],
        bias: bool = True,
        first_omega0: float = 1.0,
        hidden_omega0: float = 1.0,
    ) -> None:

        super(SirenNet, self).__init__()

        self.pe = FourierPE(
            num_atoms=inr_sizes[0], input_dim=input_dim, bias=bias, omega0=first_omega0
        )

        self.mlp = SineMLP(
            input_features=inr_sizes[0],
            output_features=output_dim,
            hidden_sizes=inr_sizes[1:],
            bias=bias,
            omega0=hidden_omega0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pe(x)
        x = self.mlp(x)

        return x


class HSNet(nn.Module):
    r"""HSNet.

    A hybrid network that combines a Herglotz positional encoding (either regular or irregular) with a sine-activated MLP.
    For an input :math:`x`, the network computes its representation as

    .. math::
        \text{HSNet}(x) = \text{SineMLP}\Bigl(\psi(x)\Bigr),

    with the choice of positional encoding determined by a parameter ("R" for regular, "I" for irregular).

    Parameters:
        input_dim (int): Dimensionality of the input.
        output_dim (int): Dimensionality of the output.
        inr_sizes (List[int]): A list where the first element specifies the number of atoms for the positional encoding and
            subsequent elements define the hidden layer sizes of the MLP.
        bias (bool, optional): If True, includes bias terms in the network layers (default: True).
        first_omega0 (float, optional): Frequency factor for the positional encoding (default: 1.0).
        hidden_omega0 (float, optional): Frequency factor for the sine activation in the MLP (default: 1.0).
        type (str, optional): Specifies the type of Herglotz positional encoding ("R" for regular or "I" for irregular).
        seed (Optional[int], optional): Seed for reproducibility.

    Raises:
        ValueError: If the specified type is not "R" or "I".
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        inr_sizes: List[int],
        bias: bool = True,
        first_omega0: float = 1.0,
        hidden_omega0: float = 1.0,
        type: str = "R",
        seed: Optional[int] = None,
    ) -> None:

        super(HSNet, self).__init__()

        if type not in ["R", "I"]:
            raise ValueError("Invalid type. Must be 'R' or 'I'.")

        self.pe = get_positional_encoding(
            "herglotz" if type == "R" else "irregular_herglotz",
            num_atoms=inr_sizes[0],
            input_dim=input_dim,
            bias=bias,
            omega0=first_omega0,
            seed=seed,
        )

        self.mlp = SineMLP(
            input_features=inr_sizes[0],
            output_features=output_dim,
            hidden_sizes=inr_sizes[1:],
            bias=bias,
            omega0=hidden_omega0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pe(x)
        x = self.mlp(x)

        return x
