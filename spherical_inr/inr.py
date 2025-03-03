import torch
import torch.nn as nn

from .transforms import *
from .positional_encoding import *
from .mlp import *

from typing import Optional, List


class INR(nn.Module):

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
        r"""Implicit Neural Representation (INR).

        This module implements an implicit neural representation by combining a positional encoding and a multilayer
        perceptron (MLP). The input :math:`x` is first transformed by a chosen positional encoding :math:`\text{PE}`,
        and then processed by the MLP. In summary, the representation is computed as

        .. math::
            \text{INR}(x) = \text{MLP}\Bigl(\text{PE}(x)\Bigr).

        Parameters:
            input_dim (int): Dimensionality of the input.
            output_dim (int): Dimensionality of the output.
            inr_sizes (List[int]): A list specifying the number of atoms for the positional encoding as its first element,
                followed by the sizes of the hidden layers for the MLP.
            pe (str, optional): Type of positional encoding to use (default: "herglotz").
            pe_kwards (Optional[dict], optional): Additional keyword arguments for configuring the positional encoding module.
            activation (str, optional): Activation function to use in the MLP (default: "relu").
            activation_kwargs (dict, optional): Additional keyword arguments for the activation function.
            bias (bool, optional): If True, includes bias terms in the network layers (default: False).

        Methods:
            forward(x: torch.Tensor) -> torch.Tensor:
                Computes the forward pass by applying the positional encoding followed by the MLP.
        """

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

    A neural network that combines a spherical-to-Cartesian coordinate transform, a Herglotz positional encoding,
    and a sine-activated MLP. This network is designed for inputs defined on the 2-sphere (S²) and accepts
    coordinates in the form :math:`(\theta, \phi)`. The computation is summarized by

    .. math::
        x_{\text{cart}} = \text{sph2\_to\_cart3}(x), \qquad
        E(x) = \text{RegularHerglotzPE}\Bigl(x_{\text{cart}}\Bigr), \qquad
        \text{HerglotzNet}(x) = \text{SineMLP}\Bigl(E(x)\Bigr).

    Attributes:
        input_dim (int): Dimensionality of the input (must be 1 or 2).
        output_dim (int): Dimensionality of the output.
        num_atoms (int): Number of encoding atoms (derived from the first element of inr_sizes).
        mlp_sizes (List[int]): List defining the hidden layer sizes of the MLP.
        bias (bool): Whether bias terms are included in the layers.
        omega0 (float): Frequency factor used in the sine activations.
        seed (Optional[int]): Seed for reproducibility.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Transforms spherical coordinates to Cartesian, applies the positional encoding, and passes the result
            through the sine-activated MLP.
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

    A neural network that integrates a spherical-to-Cartesian transform tailored for solid harmonics,
    a Herglotz positional encoding (regular or irregular), and a sine-activated MLP. The network
    accepts input in a spherical coordinate system and processes it as follows:

    .. math::
        x_{\text{cart}} = \text{rsph2\_to\_cart3}(x), \qquad
        E(x) = \text{PE}\Bigl(x_{\text{cart}}\Bigr) \quad \text{with} \quad
        \text{PE} =
        \begin{cases}
        \text{RegularHerglotzPE}, & \text{if type = "R"}, \\
        \text{IregularHerglotzPE}, & \text{if type = "I"},
        \end{cases}
        \qquad
        \text{SolidHerlotzNet}(x) = \text{SineMLP}\Bigl(E(x)\Bigr).

    Parameters:
        output_dim (int): Dimensionality of the output.
        inr_sizes (List[int]): A list specifying the number of atoms for the positional encoding and the hidden sizes for the MLP.
        bias (bool, optional): If True, includes bias terms in the network layers (default: True).
        omega0 (float, optional): Frequency factor applied to both the positional encoding and the MLP (default: 1.0).
        type (str, optional): Specifies the type of Herglotz positional encoding: "R" for regular or "I" for irregular.
        seed (Optional[int], optional): Seed for reproducibility.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Applies a spherical-to-Cartesian transform, then the positional encoding, and finally the sine-activated MLP.

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

    A neural network that employs a Fourier-based positional encoding and a sine-activated MLP following the SIREN architecture.
    For an input :math:`x`, the computation is given by

    .. math::
        E(x) = \text{FourierPE}(x), \qquad
        \text{SirenNet}(x) = \text{SineMLP}\Bigl(E(x)\Bigr).

    Parameters:
        input_dim (int): Dimensionality of the input.
        output_dim (int): Dimensionality of the output.
        inr_sizes (List[int]): A list specifying the number of atoms for the positional encoding and the hidden sizes for the MLP.
        bias (bool, optional): If True, includes bias terms in the network layers (default: True).
        first_omega0 (float, optional): Frequency factor for the Fourier positional encoding (default: 1.0).
        hidden_omega0 (float, optional): Frequency factor for the sine activation in the MLP (default: 1.0).

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Computes the forward pass by applying the Fourier positional encoding followed by the sine-activated MLP.
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

    A hybrid network that integrates a Herglotz positional encoding (either regular or irregular) with a sine-activated MLP.
    The network computes the output as

    .. math::
        E(x) = \text{PE}(x) \quad \text{with} \quad
        \text{PE} =
        \begin{cases}
        \text{RegularHerglotzPE}, & \text{if type = "R"}, \\
        \text{IregularHerglotzPE}, & \text{if type = "I"},
        \end{cases}
        \qquad
        \text{HSNet}(x) = \text{SineMLP}\Bigl(E(x)\Bigr).

    Parameters:
        input_dim (int): Dimensionality of the input.
        output_dim (int): Dimensionality of the output.
        inr_sizes (List[int]): A list specifying the number of atoms for the positional encoding and the hidden sizes for the MLP.
        bias (bool, optional): If True, includes bias terms in the network layers (default: True).
        first_omega0 (float, optional): Frequency factor for the positional encoding (default: 1.0).
        hidden_omega0 (float, optional): Frequency factor for the sine activation in the MLP (default: 1.0).
        type (str, optional): Specifies the type of Herglotz positional encoding: "R" for regular or "I" for irregular.
        seed (Optional[int], optional): Seed for reproducibility.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Applies the positional encoding to the input and passes the result through the sine-activated MLP.

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
