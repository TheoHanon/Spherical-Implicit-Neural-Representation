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
    """
    A neural network that combines a spherical-to-Cartesian transform,
    a Herglotz positional encoding, and a sine-activated MLP. The network is defined on the S2 sphere and only accepts (θ, φ) coordinates.

    Attributes:
        input_dim (int): Dimensionality of the input (must be 1 or 2).
        output_dim (int): Dimensionality of the output.
        num_atoms (int): Number of atoms/features for encoding.
        mlp_sizes (List[int]): List defining the hidden layer sizes of the MLP.
        bias (bool): Whether to include bias in the layers.
        omega0 (float): Frequency factor used in the sine activations.
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
