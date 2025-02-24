import torch
import torch.nn as nn

from .transforms import *
from .positional_encoding import *
from .mlp import *

from typing import Optional, List


class HerglotzNet(nn.Module):
    """
    A neural network that combines a spherical-to-Cartesian transform,
    a Herglotz positional encoding, and a sine-activated MLP.

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
        input_dim: int,
        output_dim: int,
        num_atoms: int,
        mlp_sizes: List[int],
        bias: bool = True,
        omega0: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:

        if input_dim not in (1, 2):
            raise ValueError("The input dimension must be 1 or 2.")

        super(HerglotzNet, self).__init__()

        self.transform = sph2_to_cart3 if input_dim == 2 else sph1_to_cart2

        self.pe = HerglotzPE(
            num_atoms=num_atoms,
            input_dim=input_dim + 1,
            bias=bias,
            omega0=omega0,
            seed=seed,
        )

        self.mlp = SineMLP(
            input_features=num_atoms,
            output_features=output_dim,
            hidden_sizes=mlp_sizes,
            bias=bias,
            omega0=omega0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.transform(x)
        x = self.pe(x)
        x = self.mlp(x)

        return x


class SolidHerlotzNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_atoms: int,
        mlp_sizes: List[int],
        bias: bool = True,
        omega0: float = 1.0,
        type: str = "R",
        seed: Optional[int] = None,
    ) -> None:

        if input_dim not in (2, 3):
            raise ValueError("The input dimension must be 2 or 3.")

        super(SolidHerlotzNet, self).__init__()

        self.transform = rsph2_to_cart3 if input_dim == 3 else rsph1_to_cart2

        if type == "R":
            self.pe = HerglotzPE(
                num_atoms=num_atoms,
                input_dim=input_dim,
                bias=bias,
                omega0=omega0,
                seed=seed,
            )
        elif type == "I":
            self.pe = IregularHerglotzPE(
                num_atoms=num_atoms,
                input_dim=input_dim,
                bias=bias,
                omega0=omega0,
                seed=seed,
            )
        else:
            raise ValueError("Invalid type. Must be 'R' or 'I'.")

        self.mlp = SineMLP(
            input_features=num_atoms,
            output_features=output_dim,
            hidden_sizes=mlp_sizes,
            bias=bias,
            omega0=omega0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.transform(x)
        x = self.pe(x)
        x = self.mlp(x)

        return x


class SirenNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_atoms: int,
        mlp_sizes: List[int],
        bias: bool = True,
        first_omega0: float = 1.0,
        hidden_omega0: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:

        super(SirenNet, self).__init__()

        self.pe = FourierPE(
            num_atoms=num_atoms, input_dim=input_dim, bias=bias, omega0=first_omega0
        )

        self.mlp = SineMLP(
            input_features=num_atoms,
            output_features=output_dim,
            hidden_sizes=mlp_sizes,
            bias=bias,
            omega0=hidden_omega0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pe(x)
        x = self.mlp(x)

        return x
