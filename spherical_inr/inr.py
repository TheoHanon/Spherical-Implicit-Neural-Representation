import torch
import torch.nn as nn

from .transforms import *
from .positional_encoding import *
from .mlp import *

from typing import Optional, Type


class BaseINR(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_atoms: int,
        hidden_layers: int,
        hidden_features: int,
        output_features: int,
        bias: bool,
        pe_class: Type[PositionalEncoding],
        transform: Optional[Transform],
        pe_omega0: float,
        hidden_omega0: float,
        seed: Optional[int],
        last_linear: bool,
    ) -> None:

        super().__init__()

        self.transform = transform
        self.pe = pe_class(
            num_atoms=num_atoms,
            input_dim=input_dim,
            bias=bias,
            seed=seed,
            omega0=pe_omega0,
        )
        self.mlp = SineMLP(
            input_features=num_atoms,
            output_features=output_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            bias=bias,
            omega0=hidden_omega0,
            last_linear=last_linear,
        )
        self.inr = self.build_inr(self.pe, self.mlp, self.transform)

    def build_inr(
        self,
        pe: PositionalEncoding,
        mlp: SineMLP,
        transform: Optional[Transform] = None,
    ) -> nn.Sequential:

        if pe.num_atoms != mlp.input_features:
            raise ValueError(
                "Number of atoms in PE must match the input features of the MLP."
            )

        if transform is not None:
            if transform.input_dim != pe.input_dim:
                raise ValueError(
                    "Dimension of the transform must match the dimension of the PE."
                )
            return nn.Sequential(transform, pe, mlp)
        else:
            return nn.Sequential(pe, mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inr(x)


class HerglotzNet(BaseINR):

    def __init__(
        self,
        input_dim: int,
        num_atoms: int,
        hidden_layers: int,
        hidden_features: int,
        output_features: int,
        bias: bool = True,
        pe_omega0: float = 1.0,
        hidden_omega0: float = 1.0,
        seed: Optional[int] = None,
        last_linear: bool = True,
        unit_sphere: bool = True,
    ) -> None:

        transform = SphericalToCartesian(input_dim, unit=unit_sphere)
        super().__init__(
            input_dim=input_dim,
            num_atoms=num_atoms,
            hidden_layers=hidden_layers,
            hidden_features=hidden_features,
            output_features=output_features,
            bias=bias,
            pe_class=HerglotzPE,
            transform=transform,
            pe_omega0=pe_omega0,
            hidden_omega0=hidden_omega0,
            seed=seed,
            last_linear=last_linear,
        )


class SirenNet(BaseINR):

    def __init__(
        self,
        input_dim: int,
        num_atoms: int,
        hidden_layers: int,
        hidden_features: int,
        output_features: int,
        bias: bool = True,
        pe_omega0: float = 1.0,
        hidden_omega0: float = 1.0,
        seed: Optional[int] = None,
        last_linear: bool = True,
    ) -> None:

        super(SirenNet, self).__init__(
            input_dim=input_dim,
            num_atoms=num_atoms,
            hidden_layers=hidden_layers,
            hidden_features=hidden_features,
            output_features=output_features,
            bias=bias,
            pe_class=FourierPE,
            transform=None,
            pe_omega0=pe_omega0,
            hidden_omega0=hidden_omega0,
            seed=seed,
            last_linear=last_linear,
        )
