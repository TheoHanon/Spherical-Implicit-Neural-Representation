import torch
import torch.nn as nn
from typing import Optional
from .positional_encoding import HerglotzPE
from .activations import Sin
import math


class HerglotzNet(nn.Module):
    """
    HerglotzNet Neural Network Module

    This network consists of a Herglotz-based positional encoding layer followed by a sequence of
    linear layers interleaved with sine activations. The final layer can be configured to either
    output a linear transformation or a sine-activated transformation.

    Parameters:
        num_atoms (int): Number of atoms used in the Herglotz positional encoding.
        hidden_layers (int): Number of hidden layers to use after the initial linear layer.
        hidden_features (int): Number of features (neurons) in each hidden layer.
        out_features (int): Number of output features.
        omega0 (float, optional): Scaling factor for the positional encoding. Default is 1.0.
        seed (Optional[int], optional): Seed for random number generation to ensure reproducibility.
        input_domain (str, optional): Domain of the input. Accepts "s2", "s1", "r3", or "r2". Default is "s2".
        outermost_linear (bool, optional): If True, the final layer is a linear transformation only.
                                           If False, a sine activation is applied after the final linear transformation.
                                           Default is False.

    Attributes:
        pe (HerglotzPE): The Herglotz-based positional encoding layer.
        hidden_layers (nn.ModuleList): A list of hidden linear layers.
        last_layer (nn.Module): The final layer, which is either a linear layer or a sequential module
                                applying a linear transformation followed by a sine activation.
    """

    def __init__(
        self,
        num_atoms: int,
        hidden_layers: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
        omega0: float = 1.0,
        seed: Optional[int] = None,
        input_domain: str = "s2",
        outermost_linear: bool = True,
    ) -> None:

        super(HerglotzNet, self).__init__()

        self.pe = HerglotzPE(num_atoms, omega0, seed, input_domain)
        self.hidden_layers = nn.ModuleList()

        self.hidden_layers.append(nn.Linear(num_atoms, hidden_features, bias))

        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_features, hidden_features, bias))

        if outermost_linear:
            self.last_layer = nn.Linear(hidden_features, out_features, bias)
        else:
            self.last_layer = nn.Sequential(
                nn.Linear(hidden_features, out_features, bias), Sin()
            )
        self.init_weights()

    def init_weights(self) -> None:
        """
        Initialize the weights of the network as SIREN."""

        last_layer = (
            self.last_layer
            if isinstance(self.last_layer, nn.Linear)
            else self.last_layer[0]
        )

        for layer in [*self.hidden_layers, last_layer]:
            fan_in = layer.weight.size(1)
            bound = math.sqrt(6 / fan_in)
            nn.init.uniform_(layer.weight, -bound, bound)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the HerglotzNet.

        The input x is first encoded using the Herglotz positional encoding.
        It is then passed through a sequence of hidden layers, where each hidden layer applies a linear
        transformation followed by a sine activation. Finally, the result is processed by the last layer,
        which may or may not include an additional sine activation based on the configuration.

        Parameters:
            x (torch.Tensor): Input tensor. Its shape should be compatible with the positional encoding layer.

        Returns:
            torch.Tensor: The output of the network.
        """

        x = self.pe(x)

        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.sin(x)

        x = self.last_layer(x)

        return x
