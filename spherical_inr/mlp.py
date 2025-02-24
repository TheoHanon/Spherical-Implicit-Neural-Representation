import torch
import torch.nn as nn
import numpy as np

from abc import ABC, abstractmethod


class MLP(nn.Module, ABC):

    def __init__(
        self,
        input_features: int,
        output_features: int,
    ) -> None:

        super(MLP, self).__init__()

        self.input_features = input_features
        self.output_features = output_features

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class VanillaMLP(MLP):

    def __init__(
        self,
        input_features: int,
        output_features: int,
        hidden_features: int,
        hidden_layers: int,
        activation: callable = nn.ReLU(),
        bias: bool = True,
        last_linear: bool = True,
    ) -> None:

        super(VanillaMLP, self).__init__(input_features, output_features)

        self.input_features = input_features
        self.output_features = output_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.bias = bias
        self.activation = activation
        self.last_linear = last_linear
        self.hidden_layers = nn.ModuleList()

        self.hidden_layers.append(nn.Linear(input_features, hidden_features, bias))

        for _ in range(hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_features, hidden_features, bias))

        self.hidden_layers.append(nn.Linear(hidden_features, output_features, bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer in self.hidden_layers[:-1]:
            x = self.activation(layer(x))

        x = self.hidden_layers[-1](x)
        return x if self.last_linear else self.activation(x)


class SineMLP(VanillaMLP):

    def __init__(
        self,
        input_features: int,
        output_features: int,
        hidden_features: int,
        hidden_layers: int,
        bias: bool = True,
        omega0: float = 1.0,
        last_linear: bool = True,
    ) -> None:

        super(SineMLP, self).__init__(
            input_features,
            output_features,
            hidden_features,
            hidden_layers,
            activation=lambda x: omega0 * torch.sin(x),
            bias=bias,
            last_linear=last_linear,
        )
        self.register_buffer("omega0", torch.tensor(omega0, dtype=torch.float32))
        self.init()

    def init(self) -> None:

        with torch.no_grad():

            for layer in self.hidden_layers:

                fan_in = layer.weight.size(1)
                bound = np.sqrt(6 / fan_in) / self.omega0
                layer.weight.uniform_(-bound, bound)

                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
