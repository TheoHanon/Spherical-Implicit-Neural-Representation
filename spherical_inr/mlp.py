import torch
import torch.nn as nn

from ._activation import get_activation

import math
from typing import List, Optional


class MLP(nn.Module):

    def __init__(
        self,
        input_features: int,
        output_features: int,
        hidden_sizes: List[int],
        bias: bool = True,
        activation: str = "relu",
        activation_kwargs: dict = {},
    ) -> None:

        super(MLP, self).__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.bias = bias

        self.hidden_layers = nn.ModuleList(
            nn.Linear(in_features, out_features, bias=bias)
            for in_features, out_features in zip(
                [input_features] + hidden_sizes[:-1],
                hidden_sizes[1:] + [output_features],
            )
        )
        self.activation = get_activation(activation, **activation_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer in self.hidden_layers[:-1]:
            x = self.activation(layer(x))

        return self.hidden_layers[-1](x)


class SineMLP(MLP):

    def __init__(
        self,
        input_features: int,
        output_features: int,
        hidden_sizes: List[int],
        bias: bool = True,
        omega0: float = 1.0,
    ) -> None:

        super(SineMLP, self).__init__(
            input_features,
            output_features,
            hidden_sizes,
            bias,
            activation="sin",
            activation_kwargs={"omega0": omega0},
        )
        self.omega0 = omega0
        self.init()

    def init(self) -> None:

        with torch.no_grad():

            for layer in self.hidden_layers:
                fan_in = layer.weight.size(1)
                bound = math.sqrt(6 / fan_in) / self.omega0
                layer.weight.uniform_(-bound, bound)

                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
