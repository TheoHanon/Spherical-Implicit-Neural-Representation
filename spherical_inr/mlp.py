import torch
import torch.nn as nn
import numpy as np

from abc import ABC, abstractmethod
from typing import List


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


class SineMLP(MLP):

    def __init__(
        self,
        input_features: int,
        output_features: int,
        hidden_sizes: List[int],
        bias: bool = True,
        omega0: float = 1.0,
    ) -> None:

        super(SineMLP, self).__init__(input_features, output_features)

        self.input_features = input_features
        self.output_features = output_features
        self.hidden_sizes = hidden_sizes
        self.bias = bias
        self.register_buffer("omega0", torch.tensor(omega0, dtype=torch.float32))

        self.hidden_layers = nn.ModuleList(
            nn.Linear(in_features, out_features, bias=bias)
            for in_features, out_features in zip(
                [input_features] + hidden_sizes[:-1],
                hidden_sizes[1:] + [output_features],
            )
        )
        self.init()

    def init(self) -> None:

        with torch.no_grad():

            for layer in self.hidden_layers:
                fan_in = layer.weight.size(1)
                bound = np.sqrt(6 / fan_in) / self.omega0
                layer.weight.uniform_(-bound, bound)

                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):

        for layer in self.hidden_layers[:-1]:

            x = torch.sin(self.omega0 * layer(x))

        return self.hidden_layers[-1](x)
