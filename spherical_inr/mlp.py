import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import List

__all__ = ["ReLUMLP", "SineMLP"]


class ReLUMLP(nn.Module):
    r"""
    ReLU-activated multi-layer perceptron.

    Hidden layers apply:

    .. math::
        h_k = \mathrm{ReLU}(W_k h_{k-1} + b_k), \quad k=1,\dots,L-1,

    and the output layer is linear:

    .. math::
        f_\theta(x) = W_L h_{L-1} + b_L.

    Parameters
    ----------
    input_features:
        Input dimension.
    output_features:
        Output dimension.
    hidden_sizes:
        List of hidden layer widths.
    bias:
        Whether to include biases in each linear layer.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        hidden_sizes: List[int],
        bias: bool = True,
    ):
        super().__init__()
        self.input_features = int(input_features)
        self.output_features = int(output_features)

        sizes = [self.input_features] + list(hidden_sizes) + [self.output_features]
        self.layers = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1], bias=bias) for i in range(len(sizes) - 1)
        )

    @property
    def in_dim(self) -> int:
        return self.input_features

    @property
    def out_dim(self) -> int:
        return self.output_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class SineMLP(nn.Module):
    r"""
    Sine-activated multi-layer perceptron.

    This module is identical to :class:`ReluMLP` except the hidden activation is
    a sine nonlinearity with frequency factor :math:`\omega_0`.
    Given an input :math:`x`, the hidden activations are

    .. math::
        h_0 = x, \qquad
        h_k = \sin\!\bigl(\omega_0 (W_k h_{k-1} + b_k)\bigr),
        \quad k=1,\dots,L-1,

    and the output layer is linear:

    .. math::
        f_\theta(x) = W_L h_{L-1} + b_L.

    The weights are initialized uniformly (per layer) as

    .. math::
        W_k \sim \mathcal{U}\!\left[-\frac{\sqrt{6/n_k}}{\omega_0},
        \frac{\sqrt{6/n_k}}{\omega_0}\right],

    where :math:`n_k` is the fan-in (number of input features) of layer :math:`k`.
    Biases are initialized to zero when present.

    Parameters
    ----------
    input_features:
        Input dimension.
    output_features:
        Output dimension.
    hidden_sizes:
        List of hidden layer widths.
    bias:
        Whether to include biases in each linear layer.
    omega0:
        Frequency factor :math:`\omega_0` used in the sine activation and in the
        weight initialization bound.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        hidden_sizes: List[int],
        bias: bool = True,
        omega0: float = 1.0,
    ) -> None:

        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.bias = bias

        sizes = [input_features] + hidden_sizes + [output_features]
        self.hidden_layers = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1], bias=bias) for i in range(len(sizes) - 1)
        )
        self.omega0 = omega0
        self.reset_parameters()

    @property
    def in_dim(self) -> int:
        return self.input_features

    @property
    def out_dim(self) -> int:
        return self.output_features

    def reset_parameters(
        self,
    ) -> None:

        with torch.no_grad():
            for layer in self.hidden_layers:
                fan_in = layer.weight.size(1)
                bound = math.sqrt(6 / fan_in) / self.omega0
                layer.weight.uniform_(-bound, bound)

                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers[:-1]:
            x = torch.sin(self.omega0 * layer(x))
        return self.hidden_layers[-1](x)
