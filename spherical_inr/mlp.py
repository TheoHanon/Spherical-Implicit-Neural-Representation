import torch
import torch.nn as nn

from ._activation import get_activation

import math
from typing import List, Optional


class MLP(nn.Module):
    r"""Multi-Layer Perceptron (MLP).

    Defines a feedforward neural network that computes a mapping
    :math:`f: \mathbb{R}^{\text{input\_features}} \to \mathbb{R}^{\text{output\_features}}`
    via a series of fully connected layers interleaved with an activation function.
    If :math:`x` is the input, then the network computes

    .. math::
        f(x) = W_L\,\phi\Bigl(W_{L-1}\,\phi\bigl(\cdots\,\phi(W_1\,x+b_1)\bigr)+b_{L-1}\Bigr)+b_L,

    where :math:`\phi` denotes the activation function and :math:`W_i` and :math:`b_i` are the weight matrices
    and bias vectors of each layer, respectively.

    Parameters:
        input_features (int): Dimensionality of the input.
        output_features (int): Dimensionality of the output.
        hidden_sizes (List[int]): List of integers specifying the sizes of hidden layers.
        bias (bool, optional): If True, each linear layer includes a bias term (default: True).
        activation (str, optional): Identifier for the activation function to use (default: "relu").
        activation_kwargs (dict, optional): Additional keyword arguments for configuring the activation function.
    """

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
    r"""Sine-Activated Multi-Layer Perceptron (SineMLP).

    A variant of the MLP where the activation function is sine with a frequency scaling factor :math:`\omega_0`.
    For an input :math:`x`, the network computes

    .. math::
        f(x) = W_L\,\sin\Bigl(\omega_0\Bigl(W_{L-1}\,\sin\bigl(\omega_0(\cdots\,\sin(W_1\,x+b_1)\bigr)+b_{L-1}\Bigr)\Bigr)+b_L.

    In addition, the weights are initialized uniformly in the range

    .. math::
        \left[-\frac{\sqrt{6/n}}{\omega_0},\,\frac{\sqrt{6/n}}{\omega_0}\right],

    where :math:`n` is the number of input features to the corresponding layer.

    Parameters:
        input_features (int): Dimensionality of the input.
        output_features (int): Dimensionality of the output.
        hidden_sizes (List[int]): List of integers specifying the sizes of hidden layers.
        bias (bool, optional): If True, each linear layer includes a bias term (default: True).
        omega0 (float, optional): Frequency factor for the sine activation and weight initialization (default: 1.0).
    """

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
