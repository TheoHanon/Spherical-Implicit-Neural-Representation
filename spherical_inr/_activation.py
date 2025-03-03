import torch
import torch.nn as nn

from collections import OrderedDict

__all__ = ["get_activation"]


class Sin(nn.Module):

    def __init__(self, omega0: float = 1.0) -> None:
        super(Sin, self).__init__()
        self.register_buffer("omega0", torch.tensor(omega0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * x)


class Identity(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        if isinstance(content, tuple):
            cls, default_kwargs = content
        else:
            cls, default_kwargs = content, {}

        return lambda **kwargs: cls(**{**default_kwargs, **kwargs})


ACT2CLS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "sin": (Sin, {"omega0": 1.0}),
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "rrelu": nn.RReLU,
    "celu": nn.CELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "glu": nn.GLU,
    "hardshrink": nn.Hardshrink,
    "hardtanh": nn.Hardtanh,
    "hardsigmoid": nn.Hardsigmoid,
    "hardswish": nn.Hardswish,
    "prelu": nn.PReLU,
    "relu6": nn.ReLU6,
    "identity": Identity,
}


ACT2FN = ClassInstantier(ACT2CLS)


def get_activation(activation: str, **kwargs) -> nn.Module:
    if activation not in ACT2CLS:
        raise ValueError(
            f"Invalid activation: {activation}. Should be one of {list(ACT2CLS.keys())}."
        )

    return ACT2FN[activation](**kwargs)
