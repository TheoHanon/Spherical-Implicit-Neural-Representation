import torch
import torch.nn as nn


class Sin(nn.Module):
    """A simple module to apply the sine function elementwise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)
