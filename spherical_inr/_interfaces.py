from __future__ import annotations
from typing import Protocol, runtime_checkable
import torch
import torch.nn as nn


@runtime_checkable
class PositionalEncoding(Protocol):
    @property
    def out_dim(self) -> int: ...
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class MLP(Protocol):
    @property
    def in_dim(self) -> int: ...
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...
