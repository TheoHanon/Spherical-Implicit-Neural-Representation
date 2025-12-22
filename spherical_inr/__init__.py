"""
spherical_inr package
=====================

This package provides modules for spherical implicit neural representations, including
activations, positional encoding, and transforms.

Modules:
    activations: Custom activation functions.
    inr: Implicit neural representations.
    positional_encoding: Herglotz and other positional encoding utilities.
    transforms: Coordinate transformation utilities.
"""

__version__ = "1.0.0"


from .inr import *
from .positional_encoding import *
from .mlp import *
from .diffops import *
from .coords import *
