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

__version__ = "0.3.4"


from .inr import *
from .positional_encoding import *
from .transforms import *
from .mlp import *
from .loss import *
from .differentiation import *
