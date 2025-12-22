Code Documentation
==================

Welcome! This page provides a structured overview of the ``spherical-inr`` toolbox.
Modules are grouped by purpose.

Conventions
-----------

Spherical coordinates are represented as ``(theta, phi)`` in **radians**, with tensors
of shape ``(..., 2)``:

- ``theta ∈ [0, π]`` (polar angle)
- ``phi   ∈ [-π, π]`` (azimuth)

Cartesian coordinates are represented as tensors of shape ``(..., 3)`` in :math:`\mathbb{R}^3`.

Core modules
------------

`Models`_
^^^^^^^^^

Ready-to-use networks combining an encoding and a backbone:

- :class:`~spherical_inr.inr.INR` — generic composition wrapper
- :class:`~spherical_inr.inr.SirenNet` — Fourier features on angles + sine MLP
- :class:`~spherical_inr.inr.HerglotzNet` — angle input with internal spherical→Cartesian conversion
- :class:`~spherical_inr.inr.SphericalSirenNet` — spherical harmonics on angles + sine MLP

`Positional encodings`_
^^^^^^^^^^^^^^^^^^^^^^^

Coordinate-to-feature mappings used by models or custom compositions:

- :class:`~spherical_inr.positional_encoding.FourierPE`
- :class:`~spherical_inr.positional_encoding.SphericalHarmonicsPE`
- :class:`~spherical_inr.positional_encoding.HerglotzPE`

`MLP backbones`_
^^^^^^^^^^^^^^^^

Pointwise networks applied to encoded coordinates:

- :class:`~spherical_inr.mlp.ReLUMLP`
- :class:`~spherical_inr.mlp.SineMLP`

`Coordinate transforms`_
^^^^^^^^^^^^^^^^^^^^^^^^

Utilities for converting between spherical and Cartesian representations:

- :func:`~spherical_inr.coords.tp_to_r3`
- :func:`~spherical_inr.coords.r3_to_tp`
- :func:`~spherical_inr.coords.rtp_to_r3`
- :func:`~spherical_inr.coords.r3_to_rtp`

`Differential operators`_
^^^^^^^^^^^^^^^^^^^^^^^^^

Autograd-based operators for implicit functions (Cartesian, spherical, and :math:`\mathbb{S}^2` variants):

- :func:`~spherical_inr.diffops.cartesian_gradient`
- :func:`~spherical_inr.diffops.cartesian_divergence`
- :func:`~spherical_inr.diffops.cartesian_laplacian`

- :func:`~spherical_inr.diffops.spherical_gradient`
- :func:`~spherical_inr.diffops.spherical_divergence`
- :func:`~spherical_inr.diffops.spherical_laplacian`

- :func:`~spherical_inr.diffops.s2_gradient`
- :func:`~spherical_inr.diffops.s2_divergence`
- :func:`~spherical_inr.diffops.s2_laplacian`

.. toctree::
   :hidden:
   :caption: Core Modules

   modules/positional_encoding
   modules/mlp
   modules/inr
   modules/coords
   modules/diffops


.. _Positional encodings: modules/positional_encoding.html
.. _MLP backbones: modules/mlp.html
.. _Models: modules/inr.html
.. _Coordinate transforms: modules/coords.html
.. _Differential operators: modules/diffops.html