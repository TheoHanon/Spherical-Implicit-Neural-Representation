Welcome to Spherical INRs
=========================

.. grid:: 1 1 3 3

  .. grid-item::
    :columns: 12 12 8 8

    ``spherical-inr`` is a lightweight PyTorch toolbox for building implicit neural
    representations (INRs) of functions defined on the sphere :math:`\mathbb{S}^2`.

    The library is designed around a clean separation of concerns:

    - **Positional encodings** define the geometric and spectral structure of the representation.
    - **MLP backbones** map encoded coordinates to scalar or vector-valued fields.
    - **Model wrappers** combine encodings and networks into ready-to-use spherical INRs.
    - **Differential operators** enable gradients, divergence, and Laplacians through automatic differentiation.

    All components are composable and can be used independently, allowing users to
    move easily between simple baselines and more structured models.  
        
  .. grid-item-card:: Contents
    :class-title: sd-fs-5
    :class-body: sd-pl-4

    .. toctree::
      :maxdepth: 1

      Installing <_install.rst>
      API <_api.rst>
      Tutorials <_tutorial.rst>
      References <_reference.rst>
