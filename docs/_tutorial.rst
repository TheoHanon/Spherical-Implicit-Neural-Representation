Quick Example
~~~~~~~~~~~~~

.. code-block:: python

    import torch
    from spherical_inr import SphericalSirenNet, tp_to_r3

    # build a spherical SIREN: L=3 harmonics, two hidden layers of size 64, output dim=1
    net = SphericalSirenNet(L=3, mlp_sizes=[64,64], output_dim=1, seed=0)

    # sample some θ,ϕ in radians
    coords = torch.rand(8,2) * torch.tensor([3.1416, 6.2832])
    y = net(coords)  # forward on sphere

