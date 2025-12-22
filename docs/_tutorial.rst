Quick Example
=============

.. code-block:: python

    import torch
    from spherical_inr.inr import SphericalSirenNet
    
    # build a spherical SIREN: up to num_atoms harmonics, two hidden layers of size 64, output dim=1
    net = SphericalSirenNet(num_atoms=25, mlp_sizes=[64,64], output_dim=1)

    # sample some θ,ϕ in radians
    coords = torch.rand(8,2) * torch.tensor([3.1416, 6.2832])
    y = net(coords)  # forward on sphere

