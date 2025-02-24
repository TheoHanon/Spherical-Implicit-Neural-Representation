# Spherical-Implicit-Neural-Representation

A package for spherical implicit neural representations using Herglotz-based positional encoding.

## Installation

You can install the package from PyPI:

```bash
pip install spherical-inr
```

Or install the development version locally:

```bash
git clone https://github.com/yourusername/spherical_inr.git
cd spherical_inr
pip install -e .
```

## Getting Started

Below is an example of how to instantiate and use the `HerglotzNet` module:

```python
import torch
import spherical_inr as sph 

# Parameters for the HerglotzNet
input_dim = 3
num_atoms = 16
hidden_layers = 2
hidden_features = 32
output_features = 8
omega0 = 1.0
unit_sphere = True
seed = 42

# Instantiate the network
model = sph.HerglotzNet(
    input_dim = input_dim
    num_atoms=num_atoms,
    hidden_layers=hidden_layers,
    hidden_features=hidden_features,
    out_features=out_features,
    omega0=omega0,
    seed=seed,
    unit_sphere = unit_sphere
)

# Example input 
dummy_input = torch.randn(4, 3)  
output = model(dummy_input)
print(output)
```

## 📚 References

1. Théo Hanon, Nicolas Mil-Homens Cavaco, John Kiely, Laurent Jacques,  
   *Herglotz-NET: Implicit Neural Representation of Spherical Data with Harmonic Positional Encoding*,  
   arXiv preprint, 2025.  
   [arXiv:2502.13777](https://arxiv.org/abs/2502.13777)

