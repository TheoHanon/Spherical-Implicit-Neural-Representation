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
num_atoms = 16
hidden_layers = 2
hidden_features = 32
out_features = 8
omega0 = 1.0
seed = 42
input_domain = "s2"  # Options: "s2", "s1", "r3", "r2"
outermost_linear = True  # If False, a sine activation is applied after the last linear layer

# Instantiate the network
model = sph.HerglotzNet(
    num_atoms=num_atoms,
    hidden_layers=hidden_layers,
    hidden_features=hidden_features,
    out_features=out_features,
    omega0=omega0,
    seed=seed,
    input_domain=input_domain,
    outermost_linear=outermost_linear,
)

# Example input 
dummy_input = torch.randn(4, 3)  
output = model(dummy_input)
print(output)
```