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

### Instantiate a Network

Below is an example of how to instantiate and use the `HerglotzNet` module:

```python
import torch
import spherical_inr as sph 


# Parameters for HerglotzNet
input_dim = 2         # must be 1 or 2 for HerglotzNet
output_dim = 8
num_atoms = 16
mlp_sizes = 3*[32]  # hidden layer sizes
omega0 = 1.0
unit_sphere = True
seed = 42

# Instantiate the network
model = sph.HerglotzNet(
    input_dim=input_dim,
    output_dim=output_dim,
    num_atoms=num_atoms,
    mlp_sizes=mlp_sizes,
    bias=True,
    omega0=omega0,
    seed=seed
)

# Example input (for input_dim=2)
dummy_input = torch.randn(4, input_dim)
output = model(dummy_input)
print(output)
```

### Instantiate and Use a Positional Encoding

You can also directly instantiate a positional encoding and use it in your own torch model:

```python
import torch
import torch.nn as nn
import spherical_inr as sph 

# Instantiate Herglotz positional encoding (input_dim must be at least 2)
pe = sph.HerglotzPE(
    num_atoms=16,
    input_dim=3,
    bias=True,
    omega0=1.0,
    seed=42
)

# Example model using the positional encoding
class MyModel(nn.Module):
    def __init__(self, pe):
        super().__init__()
        self.pe = pe
        self.linear = nn.Linear(16, 8)
        
    def forward(self, x):
        x = self.pe(x)
        return self.linear(x)

model = MyModel(pe)
dummy_input = torch.randn(4, 3)
output = model(dummy_input)
print(output)
```

## 📚 References

1. Théo Hanon, Nicolas Mil-Homens Cavaco, John Kiely, Laurent Jacques,  
   *Herglotz-NET: Implicit Neural Representation of Spherical Data with Harmonic Positional Encoding*,  
   arXiv preprint, 2025.  
   [arXiv:2502.13777](https://arxiv.org/abs/2502.13777)

