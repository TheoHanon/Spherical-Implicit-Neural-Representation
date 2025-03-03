# Spherical-Implicit-Neural-Representation
[![Documentation Status](https://readthedocs.org/projects/spherical-implicit-neural-representation/badge/?version=latest)](https://spherical-implicit-neural-representation.readthedocs.io/en/latest/)



*Spherical-Implicit-Neural-Representation* is a Python package for constructing spherical implicit neural representations using Herglotz-based positional encoding. It provides flexible modules for processing spherical data along with customizable positional encoding layers and regularization tools.

## Installation

Install the package from PyPI:

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

### Instantiate HerglotzNet

The `HerglotzNet` module is designed for (θ, φ) coordinate data. Here’s an example of how to instantiate and use it:

```python
import torch
import spherical_inr as sph

# Parameters for HerglotzNet
output_dim = 1
inr_sizes = [16] + 3 * [32]  # [PE size] + (hidden layers * hidden features)
omega0 = 1.0
seed = 42


# Instantiate the network NOTE : # HNET is defined for (θ, φ) coordinates only
model = sph.HerglotzNet(
    output_dim=output_dim,
    inr_sizes=inr_sizes,
    bias=True,
    pe_omega0=omega0,
    seed=seed
)

# Example
dummy_input = torch.randn(4, 2)
output = model(dummy_input)
print(output)
```

### Generic Cartesian INR

You can also create a customized Cartesian implicit neural representation (INR). For example:

```python
import torch
import spherical_inr as sph

# INR parameters
input_dim = 3
output_dim = 1
inr_sizes = [100] + 3 * [100]
pe = "fourier"
activation = "sin"
bias = False

# Instantiate a generic Cartesian INR
inr = sph.INR(
    input_dim=input_dim,
    output_dim=output_dim,
    inr_sizes=inr_sizes,
    pe=pe,
    activation=activation,
    bias=bias
)
```

To incorporate Laplacian regularization into your loss function:

```python
import torch
from spherical_inr.loss import CartesianLaplacianLoss

laplacian_loss = CartesianLaplacianLoss()
mse_loss = torch.nn.MSELoss()

def loss_fn(target, y_pred, y_reg, x_reg):
    reg = laplacian_loss(y_reg, x_reg)
    mse = mse_loss(target, y_pred)
    return reg + mse
```

Then train your model as usual.

### Instantiate and Use a Positional Encoding

You can also directly instantiate a positional encoding module and integrate it into your own PyTorch model. For example:

```python
import torch
import torch.nn as nn
import spherical_inr as sph

# Example model using a positional encoding
class MyModel(nn.Module):
    def __init__(self, num_atoms, input_dim, output_dim, bias, omega0, seed):
        super().__init__()
        self.pe = sph.RegularHerglotzPE(
            num_atoms=num_atoms,
            input_dim=input_dim,
            bias=bias,
            omega0=omega0,
            seed=seed
        )
        self.linear = nn.Linear(num_atoms, output_dim)
        
    def forward(self, x):
        x = self.pe(x)
        return self.linear(x)

# Instantiate the model
model = MyModel(
    num_atoms=50,
    input_dim=10,
    output_dim=5,
    bias=True,
    omega0=1.0,
    seed=42,
)

dummy_input = torch.randn(4, 10)
output = model(dummy_input)
print(output)
```




## 📚 References

1. Théo Hanon, Nicolas Mil-Homens Cavaco, John Kiely, Laurent Jacques,  
   *Herglotz-NET: Implicit Neural Representation of Spherical Data with Harmonic Positional Encoding*,  
   arXiv preprint, 2025.  
   [arXiv:2502.13777](https://arxiv.org/abs/2502.13777)

   
