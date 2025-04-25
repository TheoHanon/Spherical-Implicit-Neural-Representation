# Spherical-Implicit-Neural-Representation
![Project Overview](images/atoms.png)


[![Documentation Status](https://readthedocs.org/projects/spherical-implicit-neural-representation/badge/?version=latest)](https://spherical-implicit-neural-representation.readthedocs.io/en/latest/)


*Spherical-Implicit-Neural-Representation* unifies Fourier features (SIREN) [1], pure Spherical Harmonics (SphericalSirenNet) [3], and our learnable Herglotz‐map encodings [2] into a single PyTorch toolbox. Build implicit neural representations on:

- **S²** (HerglotzNet, SphericalSirenNet)
- **Volumetric data** in ℝ³ with radial basis (solid harmonics)
- **Generic ℝᵈ inputs** via FourierPE, HerglotzPE

> **Coordinate conventions**:
>
> - **Angles**: θ∈[0,π], φ∈[0,2π) in radians.
> - **Full spherical**: (r≥0, θ, φ).
> - **2D polar**: (r,θ) or angle-only (θ).



## 📦 Installation

```bash
pip install spherical-inr
```

*OR for development:*

```bash
git clone https://github.com/yourusername/spherical_inr.git
cd spherical_inr
pip install -e .
```


## 📦 Features

- **General INR** (`INR`): Plug-and-play Cartesian implicit networks with your choice of  
  Fourier / Herglotz / Spherical-Harmonic features + flexible MLP backbones.  
- **Sphere-only nets** (`HerglotzNet`, `SphericalSirenNet`): For data on \(S^2\), encode \((\theta,\phi)\) directly.  
- **Solid-harmonic nets** (`RegularSolid*` / `IrregularSolid*`): Capture radial & angular variation in \(\mathbb R^3\).  
- **SIREN**-style variants (`SirenNet`, `HerglotzSirenNet`): Learnable Fourier / Herglotz features + sine-activated MLPs.  
- **Standalone PEs**: `FourierPE`, `HerglotzPE`, `SphericalHarmonicsPE`, `RegularSolidHarmonicsPE`, `IrregularSolidHarmonicsPE`, …  
- **Transforms**: Handy spherical↔cartesian converters (`tp_to_r3`, `rtp_to_r3`, `rt_to_r2`, `t_to_r2`, …).  
- **Regularization**: Laplacian losses for smoothness (`CartesianLaplacianLoss`, `SphericalLaplacianLoss`, …).  
- **Differentation** : Cartesian & Spherical Differential Operators (`spherical_gradient`, `spherical_laplacian`, `spherical_divergence`, `cartesian_gradient`, ...).

## 🚀 Quickstart

### 1️⃣ HerglotzNet on S²

```python
import torch
from spherical_inr import HerglotzNet

# Create a HerglotzNet: harmonic order L → num_atoms=(L+1)**2
model = HerglotzNet(
    L=3,                # spherical-harmonic degree
    mlp_sizes=[64,64],  # two hidden layers of width 64
    output_dim=1,       # scalar output per direction
    seed=0,
)
# Random spherical angles (θ,φ)
x = torch.rand(16,2) * torch.tensor([torch.pi, 2*torch.pi])
y = model(x)
print(y.shape)  # → (16,1)
```

### 2️⃣ Generic Cartesian INR

```python
from spherical_inr import INR

# Fourier‐feature INR with sin‐activation (SIREN style)
inr = INR(
    num_atoms=128,          # Fourier channels
    mlp_sizes=[256,256],    # two hidden layers
    output_dim=3,           # 3D output
    input_dim=3,            # 3D Cartesian inputs
    pe="fourier",         # FourierPE
    activation="sin",      # sine‐activated MLP
    pe_kwargs={"omega0":10},
)
x = torch.randn(10,3)
y = inr(x)
```

### 3️⃣ Solid‐harmonic INR in ℝ³

```python
from spherical_inr import RegularSolidHarmonicsPE, SineMLP
import torch

# Regular solid harmonics encode radial+angular growth r^ℓ
pe = RegularSolidHarmonicsPE(L=2, seed=1)
mlp = SineMLP(input_features=pe.num_atoms, output_features=1, hidden_sizes=[64])
# Example input: (r,θ,φ)
x_sph = torch.stack([torch.rand(5), torch.rand(5)*torch.pi, torch.rand(5)*2*torch.pi], dim=-1)
x_cart = rtp_to_r3(x_sph)
x = pe(x_cart)
y = mlp(x)
```

---

## 📚 References

1. **Sitzmann, M., Martel, J., Berg, R., Lindell, D. B., & Wetzstein, G.** (2021). *Implicit Neural Representations with Periodic Activation Functions (SIREN)*. Advances in Neural Information Processing Systems (NeurIPS).  [https://arxiv.org/abs/2006.09661](https://arxiv.org/abs/2006.09661)
2. **Hanon, T., et al.** (2025). *Herglotz-NET: Implicit Neural Representation of Spherical Data with Harmonic Positional Encoding*. arXiv preprint arXiv:2502.13777. [https://arxiv.org/abs/2502.13777](https://arxiv.org/abs/2502.13777)  
3. **Rußwurm, M., Klemmer, K., Rolf, E., Zbinden, R., & Tuia, D.** (2024). *Geographic Location Encoding with Spherical Harmonics and Sinusoidal Representation Networks*. arXiv preprint arXiv:2310.06743. [https://arxiv.org/abs/2310.06743](https://arxiv.org/abs/2310.06743)  


