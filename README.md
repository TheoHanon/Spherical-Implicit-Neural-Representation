# Spherical-Implicit-Neural-Representation
![Project Overview](images/atoms.png)


[![Documentation Status](https://readthedocs.org/projects/spherical-implicit-neural-representation/badge/?version=latest)](https://spherical-implicit-neural-representation.readthedocs.io/en/latest/)


*Spherical-Implicit-Neural-Representation* unifies Fourier features (SIREN) [1], pure Spherical Harmonics (SphericalSirenNet) [3], and our learnable Herglotz‐map encodings [2] into a single PyTorch toolbox. Build implicit neural representations on:

- **$\mathbb{S}^2$** (HerglotzNet, SphericalSirenNet)
- **Volumetric data** in $\mathbb{R}^3$ with radial basis (solid harmonics)
- **Generic ℝᵈ inputs** via FourierPE, HerglotzPE

> **Coordinate conventions**:
>
> - **Angles**: $\theta \in [0,\pi], \varphi ∈[0,2\pi)$ in radians.
> - **Full spherical**: $(r, \theta, \varphi)$.

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

## 🚀 Quickstart

```python
import torch
from spherical_inr import HerglotzNet

# Create a HerglotzNet: harmonic order L → num_atoms=(L+1)**2
model = HerglotzNet(
    num_atoms = 50,                # spherical-harmonic degree
    mlp_sizes=[64,64],  # two hidden layers of width 64
    output_dim=1,       # scalar output per direction
)
# Random spherical angles (θ,φ)
x = torch.rand(16,2) * torch.tensor([torch.pi, 2*torch.pi])
y = model(x)
print(y.shape)  # → (16,1)
```

---

## 📚 References

1. **Sitzmann, M., Martel, J., Berg, R., Lindell, D. B., & Wetzstein, G.** (2021). *Implicit Neural Representations with Periodic Activation Functions (SIREN)*. Advances in Neural Information Processing Systems (NeurIPS).  [https://arxiv.org/abs/2006.09661](https://arxiv.org/abs/2006.09661)
2. **Hanon, T., et al.** (2025). *Herglotz-NET: Implicit Neural Representation of Spherical Data with Harmonic Positional Encoding*. arXiv preprint arXiv:2502.13777. [https://arxiv.org/abs/2502.13777](https://arxiv.org/abs/2502.13777)  
3. **Rußwurm, M., Klemmer, K., Rolf, E., Zbinden, R., & Tuia, D.** (2024). *Geographic Location Encoding with Spherical Harmonics and Sinusoidal Representation Networks*. arXiv preprint arXiv:2310.06743. [https://arxiv.org/abs/2310.06743](https://arxiv.org/abs/2310.06743)  


