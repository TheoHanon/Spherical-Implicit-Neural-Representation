import torch
import torch.nn as nn

from .transforms import (
    tp_to_r3,
    rtp_to_r3,
)
from .positional_encoding import (
    get_positional_encoding,
    HerglotzPE,
    FourierPE,
    SphericalHarmonicsPE,
    RegularSolidHarmonicsPE,
    IrregularSolidHarmonicsPE,
    RegularHerglotzPE,
    IrregularHerglotzPE,
    )

from .mlp import (
    MLP,
    SineMLP,
)

from typing import Optional, List


class INR(nn.Module):
    r"""Implicit Neural Representation (INR).

    Maps inputs in ℝᵈ through a positional encoding ψ onto a multilayer perceptron.

    For each input **x** of shape `(..., input_dim)`, you get an output of shape `(..., output_dim)` by
    1. Computing **ψ(x)** via a chosen PE: shape `(..., num_atoms)`.
    2. Passing that through an MLP with hidden sizes `mlp_sizes`.

    Parameters:
        num_atoms (int):
            Number of channels (atoms) output by the positional encoding ψ.
        mlp_sizes (List[int]):
            Hidden‐layer sizes for the MLP. E.g. `[64, 64]` for two hidden layers of width 64.
        output_dim (int):
            Number of output features per input point.
        input_dim (int, keyword-only):
            Dimensionality of each input x. Must match the PE’s requirement:
            - For `"herglotz"` or `"fourier"`: any positive int (commonly 2 or 3).
            - For `"spherical_harmonics"`: **must** be 2 (θ,φ).
        pe (str, optional):
            Which PE to use. One of:
            - `"herglotz"`: Herglotz map in ℝᵈ.  
            - `"spherical_harmonics"`: real SH on S² (needs `input_dim=2`).  
            - `"fourier"`: Fourier‐feature map.
        activation (str, optional):
            Activation for MLP layers, e.g. `"relu"`, `"gelu"`, etc.
        pe_kwargs (dict, optional):
            Passed directly into the chosen PE’s constructor—see that class’s docstring.
        mlp_kwargs (dict, optional):
            Extra `MLP(…)` keyword args (e.g. `bias=True`).
        activation_kwargs (dict, optional):
            Extra kwargs for the activation function (e.g. `{"inplace":True}`).

    Input:
        - **x**: Tensor of shape `(..., input_dim)`.
          * For Fourier or Herglotz: any real d-vector.
          * For SH: last two components are (θ,φ) in radians.
    Output:
        - Tensor of shape `(..., output_dim)`.

    """

    def __init__(
        self,
        num_atoms: int,
        mlp_sizes: List[int],
        output_dim: int,
        *,
        input_dim: int, 
        pe: str = "herglotz",
        activation: str = "relu",
        pe_kwargs: Optional[dict] = None,
        mlp_kwargs: Optional[dict] = None,
        activation_kwargs : Optional[dict] = None,
    ) -> None:

        super(INR, self).__init__()

        if pe not in ["herglotz", "spherical_harmonics", "fourier"]:
            raise ValueError(
                "Invalid positional encoding type. Choose from 'herglotz', 'spherical_harmonics', or 'fourier'."
            )
        
        if pe == "spherical_harmonics" and input_dim != 2:
            raise ValueError(
                "Spherical harmonics positional encoding requires input_dim to be 2."
            )
            

        self.pe = get_positional_encoding(
            pe,
            **{
                "num_atoms": num_atoms,
                "input_dim": input_dim,
                **(pe_kwargs or {}),
            },
        )

        if activation == "sin":

            self.mlp = SineMLP(
                input_features=self.pe.num_atoms,
                output_features=output_dim,
                hidden_sizes=mlp_sizes,
                **(mlp_kwargs or {}),
            )
        else:
            self.mlp = MLP(
                input_features=self.pe.num_atoms,
                output_features=output_dim,
                hidden_sizes=mlp_sizes,
                activation=activation,
                activation_kwargs= activation_kwargs or {},
                **(mlp_kwargs or {}),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pe(x)
        x = self.mlp(x)

        return x


class HerglotzNet(nn.Module):
    r"""HerglotzNet on the 2-sphere.

    Expects inputs in spherical coords (θ,φ), converts them to Cartesian (x,y,z),
    then applies a 3-D Herglotz PE and a sine-activated MLP (SineMLP).

    Workflow:
        x_sph ∈ S² ──tp_to_r3──▶ x_cart ∈ ℝ³  
                          └─HerglotzPE─▶ ψ(x) ∈ ℝⁿ  
                                       └─SineMLP──▶ output ∈ ℝᵒ

    Parameters:
        L (int):
            Harmonic order. The PE creates `num_atoms = (L+1)**2` channels.
        mlp_sizes (List[int]):
            Hidden layer sizes for the SineMLP.
        output_dim (int):
            Number of output features.
        seed (int, optional):
            RNG seed for reproducible atom initialization in HerglotzPE.
        pe_kwargs (dict, optional):
            Extra args for `HerglotzPE(…)`—see its docstring.
        mlp_kwargs (dict, optional):
            Extra args for `SineMLP(…)` (e.g. `omega0`).

    Input:
        - **x**: Tensor `(..., 2)` of spherical coords (θ ∈ [0,π], φ ∈ [0,2π)).
    Output:
        - Tensor `(..., output_dim)`.

    """

    def __init__(
        self,
        L : int, 
        mlp_sizes: List[int],
        output_dim: int,
        *,
        seed: Optional[int] = None,
        pe_kwargs: Optional[dict] = None,
        mlp_kwargs: Optional[dict] = None,
    ) -> None:

        super(HerglotzNet, self).__init__()

        self.pe = HerglotzPE(
            L=L,
            input_dim=3,
            seed=seed,
            **(pe_kwargs or {}),
        )

        self.mlp = SineMLP(
            input_features=self.pe.num_atoms,
            output_features=output_dim,
            hidden_sizes=mlp_sizes,
            **(mlp_kwargs or {}),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = tp_to_r3(x)
        x = self.pe(x)
        x = self.mlp(x)

        return x


class RegularHerglotzNet(nn.Module):
    r"""Regular Solid HerglotzNet.

    Like HerglotzNet but uses **regular** solid harmonics → features grow like rˡ.
    Inputs are full spherical coords (r,θ,φ) so that features encode radial and angular info.

    Workflow:
        x_sph ∈ ℝ³ ──rtp_to_r3──▶ x_cart ∈ ℝ³  
                            └─RegularSolidHerglotzPE─▶ ψ(x)  
                                                     └─SineMLP──▶ output

    Parameters:
        L (int):
            Harmonic order. `num_atoms=(L+1)**2`.
        mlp_sizes (List[int]):
            Hidden layer widths for the SineMLP.
        output_dim (int):
            Dimensionality of the network’s final output.
        seed (int, optional):
            RNG seed for PE.
        pe_kwargs (dict, optional):
            Extra args for `RegularHerglotzPE(…)`.
        mlp_kwargs (dict, optional):
            Extra args for `SineMLP(…)`.

    Input:
        - **x**: Tensor `(..., 3)` as (r,θ,φ).
    Output:
        - Tensor `(..., output_dim)`.

    """

    def __init__(
        self,
        L :int,
        mlp_sizes: List[int],
        output_dim: int,
        *,
        seed: Optional[int] = None,
        pe_kwargs: Optional[dict] = None,
        mlp_kwargs: Optional[dict] = None,
    ) -> None:

        super(RegularHerglotzNet, self).__init__()

    
        self.pe = RegularHerglotzPE(
            L=L,
            seed=seed,
            ** (pe_kwargs or {}),
        )

        self.mlp = SineMLP(
            input_features=self.pe.num_atoms,
            output_features=output_dim,
            hidden_sizes=mlp_sizes,
            **(mlp_kwargs or {}),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = rtp_to_r3(x)
        x = self.pe(x)
        x = self.mlp(x)

        return x


class IrregularHerglotzNet(nn.Module):
    r"""Irregular Solid HerglotzNet.

    Identical to RegularHerglotzNet but uses **irregular** solid harmonics → features decay like 1/rˡ⁺¹.

    Use this when you want the encoding to vanish at infinity.

    Parameters:
        L (int): Harmonic order → `num_atoms=(L+1)**2`.
        mlp_sizes (List[int]): Hidden widths for SineMLP.
        output_dim (int): Output feature count.
        seed (int, optional): RNG seed.
        pe_kwargs (dict, optional): Extra for `IrregularHerglotzPE`.
        mlp_kwargs (dict, optional): Extra for `SineMLP`.

    Input / Output: same shapes as RegularHerglotzNet.

    """

    def __init__(
        self,
        L :int,
        mlp_sizes: List[int],
        output_dim: int,
        *,
        seed: Optional[int] = None,
        pe_kwargs: Optional[dict] = None,
        mlp_kwargs: Optional[dict] = None,
    ) -> None:

        super(IrregularHerglotzNet, self).__init__()

    
        self.pe = IrregularHerglotzPE(
            L=L,
            seed=seed,
            ** (pe_kwargs or {}),
        )

        self.mlp = SineMLP(
            input_features=self.pe.num_atoms,
            output_features=output_dim,
            hidden_sizes=mlp_sizes,
            **(mlp_kwargs or {}),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = rtp_to_r3(x)
        x = self.pe(x)
        x = self.mlp(x)

        return x

class SirenNet(nn.Module):
    r"""Standard SIREN network with learnable Fourier PE.

    Applies a FourierPE followed by a sine-activated MLP (SineMLP).

    Workflow:
        x ∈ ℝᵈ ──FourierPE(num_atoms, ω₀)─▶ ψ(x) ∈ ℝⁿ  
                          └─SineMLP(ω₀)──▶ output ∈ ℝᵒ

    Parameters:
        num_atoms (int):
            Channels for the FourierPE.
        mlp_sizes (List[int]):
            Hidden layer sizes for the SineMLP.
        output_dim (int):
            Final output dimensionality.
        input_dim (int, keyword-only):
            Dimensionality d of x.
        pe_kwargs (dict, optional):
            Extra args for `FourierPE(…)` (e.g. `omega0`).
        mlp_kwargs (dict, optional):
            Extra args for `SineMLP(…)` (e.g. `omega0`).

    Input:
        - **x**: Tensor `(..., input_dim)`.
    Output:
        - Tensor `(..., output_dim)`.

    """

    def __init__(
        self,
        num_atoms: int,
        mlp_sizes: List[int],
        output_dim: int,
        *,
        input_dim: int,
        pe_kwargs: Optional[dict] = None,
        mlp_kwargs: Optional[dict] = None,
    ) -> None:

        super(SirenNet, self).__init__()

        self.pe = FourierPE(
            num_atoms=num_atoms, input_dim=input_dim, **(pe_kwargs or {})
        )

        self.mlp = SineMLP(
            input_features=self.pe.num_atoms,
            output_features=output_dim,
            hidden_sizes=mlp_sizes,
            **(mlp_kwargs or {}),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pe(x)
        x = self.mlp(x)

        return x
    
class HerglotzSirenNet(nn.Module):
    r"""HerglotzSirenNet.

    Cartesian‐coordinate SIREN that uses a learnable Herglotz positional encoding
    in place of the usual Fourier features.

    Workflow:
        x ∈ ℝᵈ                          # user‐supplied Cartesian input
          └──HerglotzPE(num_atoms,d)──▶ ψ(x) ∈ ℝⁿ
                              └─SineMLP──▶ output ∈ ℝᵒ

    Parameters:
        num_atoms (int):
            Number of atoms (channels) in the HerglotzPE.  
            The PE buffer “A” will have shape `(num_atoms, input_dim)`.
        mlp_sizes (List[int]):
            Hidden‐layer widths for the sine‐activated MLP.  
            e.g. `[64,64]` for two hidden layers of 64 units each.
        output_dim (int):
            Dimensionality of the final output (o).
        input_dim (int, keyword-only):
            Dimensionality d of each input vector x.  
            Must match the HerglotzPE’s `input_dim` requirement (commonly 3).
        pe_kwargs (Optional[dict]):
            Extra keyword args forwarded to `HerglotzPE(…)`.  
            See `HerglotzPE` docstring for full parameter list.
        mlp_kwargs (Optional[dict]):
            Extra keyword args forwarded to `SineMLP(…)` (e.g. `omega0`).

    Input:
        - **x**: Tensor of shape `(..., input_dim)`, in Cartesian coords.
    Output:
        - Tensor of shape `(..., output_dim)`.

    """

    def __init__(
        self,
        num_atoms: int,
        mlp_sizes: List[int],
        output_dim: int,
        *,
        input_dim: int,
        pe_kwargs: Optional[dict] = None,
        mlp_kwargs: Optional[dict] = None,
    ) -> None:

        super(HerglotzSirenNet, self).__init__()

        self.pe = HerglotzPE(
            num_atoms=num_atoms, 
            input_dim=input_dim, 
            **(pe_kwargs or {})
        )

        self.mlp = SineMLP(
            input_features=self.pe.num_atoms,
            output_features=output_dim,
            hidden_sizes=mlp_sizes,
            **(mlp_kwargs or {}),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pe(x)
        x = self.mlp(x)

        return x


class SphericalSirenNet(nn.Module):

    r"""SphericalSirenNet.

    Angular SIREN on the 2‐sphere: encodes (θ,φ) via real spherical harmonics,
    then processes with a sine‐activated MLP.

    Workflow:
        x_sph ∈ S²         # (θ,φ) in radians
            └─SphericalHarmonicsPE──▶ ψ(x) ∈ ℝⁿ
                                        └─SineMLP──▶ output ∈ ℝᵒ

    Parameters:
        L (int):
            Maximum spherical‐harmonic degree. PE will output `(L+1)**2` channels.
        mlp_sizes (List[int]):
            Hidden‐layer widths for the SineMLP.
        output_dim (int):
            Dimensionality of the network’s final output.
        seed (int, optional):
            RNG seed for reproducible behavior in `SphericalHarmonicsPE`.
        pe_kwargs (Optional[dict]):
            Extra keyword args for `SphericalHarmonicsPE(…)`.
        mlp_kwargs (Optional[dict]):
            Extra keyword args for `SineMLP(…)`.

    Input:
        - **x**: Tensor of shape `(..., 2)`, representing (θ,φ).
    Output:
        - Tensor of shape `(..., output_dim)`.

    """

    def __init__(
        self, 
        L : int,
        mlp_sizes : List[int],
        output_dim : int,
        *,
        seed : Optional[int] = None,
        pe_kwargs : Optional[dict] = None,
        mlp_kwargs : Optional[dict] = None,
    ) -> None:
    
        super(SphericalSirenNet, self).__init__()

        self.pe = SphericalHarmonicsPE(
            L=L,
            seed = seed,
            **(pe_kwargs or {}),
        )

        self.mlp = SineMLP(
            input_features=self.pe.num_atoms,
            output_features=output_dim,
            hidden_sizes=mlp_sizes,
            **(mlp_kwargs or {}),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.shape[-1] != 2:
            raise ValueError(
                f"Expected input shape (..., 2) for spherical coordinates (θ, φ), but got {x.shape}."
            )

        x = self.pe(x)
        x = self.mlp(x)

        return x
    

class IrregularSolidSirenNet(nn.Module):
    r"""IrregularSolidSirenNet.

    Solid‐harmonic SIREN on ℝ³ with **irregular** (decaying) basis functions.

    Workflow:
        x_sph ∈ ℝ³           # (r,θ,φ)
          └──rtp_to_r3──▶ x_cart ∈ ℝ³
                            └─IrregularSolidHarmonicsPE──▶ ψ(x) ∈ ℝⁿ
                                                           └─SineMLP──▶ output ∈ ℝᵒ

    Parameters:
        L (int):
            Maximum harmonic degree; `num_atoms=(L+1)**2`.
        mlp_sizes (List[int]):
            Hidden‐layer widths for the SineMLP.
        output_dim (int):
            Dimensionality of the final output.
        seed (int, optional):
            RNG seed for `IrregularSolidHarmonicsPE`.
        pe_kwargs (Optional[dict]):
            Extra keyword args forwarded to `IrregularSolidHarmonicsPE(…)`.
        mlp_kwargs (Optional[dict]):
            Extra keyword args forwarded to `SineMLP(…)`.
            
    Input:
        - **x**: Tensor of shape `(..., 3)`, representing (r,θ,φ).
    Output:
        - Tensor of shape `(..., output_dim)`.

    """

    def __init__(
        self, 
        L : int,
        mlp_sizes : List[int],
        output_dim : int,
        *,
        seed : Optional[int] = None,
        pe_kwargs : Optional[dict] = None,
        mlp_kwargs : Optional[dict] = None,
    ) -> None:
        

        super(IrregularSolidSirenNet, self).__init__()

        self.pe = IrregularSolidHarmonicsPE(
            L=L,
            seed=seed,
            ** (pe_kwargs or {}),
        )

        self.mlp = SineMLP(
            input_features=self.pe.num_atoms,
            output_features=output_dim,
            hidden_sizes=mlp_sizes,
            **(mlp_kwargs or {}),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rtp_to_r3(x)
        x = self.pe(x)
        x = self.mlp(x)

        return x


class RegularSolidSirenNet(nn.Module):
    r"""RegularSolidSirenNet.

    Solid‐harmonic SIREN on ℝ³ using **regular** solid harmonics (features grow like rˡ).

    Workflow:
        x_sph ∈ ℝ³           # input spherical coords (r, θ, φ)
          └──rtp_to_r3──▶ x_cart ∈ ℝ³
                            └─RegularSolidHarmonicsPE(L)──▶ ψ(x) ∈ ℝⁿ
                                                            └─SineMLP──▶ output ∈ ℝᵒ

    Parameters:
        L (int):
            Maximum spherical‐harmonic degree.  
            The PE will produce `num_atoms = (L+1)**2` channels.
        mlp_sizes (List[int]):
            Sizes of hidden layers for the sine‐activated MLP (e.g. `[64, 64]`).
        output_dim (int):
            Dimensionality o of the network’s final output.
        seed (int, optional):
            Random‐seed for initializing the solid‐harmonic basis in the PE.
        pe_kwargs (dict, optional):
            Additional keyword arguments forwarded to `RegularSolidHarmonicsPE`.
            See `RegularSolidHarmonicsPE` docstring for the full API.
        mlp_kwargs (dict, optional):
            Additional keyword arguments forwarded to `SineMLP`.
    Input:
        - **x**: Tensor of shape `(..., 3)`, representing spherical coordinates `(r ≥ 0, θ ∈ [0,π], φ ∈ [0,2π))`.  

    Output:
        - Tensor of shape `(..., output_dim)`, the MLP’s prediction per input point.

    """

    def __init__(
        self, 
        L :int,
        mlp_sizes : List[int],
        output_dim : int,
        *,
        seed : Optional[int] = None,
        pe_kwargs : Optional[dict] = None,
        mlp_kwargs : Optional[dict] = None,
    ) -> None:
        
    
        super(RegularSolidSirenNet, self).__init__()

        self.pe = RegularSolidHarmonicsPE(
            L=L,
            seed=seed,
            ** (pe_kwargs or {}),
        )

        self.mlp = SineMLP(
            input_features=self.pe.num_atoms,
            output_features=output_dim,
            hidden_sizes=mlp_sizes,
            **(mlp_kwargs or {}),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rtp_to_r3(x)
        x = self.pe(x)
        x = self.mlp(x)

        return x
    


