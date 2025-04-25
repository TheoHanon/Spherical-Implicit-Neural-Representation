import torch
import torch.nn as nn
import math
import warnings
from collections import OrderedDict


from .rotations import QuaternionRotation
from .third_party.locationencoder.sh import SH

from typing import Optional, List
from abc import ABC, abstractmethod
import inspect

__all__ = [
    "RegularHerglotzPE",
    "IrregularHerglotzPE",
    "HerglotzPE",
    "FourierPE",
    "SphericalHarmonicsPE",
    "RegularSolidHarmonicsPE",
    "IrregularSolidHarmonicsPE",
    "get_positional_encoding",
]


def _generate_herglotz_vector(dim, gen : Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Generates a complex vector (atom) for the Herglotz encoding.

    The vector is constructed by generating two independent random vectors,
    normalizing them, and ensuring the imaginary part is orthogonal to the real part.

    Parameters:
        dim (int): The dimension of the vector.
        gen (Optional[torch.Generator]): A random number generator for reproducibility. Default is None.

    Returns:
        torch.Tensor: A complex tensor representing the atom (dtype=torch.complex64).
    """

    a_R = torch.randn(dim, dtype=torch.float32, generator=gen)
    a_R /= (2**0.5) * torch.norm(a_R)
    a_I = torch.randn(dim, dtype=torch.float32, generator=gen)
    a_I -= 2 * torch.dot(a_I, a_R) * a_R  # Orthogonalize a_I with respect to a_R
    a_I /= (2**0.5) * torch.norm(a_I)

    return torch.complex(a_R, a_I)


class _PositionalEncoding(ABC, nn.Module):
    r"""Abstract base class for positional encoding modules.

    This class defines the interface for generating a positional encoding,
    denoted by :math:`\psi(x)`, from an input :math:`x \in \mathbb{R}^{\text{input_dim}}`.
    The encoding is parameterized by the number of atoms and may use an optional random seed for reproducibility.

    Parameters:
        num_atoms (int): Number of encoding atoms.
        input_dim (int): Dimensionality of the input.
        seed (Optional[int]): Random seed for reproducibility.

    Attributes:
        num_atoms (int): Number of encoding atoms.
        input_dim (int): Dimensionality of the input.
        gen (Optional[torch.Generator]): Random number generator (if a seed is provided).
    """

    def __init__(
        self, num_atoms: int, input_dim: int, seed: Optional[int] = None
    ) -> None:
        super(_PositionalEncoding, self).__init__()
        self.num_atoms = num_atoms
        self.input_dim = input_dim

        self.gen: Optional[torch.Generator] = None

        if seed is not None:
            self.gen = torch.Generator()
            self.gen.manual_seed(seed)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def extra_repr(self) -> str:
        return f"num_atoms={self.num_atoms}, " f"input_dim={self.input_dim}"

class SphericalHarmonicsPE(_PositionalEncoding):
    r"""Real Spherical Harmonics Positional Encoding.

    Encodes a direction given by polar angles :math:`(\theta,\phi)\in\mathbb{R}^2` into
    a vector of real spherical harmonics up to degree :math:`L`.  By default
    the number of output channels is :math:`(L+1)^2`, but you may override it
    via the `num_atoms` keyword.

    The real spherical harmonics are computed as

    .. math::
        Y_\ell^m(\theta,\phi)
        = N_{\ell,m}\,P_\ell^{|m|}(\cos\theta)\times
        \begin{cases}
          \cos(m\phi), & m\ge0,\\
          \sin(|m|\phi), & m<0,
        \end{cases}

    where :math:`\ell=0,\dots,L` and :math:`m=-\ell,\dots,\ell`.  Internally
    this class builds two Python lists `ls_list` and `ms_list` of length
    `num_atoms` to drive a TorchScript-compatible loop over all `(ℓ,m)` pairs.

    Parameters:
        L (Optional[int]):
            Maximum harmonic order.  If provided, the default
            `num_atoms=(L+1)**2`.  Must be ≥0.
        num_atoms (Optional[int], keyword-only):
            Explicit number of output channels; if given, overrides `(L+1)**2`.
        input_dim (int, optional):
            Dimensionality of each input `x`; must be 2 (for `(θ,φ)`). Default: 2.
        seed (Optional[int], keyword-only):
            Random seed for reproducible behavior (unused here).

    Attributes:
        num_atoms (int):
            Number of output channels (atoms).
        input_dim (int):
            Expected dimensionality of the input tensor (2).
        ls_list (List[int]):
            List of ℓ indices for each channel.
        ms_list (List[int]):
            List of m indices for each channel.
    """

    def __init__(self, 
                L : Optional[int] = None, 
                *,
                num_atoms: Optional[int] = None,
                input_dim : int = 2,
                seed: Optional[int] = None,) -> None:
        
        if num_atoms is not None and L is not None:
            warnings.warn(
                "Both `num_atoms` and `L` were given; ignoring `L` and using the explicit `num_atoms`.",
                UserWarning
            )
        elif num_atoms is None:

            if L is None:
                raise ValueError("Either `num_atoms` or `L` must be provided.")
            
            num_atoms = (L + 1)**2

        L_upper = L if L is not None else math.ceil(math.sqrt(num_atoms)) - 1
        
        super(SphericalHarmonicsPE, self).__init__(num_atoms, input_dim = input_dim, seed=seed)

        self.ms_list : List[int] = [m for l in range(L_upper+1) for m in range(-l, l+1)][:self.num_atoms]
        self.ls_list : List[int] = [l for l in range(L_upper+1) for m in range(-l, l+1)][:self.num_atoms]

    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        
        theta, phi = x[..., 0], x[..., 1]
        outs : List[torch.Tensor] = []


        for l, m in zip(self.ls_list, self.ms_list):
            outs.append(SH(l, m, theta, phi))

        return torch.stack(outs, dim=-1)
    

class RegularSolidHarmonicsPE(SphericalHarmonicsPE):
    r"""Regular Solid Harmonics Positional Encoding.

    Extends `SphericalHarmonicsPE` to encode a full 3-D point
    :math:`(r,\theta,\phi)\in\R^3` into the regular solid harmonics basis
    functions

    .. math::
       R_\ell^m(r,\theta,\phi)
       = r^\ell\,Y_\ell^m(\theta,\phi),

    for :math:`\ell=0,\dots,L` and :math:`m=-\ell,\dots,\ell`.  Output shape is
    `(..., num_atoms)` with `num_atoms=(L+1)**2` by default.

    Parameters:
        L (Optional[int]):
            Maximum harmonic order; default `num_atoms=(L+1)**2`.
        num_atoms (Optional[int], keyword-only):
            Override the number of basis functions.
        seed (Optional[int], keyword-only):
            Random seed for reproducible behavior.

    Attributes:
        exponents (torch.Tensor, buffer):
            Float tensor of shape `(1, num_atoms)` containing the exponent ℓ for
            each channel, used to compute `r**ℓ`.
    """

    def __init__(self, L:Optional[int] = None, *, num_atoms : Optional[int] = None, seed : Optional[int] = None) -> None:
        super(RegularSolidHarmonicsPE, self).__init__(L = L, num_atoms=num_atoms, seed = seed, input_dim = 3) 

        L_upper = L if L is not None else math.ceil(math.sqrt(self.num_atoms)) - 1
        exps = torch.cat([torch.full((2*l+1,), l) for l in range(L_upper+1)])[:self.num_atoms]
        self.register_buffer("exponents", exps.view(1, -1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x[..., 0]
        ylm = super().forward(x[..., 1:3])
        return ylm * r.unsqueeze(-1).pow(self.exponents)  


class IrregularSolidHarmonicsPE(SphericalHarmonicsPE):

    r"""Irregular Solid Harmonics Positional Encoding.

    Encodes a point :math:`(r,\theta,\phi)\in\mathbb{R}^3` into the irregular solid
    harmonics basis

    .. math::
       I_\ell^m(r,\theta,\phi)
       = \frac{1}{r^{\ell+1}}\,Y_\ell^m(\theta,\phi),

    which decays like :math:`1/r` for large radius.  Useful when you want
    features that vanish at infinity.  Output shape is
    `(..., num_atoms)`, with `num_atoms=(L+1)**2` by default.

    Inherits all constructor arguments from `RegularSolidHarmonicsPE`.

    Attributes:
        exponents (torch.Tensor, buffer):
            Float tensor of shape `(1, num_atoms)` containing :math: `\ell` for each channel,
            used to compute :math:`1/r^{\ell+1}`.
    """

    def __init__(self, L:Optional[int] = None, *, num_atoms : Optional[int] = None, seed : Optional[int] = None) -> None:
        super(IrregularSolidHarmonicsPE, self).__init__(L = L, num_atoms=num_atoms, seed = seed, input_dim = 3) 
        L_upper = L if L is not None else math.ceil(math.sqrt(self.num_atoms)) - 1
        exps = torch.cat([torch.full((2*l+1,), l) for l in range(L_upper+1)])[:self.num_atoms]
        self.register_buffer("exponents", exps.view(1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x[..., 0]
        ylm = super().forward(x[..., 1:3])
        return ylm * r.unsqueeze(-1).pow(-(self.exponents + 1))


class HerglotzPE(_PositionalEncoding):
    r"""Herglotz Positional Encoding.

    This module implements the Herglotz map with optional quaternion‐based rotation. Each atom is a
    complex n dim vector whose real and imaginary parts are unit‐length and orthogonal,
    initialized either by specifying:

    • `num_atoms`: an explicit total count, **or**  
    • `L`: a “stacking depth” so that `num_atoms = (L+1)**2`.  

    A radial reference `rref` scales the projections so that for ‖x‖<rref the
    encoding remains bounded by 1.  

    The forward mapping is given by

    .. math::
        \psi(x) \;=\;\sin\bigl(w_{R}\,a_{\Im} + b_{I}\bigr)\;\exp\!\bigl(w_{R}\,(a_{\Im} - a_{norm}) + b_{R}\bigr)

    where :
        - :math:`a_{\Re} = \frac{\Re \{x^\top A\}}{rref}`.
        - :math:`a_{\Im} = \frac{\Im\{x^\top A\}}{rref}`.
        - :math:`a_{norm} = \texttt{norm\_const}`.

    The model consider input x = (x_1, ..., x_n) in R^n. However, there are not theoretical guarantees that the model will perform well in n != 3. If your input is in spherical coordinates, you must convert them in cartesian before feeding them to the model.

    Parameters:
        num_atoms (Optional[int]):
            Explicit number of atoms; if set, overrides `L`.
        input_dim (int, optional):
            Dimensionality of the input. Default: 3.
        *  
        bias (bool, optional):
            If True, includes bias terms for the sine and exponential components. Default: True.
        L (Optional[int], keyword-only):
            Initialize a number of atoms equals to the number of spherical harmonics up to order L.  Used when `num_atoms` is None.
        seed (Optional[int], keyword-only):
            RNG seed for reproducible atom initialization.
        rref (float, keyword-only):
            Radial reference scale; for ‖x‖<rref, outputs ≤ 1. Default: 1.0.
        init_exponents (bool, keyword-only):
            If True, initialize `w_R` such that it activates the first L spherical harmonic orders (moments included). Recommended to use with `L` specified and `normalize` set to True.
        normalize (bool, keyword-only):
            If False, uses 1/√2 as the internal normalization constant; else 0. Bounding up the atoms by <= exp(b_R) if r <=rref. Default: True.
        rotation (bool, keyword-only):
            If True, applies per-atom quaternion rotation. Default: True.

    Attributes:
        A (Tensor, buffer):
            Complex atoms of shape `(num_atoms, 3)`, dtype `torch.complex64`.
        rref (Tensor, buffer):
            Radial reference.
        w_R (Parameter):
            Frequency weights for both sine and exponential terms.
        b_I (Parameter):
            Bias for the sine term.
        b_R (Parameter):
            Bias for the exponential term.
        norm_const_buf (Tensor, buffer):
            Normalization constant (`0.0` or `1/√2`).
        quaternion_rotation (Module or callable):
            Applies each atom’s quaternion in `forward()`.
    """



    def __init__(self, 
                 num_atoms : Optional[int] = None, 
                 input_dim: int = 3, 
                 *,
                 bias : bool = True,
                 L: Optional[int] = None, 
                 seed: Optional[int] = None, 
                 rref : float = 1.0, 
                 init_exponents: bool =  True, 
                 normalize : bool = True,) -> None:
        
        if num_atoms is not None and L is not None:
            warnings.warn(
                "Both `num_atoms` and `L` were given; ignoring `L` and using the explicit `num_atoms`.",
                UserWarning
            )
        elif num_atoms is None:

            if L is None:
                raise ValueError("Either `num_atoms` or `L` must be provided.")
            
            num_atoms = (L + 1)**2

        super(HerglotzPE, self).__init__(
            num_atoms= num_atoms, 
            input_dim=input_dim, 
            seed=seed
        )
    
        A = torch.stack(
            [_generate_herglotz_vector(self.input_dim, self.gen) for i in range(self.num_atoms)],
            dim=0
        )
        self.register_buffer("A", A)

        self.w_R = nn.Parameter(torch.zeros(self.num_atoms))

        if init_exponents:
            L_upper = math.ceil(math.sqrt(self.num_atoms)) - 1
            exps = torch.tensor(
                [l for l in range(L_upper+1) for _ in range(2*l + 1)],
                dtype=torch.float32,
                device=self.w_R.device
            ) / math.e
            exps = exps[:self.num_atoms]
            with torch.no_grad():
                self.w_R.copy_(exps)
        else:
            nn.init.uniform_(self.w_R, -1 / self.input_dim, 1 / self.input_dim)
        
        if bias :
            self.b_I = nn.Parameter(torch.zeros(self.num_atoms))
            self.b_R = nn.Parameter(torch.zeros(self.num_atoms))
        else:
            self.register_buffer("b_I", torch.zeros(self.num_atoms))
            self.register_buffer("b_R", torch.zeros(self.num_atoms))

        norm_const = 0. if not normalize else 1.0 / math.sqrt(2)
        self.register_buffer("norm_const_buf", torch.tensor(norm_const))
        self.register_buffer("rref_buf", torch.tensor(rref))
         
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_c = x.to(self.A.dtype)
        ax = torch.matmul(x_c, self.A.t())

        ax_R = ax.real / self.rref_buf
        ax_I = ax.imag / self.rref_buf

        sin_term = torch.sin(self.w_R * ax_I  + self.b_I)
        exp_term = torch.exp(self.w_R * (ax_R - self.norm_const_buf) + self.b_R)
    
        return sin_term * exp_term


        
class RegularHerglotzPE(_PositionalEncoding):
    r"""Regular Herglotz Map Positional Encoding.

    This module implements the **regular** Herglotz map with optional quaternion‐based rotation. Each atom is a
    complex 3D vector whose real and imaginary parts are unit‐length and orthogonal,
    initialized either by specifying:

    • `num_atoms`: an explicit total count, **or**  
    • `L`: a “stacking depth” so that `num_atoms = (L+1)**2`.  

    A radial reference `rref` scales the projections so that for ‖x‖<rref the
    encoding remains bounded by 1.  If `rotation=True`, each atom is first
    rotated by a learnable quaternion in ℝ³.

    The forward mapping is given by

    .. math::
        \psi(x) \;=\;\sin\bigl(w_{R}\,a_{\Im} + b_{I}\bigr)\;\exp\!\bigl(w_{R}\,(a_{\Im} - a_{norm}) + b_{R}\bigr)

    where :
        - :math:`a_{\Re} = \frac{\Re \{x^\top A\}}{rref}`.
        - :math:`a_{\Im} = \frac{\Im\{x^\top A\}}{rref}`.
        - :math:`a_{norm} = \texttt{norm\_const}`.

    We only consider input :math:`x = (x, y, z) \in \mathbb{R}^3`. If your input is in spherical coordinates, you must convert them in cartesian before feeding them to the model.

    Parameters:
        num_atoms (Optional[int]):
            Explicit number of atoms; if set, overrides `L`.
        *  
        bias (bool, optional):
            If True, includes bias terms for the sine and exponential components. Default: True.
        L (Optional[int], keyword-only):
            Initialize a number of atoms equals to the number of spherical harmonics up to order L.  Used when `num_atoms` is None.
        seed (Optional[int], keyword-only):
            RNG seed for reproducible atom initialization.
        rref (float, keyword-only):
            Radial reference scale; for ‖x‖<rref, outputs ≤ 1. Default: 1.0.
        init_exponents (bool, keyword-only):
            If True, initialize `w_R` such that it activates the first L spherical harmonic orders (moments included). Recommended to use with `L` specified and `normalize` set to True.
        normalize (bool, keyword-only):
            If False, uses 1/√2 as the internal normalization constant; else 0. Bounding up the atoms by <= exp(b_R) if r <=rref. Default: True.
        rotation (bool, keyword-only):
            If True, applies per-atom quaternion rotation. Default: True.

    Attributes:
        A (Tensor, buffer):
            Complex atoms of shape `(num_atoms, 3)`, dtype `torch.complex64`.
        rref (Tensor, buffer):
            Radial reference.
        w_R (Parameter):
            Frequency weights for both sine and exponential terms.
        b_I (Parameter):
            Bias for the sine term.
        b_R (Parameter):
            Bias for the exponential term.
        norm_const_buf (Tensor, buffer):
            Normalization constant (`0.0` or `1/√2`).
        quaternion_rotation (Module or callable):
            Applies each atom’s quaternion in `forward()`.
    """



    def __init__(self, 
                 num_atoms : Optional[int] = None, 
                 *,
                 bias : bool = True,
                 L: Optional[int] = None, 
                 seed: Optional[int] = None, 
                 rref : float = 1.0, 
                 init_exponents: bool =  False, 
                 normalize : bool = True,
                 rotation : bool = True,) -> None:
        
        if num_atoms is not None and L is not None:
            warnings.warn(
                "Both `num_atoms` and `L` were given; ignoring `L` and using the explicit `num_atoms`.",
                UserWarning
            )
        elif num_atoms is None:

            if L is None:
                raise ValueError("Either `num_atoms` or `L` must be provided.")
            
            num_atoms = (L + 1)**2
        
        super(RegularHerglotzPE, self).__init__(
            num_atoms= num_atoms, 
            input_dim=3, 
            seed=seed
        )
    
        A = torch.stack(
            [_generate_herglotz_vector(self.input_dim, self.gen) for i in range(self.num_atoms)],
            dim=0
        )
        self.register_buffer("A", A)

        self.w_R = nn.Parameter(torch.zeros(self.num_atoms))
        if init_exponents:
            L_upper = math.ceil(math.sqrt(self.num_atoms)) - 1
            exps = torch.tensor(
                [l for l in range(L+1) for _ in range(2*l + 1)],
                dtype=torch.float32,
                device=self.w_R.device
            ) / math.e
            with torch.no_grad():
                self.w_R.copy_(exps)
        else:
            nn.init.uniform_(self.w_R, -1 / self.input_dim, 1 / self.input_dim)
        

        if bias :
            self.b_I = nn.Parameter(torch.zeros(self.num_atoms))
            self.b_R = nn.Parameter(torch.zeros(self.num_atoms))
        else:
            self.register_parameter("b_I", torch.zeros(self.num_atoms))
            self.register_parameter("b_R", torch.zeros(self.num_atoms))

        norm_const = 0. if normalize else 1.0 / math.sqrt(2)
        self.register_buffer("norm_const_buf", torch.tensor(norm_const))
        self.register_buffer("rref_buf", torch.tensor(rref))
        self.quaternion_rotation = QuaternionRotation(self.num_atoms, self.gen) if rotation else lambda x : x
        
        
    def _rotate_atoms(self) -> torch.Tensor:
        A_rotated_real = self.quaternion_rotation(self.A.real)
        A_rotated_imag = self.quaternion_rotation(self.A.imag)

        return torch.complex(A_rotated_real, A_rotated_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
                
        A_rotated = self._rotate_atoms()

        x_c = x.to(A_rotated.dtype)
        ax = torch.matmul(x_c, A_rotated.t())

        ax_R = ax.real / self.rref_buf
        ax_I = ax.imag / self.rref_buf

        sin_term = torch.sin(self.w_R * ax_I  + self.b_I)
        exp_term = torch.exp(self.w_R * (ax_R - self.norm_const_buf) + self.b_R)
    
        return sin_term * exp_term

        
class IrregularHerglotzPE(RegularHerglotzPE):
    r"""Irregular Herglotz Map Positional Encoding.

    This variant of the Herglotz map applies a **1/‖x‖** decay so that features
    smoothly vanish at large radius. All initialization options
    (`num_atoms` vs. `L`, `rref`, `init_exponents`, `normalize`, `rotation`)
    carry over from the regular version.

    After (optional) quaternion rotation, we compute

    .. math::
        \psi(x) \;=\; \frac{1}{r}\,\sin\bigl(w_{R}\,a_{\Im} + b_{I}\bigr)\;\exp\!\bigl(w_{R}\,(a_{\Re} - a_{norm}) + b_{R}\bigr)

    where :
        - :math:`a_{\Re} = \frac{\Re \{x^\top A\}}{r} \frac{r_{ref}}{r}`. 
        - :math:`a_{\Im} = \frac{\Im\{x^\top A\}}{r} \frac{r_{ref}}{r}`.
        - :math:`a_{norm} = \texttt{norm\_const}`


    We only consider input :math:`x = (x, y, z) \in \mathbb{R}^3`. If your input is in spherical coordinates, you must convert them in cartesian before feeding them to the model.

    Here the extra factors of **1/r** ensure that as ‖x‖→∞, both the sine
    and exponential terms decay like 1/‖x‖, yielding a positional encoding
    that vanishes at infinity.

    Parameters:
        num_atoms (Optional[int]):
            Explicit number of atoms; if set, overrides `L`.
        *  
        L (Optional[int], keyword-only):
            Stacking depth. Used when `num_atoms` is None.
        seed (Optional[int], keyword-only):
            RNG seed for reproducible atom initialization.
        rref (float, keyword-only):
            Radial reference scale. Default: 1.0.
        init_exponents (bool, keyword-only):
            If True, initialize `w_R` from harmonic orders up to L.
        normalize (bool, keyword-only):
            If False, uses 1/√2 as the internal normalization constant; else 0.
            Bounds the atom responses by ≤ exp(b_R) when r ≥ rref. Default: True.
        rotation (bool, keyword-only):
            If True, applies per-atom quaternion rotation. Default: True.

    Attributes:
        (inherited from `RegularHerglotzPE`)
    """

    def forward(self, x):
            
        A_rotated = self._rotate_atoms()
        
        x_c = x.to(A_rotated.dtype)
        r = x_c.norm(dim = -1, keepdim = True).clamp_min(1e-6)
        ax = torch.matmul(x_c, A_rotated.t())
    
        ax_R = (ax.real / r) * (self.rref_buf/r)
        ax_I = (ax.imag / r) * (self.rref_buf/r)

        sin_term = torch.sin(self.w_R * ax_I + self.b_I)
        exp_term = torch.exp(self.w_R * (ax_R - self.norm_const_buf) + self.b_R)
 
        return  (1/r) * sin_term * exp_term


class FourierPE(_PositionalEncoding):
    r"""Fourier Positional Encoding.

    Computes the positional encoding :math:`\psi(x)` by applying a learnable linear transformation followed by a sinusoidal activation.
    For an input :math:`x`, the encoding is given by

    .. math::
        z = \Omega(x),
        \quad
        \psi(x) = \sin\bigl(\omega_0\,z\bigr),

    where :math:`\Omega` is a linear mapping from :math:`\mathbb{R}^{\text{input_dim}}` to
    :math:`\mathbb{R}^{\text{num_atoms}}` and :math:`\omega_0` is a frequency factor.

    Parameters:
        num_atoms (int): Number of output features (atoms).
        input_dim (int): Dimensionality of the input.
        bias (bool, optional): If True, the linear mapping includes a bias term (default: True).
        seed (Optional[int], optional): Seed for reproducibility.
        omega0 (float, optional): Frequency factor applied to the sinusoidal activation (default: 1.0).

    Attributes:
        omega0 (torch.Tensor): Buffer holding the frequency factor.
        Omega (nn.Linear): Linear layer mapping :math:`\mathbb{R}^{\text{input_dim}}` to :math:`\mathbb{R}^{\text{num_atoms}}`.
    """

    def __init__(
        self,
        num_atoms: int,
        input_dim: int,
        bias: bool = True,
        seed: Optional[int] = None,
        omega0: float = 1.0,
    ) -> None:

        super(FourierPE, self).__init__(
            num_atoms=num_atoms, input_dim=input_dim, seed=seed
        )
        self.register_buffer("omega0", torch.tensor(omega0, dtype=torch.float32))
        self.Omega = nn.Linear(self.input_dim, self.num_atoms, bias)

        with torch.no_grad():
            self.Omega.weight.uniform_(-1 / self.input_dim, 1 / self.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Omega(x)
        return torch.sin(self.omega0 * x)



class ClassInstantier(OrderedDict):
    r"""Helper class for instantiating classes with default parameters.

    This class wraps an OrderedDict to allow lazy instantiation of classes.
    When an item is accessed, it returns a lambda function that creates an instance of the class,
    merging default keyword arguments with those provided by the user.
    """

    def __getitem__(self, key):
        content = super().__getitem__(key)
        if isinstance(content, tuple):
            cls, default_kwargs = content
        else:
            cls, default_kwargs = content, {}

        return lambda **kwargs: cls(**{**default_kwargs, **kwargs})


PE2CLS = {
    "regular_herglotz": (RegularHerglotzPE, {}),
    "irregular_herglotz": (IrregularHerglotzPE, {}),
    "herglotz": (HerglotzPE, {"init_exponents" : False}),
    "fourier": (FourierPE, {}),
    "spherical_harmonics": (SphericalHarmonicsPE, {}),
    "solid_harmonics": (RegularSolidHarmonicsPE, {}),
    "irregular_solid_harmonics": (IrregularSolidHarmonicsPE, {}),
}

PE2FN = ClassInstantier(PE2CLS)


def get_positional_encoding(pe: str, **kwargs) -> nn.Module:
    r"""Construct a positional encoding module.

    This function returns an instance of a positional encoding module corresponding to the specified

    type. The available types are: ``"herglotz"``, ``"solid_herglotz"``, ``"irregular_solid_herglotz"``, ``"fourier"``, ``"spherical_harmonics"``, ``"solid_harmonics"``, and ``"irregular_solid_harmonics"``.
    Additional parameters are forwarded to the constructor of the chosen module.

    Parameters:
        pe (str): Identifier for the type of positional encoding. Must be one of ``"herglotz"``, ``"solid_herglotz"``, ``"irregular_solid_herglotz"``, ``"fourier"``, ``"spherical_harmonics"``, ``"solid_harmonics"``, and ``"irregular_solid_harmonics"``.
        **kwargs: Additional keyword arguments to configure the positional encoding module. Drop any kwargs not in the constructor.

    Returns:
        nn.Module: An instance of the specified positional encoding module.

    Raises:
        ValueError: If the specified positional encoding type is not supported.
    """

    if pe not in PE2CLS:
        raise ValueError(f"Invalid positional encoding: {pe}")

    cls, defaults = PE2CLS[pe]
    sig = inspect.signature(cls.__init__)
    # drop any kwargs not in the constructor
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return cls(**defaults, **filtered)

