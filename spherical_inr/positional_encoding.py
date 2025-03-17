import torch
import torch.nn as nn
from collections import OrderedDict

from typing import Optional
from abc import ABC, abstractmethod

__all__ = [
    "RegularHerglotzPE",
    "IregularHerglotzPE",
    "FourierPE",
    "StackedRegularHerglotzMapPE",
    "StackedIregularHerglotzMapPE",
    "get_positional_encoding",
]


def _generate_herglotz_vector(dim, gen : Optional[int] = None) -> torch.Tensor:
    """
    Generates a complex vector (atom) for the Herglotz encoding.

    The vector is constructed by generating two independent random vectors,
    normalizing them, and ensuring the imaginary part is orthogonal to the real part.

    Parameters:
        input_dim (int): The dimension of the vector (2 or 3).
        generator (Optional[torch.Generator]): A random number generator for reproducibility. Default is None.

    Returns:
        torch.Tensor: A complex tensor representing the atom (dtype=torch.complex64).
    """

    a_R = torch.randn(dim, dtype=torch.float32, generator=gen)
    a_R /= (2**0.5) * torch.norm(a_R)
    a_I = torch.randn(dim, dtype=torch.float32, generator=gen)
    a_I -= 2 * torch.dot(a_I, a_R) * a_R  # Orthogonalize a_I with respect to a_R
    a_I /= (2**0.5) * torch.norm(a_I)

    return a_R + 1j * a_I

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


class RegularHerglotzPE(_PositionalEncoding):
    r"""Regular Herglotz Positional Encoding.

    Generates a positional encoding :math:`\psi(x)` based on the Herglotz approach.
    Complex encoding atoms are constructed by combining two independent random vectors that are normalized
    and rendered orthogonal. The encoding is defined by

    .. math::
        z = \omega_0 \left[ \left(w_{\mathrm{real}} + i\,w_{\mathrm{imag}}\right) \,(A\,x)
            + \left(b_{\mathrm{real}} + i\,b_{\mathrm{imag}}\right) \right],
        \quad
        \psi(x) = \exp\bigl(-\operatorname{Im}(z)\bigr) \cos\bigl(\operatorname{Re}(z)\bigr),

    where :math:`A` is the matrix of complex atoms, :math:`\omega_0` is a frequency factor, and
    :math:`w_{\mathrm{real}},\,w_{\mathrm{imag}},\,b_{\mathrm{real}},\,b_{\mathrm{imag}}` are learnable parameters.

    Parameters:
        num_atoms (int): Number of atoms to generate.
        input_dim (int): Dimensionality of the input (must be at least 2).
        bias (bool, optional): If True, uses learnable bias parameters (default: True).
        seed (Optional[int], optional): Seed for reproducibility.
        omega0 (float, optional): Frequency factor applied to the encoding (default: 1.0).

    Attributes:
        A (torch.Tensor): Buffer containing the generated complex atoms.
        omega0 (torch.Tensor): Buffer holding the frequency factor.
        w_real (nn.Parameter): Learnable real part of the weights.
        w_imag (nn.Parameter): Learnable imaginary part of the weights.
        bias_real (nn.Parameter or buffer): Real part of the bias.
        bias_imag (nn.Parameter or buffer): Imaginary part of the bias.
    """

    def __init__(
        self,
        num_atoms: int,
        input_dim: int,
        bias: bool = True,
        seed: Optional[int] = None,
        omega0: float = 1.0,
    ) -> None:

        super(RegularHerglotzPE, self).__init__(
            num_atoms=num_atoms, input_dim=input_dim, seed=seed
        )
        if input_dim < 2:
            raise ValueError("Input dimension must be at least 2.")

        A = torch.stack(
            [_generate_herglotz_vector(self.input_dim, self.gen) for i in range(self.num_atoms)],
            dim=0,
        )

        self.register_buffer("A", A)
        self.register_buffer("omega0", torch.tensor(omega0, dtype=torch.float32))

        self.w_real = nn.Parameter(
            torch.empty(self.num_atoms, dtype=torch.float32).uniform_(
                -1 / self.input_dim, 1 / self.input_dim, generator=self.gen
            )
        )
        self.w_imag = nn.Parameter(
            torch.empty(self.num_atoms, dtype=torch.float32).uniform_(
                -1 / self.input_dim, 1 / self.input_dim, generator=self.gen
            )
        )

        if bias is True:
            self.bias_real = nn.Parameter(
                torch.zeros(self.num_atoms, dtype=torch.float32)
            )
            self.bias_imag = nn.Parameter(
                torch.zeros(self.num_atoms, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "bias_real", torch.zeros(self.num_atoms, dtype=torch.float32)
            )
            self.register_buffer(
                "bias_imag", torch.zeros(self.num_atoms, dtype=torch.float32)
            )


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.to(self.A.dtype)
        x = torch.matmul(x, self.A.t())

        x = self.omega0 * (
            (self.w_real + 1j * self.w_imag) * x
            + (self.bias_real + 1j * self.bias_imag)
        )

        return torch.exp(-x.imag) * torch.cos(x.real)

    def extra_repr(self) -> str:
        repr = super().extra_repr()
        return repr + f", omega0={self.omega0.item()}"


class StackedRegularHerglotzMapPE(_PositionalEncoding):
    """
    Stacked Regular Herglotz Map Positional Encoding.

    This module computes a stacked variant of the Regular Herglotz positional encoding.
    It first applies a linear transformation of the input using pre-generated complex atoms.
    Then, it “stacks” the encoding by raising selected components to integer powers corresponding
    to different orders in a harmonic expansion. The total number of atoms is determined as:
        num_atoms = L * (L + 1) // 2,
    where L is the stacking depth.

    Parameters:
        L (int): The stacking depth, determining the highest order of the expansion.
        input_dim (int, optional): Dimensionality of the input (default is 3).
        seed (Optional[int], optional): Seed for reproducibility.

    Attributes:
        L (int): The stacking depth.
        A (torch.Tensor): Buffer containing the generated complex atoms of shape 
            (num_atoms, input_dim), where num_atoms = L*(L+1)//2.

    """
    def __init__(self, L: int, input_dim: int = 3, seed: Optional[int] = None) -> None:
        super(StackedRegularHerglotzMapPE, self).__init__(
            num_atoms=(L+1)*(L+2) // 2, input_dim=input_dim, seed=seed
        )
        A = torch.stack(
            [_generate_herglotz_vector(self.input_dim, self.gen) for i in range(self.num_atoms)],
            dim=0
        )
        exponents = [0]
        for l in range(1, L+1):
            exponents.extend([l] * (l + 1))

        self.L = L
        self.register_buffer("A", A)
        self.register_buffer("exponents", torch.tensor(exponents, dtype=torch.float32))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = x.to(self.A.dtype)
        x = torch.matmul(x, self.A.t())
        x = torch.pow(x, self.exponents.view(1, -1))
                
        return x.real

    
    def extra_repr(self):
        return super().extra_repr() + f", L={self.L}"

        

class IregularHerglotzPE(RegularHerglotzPE):
    r"""Irregular Herglotz Positional Encoding.

    Extends the regular Herglotz encoding by incorporating a normalization factor based on the input norm.
    For an input :math:`x` with Euclidean norm :math:`r = \|x\|`, the encoding is defined by

    .. math::
        z = \omega_0 \left[ \left(w_{\mathrm{real}} + i\,w_{\mathrm{imag}}\right) \frac{A\,x}{r^2}
            + \left(b_{\mathrm{real}} + i\,b_{\mathrm{imag}}\right) \right],
        \quad
        \psi(x) = \frac{1}{r} \exp\bigl(-\operatorname{Im}(z)\bigr) \cos\bigl(\operatorname{Re}(z)\bigr).

    Parameters:
        num_atoms (int): Number of atoms to generate.
        input_dim (int): Dimensionality of the input.
        bias (bool, optional): If True, uses learnable bias parameters (default: True).
        seed (Optional[int], optional): Seed for reproducibility.
        omega0 (float, optional): Frequency factor applied to the encoding (default: 1.0).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.to(self.A.dtype)

        r = torch.norm(x, dim=-1, keepdim=True)
        x = torch.matmul(x, self.A.t())

        x = self.omega0 * (
            (self.w_real + 1j * self.w_imag) * (x / (r * r))
            + (self.bias_real + 1j * self.bias_imag)
        )

        return 1 / r * torch.exp(-x.imag) * torch.cos(x.real)
    
class StackedIregularHerglotzMapPE(StackedRegularHerglotzMapPE):
    """
    Stacked Irregular Herglotz Map Positional Encoding.

    This module extends the stacked regular Herglotz encoding by incorporating a radial normalization.
    The input is first normalized by its norm, then transformed using the pre-generated complex atoms.
    Similar to its regular counterpart, the encoding is “stacked” by raising certain components to 
    integer powers adjusted by the radial factor. This variant is useful when a band-limited 
    irregular solid harmonic expansion is desired.

    Parameters:
        L (int): The stacking depth, determining the highest order of the expansion.
        input_dim (int, optional): Dimensionality of the input (default is 3).
        seed (Optional[int], optional): Seed for reproducibility.

    Attributes:
        L (int): The stacking depth.
        A (torch.Tensor): Buffer containing the generated complex atoms of shape 
            (num_atoms, input_dim), where num_atoms = L*(L+1)//2.

    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.A.dtype)
        r = torch.norm(x, dim=-1, keepdim=True)
        x = torch.matmul(x, self.A.t())
        
        normalized = x / (r * r)
        x = (1/r) * torch.pow(normalized, self.exponents.view(1, -1))

        return x.real

    def extra_repr(self):
        return super().extra_repr() + f", L={self.L}"



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

    def extra_repr(self) -> str:
        repr = super().extra_repr()
        return repr + f", omega0={self.omega0.item()}"


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
    "herglotz": (RegularHerglotzPE, {"bias": True, "omega0": 1.0}),
    "irregular_herglotz": (IregularHerglotzPE, {"bias": True, "omega0": 1.0}),
    "fourier": (FourierPE, {"bias": True, "omega0": 1.0}),
    "stacked_herglotz": (StackedRegularHerglotzMapPE, {}),
    "stacked_irregular_herglotz": (StackedIregularHerglotzMapPE, {}),
}

PE2FN = ClassInstantier(PE2CLS)


def get_positional_encoding(pe: str, **kwargs) -> nn.Module:
    r"""Construct a positional encoding module.

    This function returns an instance of a positional encoding module corresponding to the specified
    type. The available types are: ``"herglotz"``, ``"irregular_herglotz"``, ``"fourier"``, ``"stacked_herglotz"``, and ``"stacked_irregular_herglotz"``
    Additional parameters are forwarded to the constructor of the chosen module.

    Parameters:
        pe (str): Identifier for the type of positional encoding. Must be one of ``"herglotz"``, ``"irregular_herglotz"``, ``"fourier"``, ``"stacked_herglotz"`` or ``"stacked_irregular_herglotz"``.
        **kwargs: Additional keyword arguments to configure the positional encoding module.

    Returns:
        nn.Module: An instance of the specified positional encoding module.

    Raises:
        ValueError: If the specified positional encoding type is not supported.
    """

    if pe not in PE2CLS:
        raise ValueError(f"Invalid positional encoding: {pe}")

    return PE2FN[pe](**kwargs)
