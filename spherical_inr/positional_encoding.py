import torch
import torch.nn as nn

from typing import Optional
from abc import ABC, abstractmethod


class PositionalEncoding(ABC, nn.Module):
    """
    Abstract base class for positional encoding modules.

    Parameters:
        num_atoms (int): Number of atoms (encoding vectors) to generate.
        input_dim (int): Dimensionality of the input.
        seed (Optional[int]): Optional random seed for reproducibility.

    Attributes:
        num_atoms (int): Number of atoms used in the encoding.
        input_dim (int): Dimensionality of the input.
        gen (Optional[torch.Generator]): Random number generator for reproducibility (if seed is provided).

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Abstract method to compute the positional encoding of input tensor x.
        extra_repr() -> str:
            Returns a string representation of the module's parameters.
    """

    def __init__(
        self, num_atoms: int, input_dim: int, seed: Optional[int] = None
    ) -> None:
        super(PositionalEncoding, self).__init__()
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


class HerglotzPE(PositionalEncoding):
    """
    Herglotz Positional Encoding.

    This module generates a positional encoding based on the Herglotz approach, constructing complex atoms
    by generating two independent and orthogonal random vectors.

    Parameters:
        num_atoms (int): Number of atoms to generate.
        input_dim (int): Dimensionality of the input (must be at least 2).
        bias (bool, optional): If True, uses learnable bias parameters. Defaults to True.
        seed (Optional[int], optional): Seed for reproducibility.
        omega0 (float, optional): Frequency factor used in the encoding. Defaults to 1.0.

    Attributes:
        A (torch.Tensor): Buffer containing the generated complex atoms.
        omega0 (torch.Tensor): Buffer holding the frequency factor.
        w_real (nn.Parameter): Learnable real part of the weights.
        w_imag (nn.Parameter): Learnable imaginary part of the weights.
        bias_real (nn.Parameter or buffer): Real part of the bias (learnable if bias is True).
        bias_imag (nn.Parameter or buffer): Imaginary part of the bias (learnable if bias is True).

    Methods:
        generate_herglotz_vector() -> torch.Tensor:
            Generates a complex vector (atom) for the encoding.
        forward(x: torch.Tensor) -> torch.Tensor:
            Computes the positional encoding for input tensor x.
        extra_repr() -> str:
            Returns a string representation of the module's parameters.
    """

    def __init__(
        self,
        num_atoms: int,
        input_dim: int,
        bias: bool = True,
        seed: Optional[int] = None,
        omega0: float = 1.0,
    ) -> None:

        if input_dim < 2:
            raise ValueError("The dimension must be at least 2.")

        super(HerglotzPE, self).__init__(
            num_atoms=num_atoms, input_dim=input_dim, seed=seed
        )

        A = torch.stack(
            [self.generate_herglotz_vector() for i in range(self.num_atoms)],
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

    def generate_herglotz_vector(self) -> torch.Tensor:
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

        a_R = torch.randn(self.input_dim, dtype=torch.float32, generator=self.gen)
        a_R /= (2**0.5) * torch.norm(a_R)
        a_I = torch.randn(self.input_dim, dtype=torch.float32, generator=self.gen)
        a_I -= 2 * torch.dot(a_I, a_R) * a_R  # Orthogonalize a_I with respect to a_R
        a_I /= (2**0.5) * torch.norm(a_I)

        return a_R + 1j * a_I

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the positional encoding.

        Depending on the input_domain, the input x is converted from spherical to Cartesian coordinates.
        Then, a linear transformation is applied using the generated complex atoms and learnable parameters.
        Finally, a non-linear transformation involving the complex exponential and cosine is applied.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim) or appropriate spherical shape.

        Returns:
            torch.Tensor: The encoded output tensor.
        """

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


class IregularHerglotzPE(HerglotzPE):
    """
    Irregular Herglotz Positional Encoding.

    An extension of HerglotzPE which create an PE where each atom can be decomposed as band limited Irregular Solid Harmonic Series.

    Parameters:
        Inherits all parameters from HerglotzPE.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Computes the irregular positional encoding for input tensor x, including normalization.
        extra_repr() -> str:
            Returns a string representation of the module's parameters.
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

    def extra_repr(self) -> str:
        repr = super().extra_repr()
        return repr + f", omega0={self.omega0.item()}"


class FourierPE(PositionalEncoding):
    """
    Fourier Positional Encoding.

    This module applies a learnable linear transformation to the input (via the Omega weight matrix) followed
    by a sine activation scaled by a frequency factor omega0.

    Parameters:
        num_atoms (int): Number of output features (atoms).
        input_dim (int): Dimensionality of the input.
        bias (bool, optional): If True, the linear layer includes a bias term. Defaults to True.
        seed (Optional[int], optional): Seed for reproducibility.
        omega0 (float, optional): Frequency factor applied to the activation. Defaults to 1.0.

    Attributes:
        omega0 (torch.Tensor): Buffer holding the frequency factor.
        Omega (nn.Linear): Linear layer mapping input_dim to num_atoms.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Computes the Fourier positional encoding for input tensor x.
        extra_repr() -> str:
            Returns a string representation of the module's parameters.
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
