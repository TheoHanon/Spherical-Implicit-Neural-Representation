import torch
import torch.nn as nn
from .transforms import sph2_to_cart3, sph1_to_cart2
from typing import Optional, Union


class HerglotzPE(nn.Module):
    """
    HerglotzPE Positional Encoding Module

    This module implements a positional encoding based on Herglotz functions. It is designed to
    work with inputs in different domains: either spherical (s1 or s2) or Cartesian (r2 or r3).
    It generates a set of complex atoms and applies a linear transformation in a vectorized manner
    using complex arithmetic.

    Parameters:
        num_atoms (int): The number of atoms (or frequency components) used in the encoding.
        omega0 (float, optional): A scaling factor applied to the transformation. Default is 1.0.
        seed (Optional[int], optional): Random seed for reproducibility of the atom generation. Default is None.
        input_domain (str, optional): The domain of the input. Options are "s1", "s2", "r2", or "r3".
                                      "s2" and "r3" are treated as 3-dimensional, while "s1" and "r2" are 2-dimensional.
                                      Default is "s2".

    Attributes:
        input_domain (str): The normalized input domain.
        input_dim (int): The dimensionality of the input (2 or 3) derived from the input_domain.
        num_atoms (int): Number of atoms used in the encoding.
        A (torch.Tensor): Buffer containing generated complex atoms (shape: [num_atoms, input_dim]),
                          where each atom is a complex vector.
        omega0 (torch.Tensor): A scalar tensor holding the omega0 value.
        w_real (nn.Parameter): Learnable parameter for the real part of the complex weights (shape: [num_atoms]).
        w_imag (nn.Parameter): Learnable parameter for the imaginary part of the complex weights (shape: [num_atoms]).
        bias_real (nn.Parameter): Learnable parameter for the real part of the bias (shape: [num_atoms]).
        bias_imag (nn.Parameter): Learnable parameter for the imaginary part of the bias (shape: [num_atoms]).
    """

    def __init__(
        self,
        num_atoms: int,
        omega0: float = 1.0,
        seed: Optional[int] = None,
        input_domain: str = "s2",
    ) -> None:

        super(HerglotzPE, self).__init__()

        self.input_domain = input_domain.lower()

        if self.input_domain in ["s2", "r3"]:
            self.input_dim = 3
        elif self.input_domain in ["s1", "r2"]:
            self.input_dim = 2
        else:
            raise ValueError(f"Unknown input domain: {input_domain}")

        self.num_atoms = num_atoms

        gen: Optional[torch.Generator] = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)

        A = torch.stack(
            [
                self.generate_herglotz_vector(dim=self.input_dim, generator=gen)
                for i in range(self.num_atoms)
            ],
            dim=0,
        )
        self.register_buffer("A", A)
        self.register_buffer("omega0", torch.tensor(omega0, dtype=torch.float32))

        bound = 1 / self.input_dim
        self.w_real = nn.Parameter(
            torch.empty(self.num_atoms, dtype=torch.float32).uniform_(
                -bound, bound, generator=gen
            )
        )
        self.w_imag = nn.Parameter(
            torch.empty(self.num_atoms, dtype=torch.float32).uniform_(
                -bound, bound, generator=gen
            )
        )
        self.bias_real = nn.Parameter(torch.zeros(self.num_atoms, dtype=torch.float32))
        self.bias_imag = nn.Parameter(torch.zeros(self.num_atoms, dtype=torch.float32))

    def generate_herglotz_vector(
        self, dim: int, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Generates a complex vector (atom) for the Herglotz encoding.

        The vector is constructed by generating two independent random vectors,
        normalizing them, and ensuring the imaginary part is orthogonal to the real part.

        Parameters:
            dim (int): The dimension of the vector (2 or 3).
            generator (Optional[torch.Generator]): A random number generator for reproducibility. Default is None.

        Returns:
            torch.Tensor: A complex tensor representing the atom (dtype=torch.complex64).
        """
        if dim not in (2, 3):
            raise ValueError(f"Unsupported dimension: {dim}")

        a_R = torch.randn(dim, dtype=torch.float32, generator=generator)
        a_R /= (2**0.5) * torch.norm(a_R)
        a_I = torch.randn(dim, dtype=torch.float32, generator=generator)
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
        if self.input_domain == "s2":
            x = sph2_to_cart3(x)
        elif self.input_domain == "s1":
            x = sph1_to_cart2(x)

        x = x.to(self.A.dtype)
        x = torch.matmul(x, self.A.t())

        x = self.omega0 * (
            (self.w_real + 1j * self.w_imag) * x
            + (self.bias_real + 1j * self.bias_imag)
        )

        return torch.exp(-x.imag) * torch.cos(x.real)

    def extra_repr(self) -> str:
        return (
            f"num_atoms={self.num_atoms}, "
            f"omega0={self.omega0.item():.3f}, "
            f"input_domain='{self.input_domain}', "
            f"input_dim={self.input_dim}"
        )
