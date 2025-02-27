import unittest
import torch
import math

from spherical_inr import differentiation as D


class TestGradDivFunctions(unittest.TestCase):
    def test_cartesian_gradient(self):
        # Test with f(x) = x^T x, which gives gradient 2*x.
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        f = (x**2).sum()
        grad = D.cartesian_gradient(f, x, track=False)
        expected = 2 * x
        self.assertTrue(
            torch.allclose(grad, expected), f"Expected {expected}, got {grad}"
        )

    def test_spherical_gradient(self):
        # Test with f(r,θ,φ) = r^2.
        # Expected spherical gradient: [2*r, 0, 0].
        r_val = 2.0
        theta_val = 0.5  # radians
        phi_val = 1.0
        inputs = torch.tensor([r_val, theta_val, phi_val], requires_grad=True)
        f = inputs[0] ** 2  # f = r^2
        grad = D.spherical_gradient(f, inputs, track=False)
        expected = torch.tensor([2 * r_val, 0.0, 0.0])
        self.assertTrue(
            torch.allclose(grad, expected, atol=1e-6),
            f"Expected {expected}, got {grad}",
        )

    def test_cartesian_divergence(self):
        # Test with a 2D vector field F(x, y) = (x, y), for which the divergence is 1 + 1 = 2.
        inputs = torch.tensor([[3.0, 4.0]], requires_grad=True)
        F = inputs  # F = (x, y)

        div = D.cartesian_divergence(F, inputs, track=False)
        expected = torch.tensor([2.0])
        self.assertTrue(
            torch.allclose(div, expected, atol=1e-6), f"Expected {expected}, got {div}"
        )

    def test_spherical_divergence(self):
        # Test with a spherical vector field F = (r, 0, 0), which should yield divergence = 3.
        # For F = (r, 0, 0):
        #   divergence = (1/r²) ∂/∂r (r² * r) = (1/r²) ∂/∂r (r³) = 3.
        r_val = 2.0
        theta_val = 0.5  # radians
        phi_val = 1.0
        inputs = torch.tensor([[r_val, theta_val, phi_val]], requires_grad=True)
        F = torch.stack(
            [
                inputs[..., 0],
                torch.zeros_like(inputs[..., 1]),
                torch.zeros_like(inputs[..., 2]),
            ],
            dim=-1,
        )

        div = D.spherical_divergence(F, inputs, track=False)
        expected = torch.tensor([3.0])
        self.assertTrue(
            torch.allclose(div, expected, atol=1e-6), f"Expected {expected}, got {div}"
        )


class TestLaplacianFunctions(unittest.TestCase):
    def test_cartesian_laplacian(self):
        # Test with f(x)=x^2; second derivative is 2.
        inputs = torch.randn(10, 1, requires_grad=True)
        f = inputs**2
        laplacian = D.cartesian_laplacian(f, inputs, track=False)
        expected = 2 * torch.ones_like(f)
        self.assertTrue(
            torch.allclose(laplacian, expected, atol=1e-5),
            f"Cartesian laplacian mismatch: got {laplacian}, expected {expected}",
        )

    def test_spherical_laplacian_rs2(self):
        # Test with f(r,θ,φ)= 1/r; its Laplacian in R³ is 0 for r > 0.
        n = 10
        r = torch.rand(n) + 1.0  # ensure r > 0
        theta = torch.rand(n) * math.pi
        phi = torch.rand(n) * 2 * math.pi
        inputs = torch.stack([r, theta, phi], dim=-1).requires_grad_(True)
        f = 1 / inputs[..., 0]
        laplacian = D.spherical_laplacian(f, inputs, track=False)
        expected = torch.zeros_like(f)
        self.assertTrue(
            torch.allclose(laplacian, expected, atol=1e-5),
            f"Spherical laplacian mismatch: got {laplacian}, expected {expected}",
        )


if __name__ == "__main__":
    unittest.main()
