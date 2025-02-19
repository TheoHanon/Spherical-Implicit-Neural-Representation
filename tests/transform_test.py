import math
import torch
import unittest
from spherical_inr.transforms import (
    sph2_to_cart3,
    sph1_to_cart2,
)


class TestTransforms(unittest.TestCase):

    def test_sph2_to_cart3_valid(self):
        # Test a known conversion:
        # For theta = pi/2 (90°) and phi = 0,
        # x = sin(pi/2)*cos(0) = 1, y = sin(pi/2)*sin(0) = 0, z = cos(pi/2) = 0.
        theta = torch.tensor([math.pi / 2])
        phi = torch.tensor([0.0])
        sph_coords = torch.stack([theta, phi], dim=-1)  # shape: (1, 2)
        cart = sph2_to_cart3(sph_coords)
        expected = torch.tensor([[1.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(cart, expected, atol=1e-6))

        # Test with a batch of values.
        # Use theta = [pi/2, pi/3] and phi = [0, pi/4]
        theta = torch.tensor([math.pi / 2, math.pi / 3])
        phi = torch.tensor([0.0, math.pi / 4])
        sph_coords = torch.stack([theta, phi], dim=-1)  # shape: (2, 2)
        cart = sph2_to_cart3(sph_coords)
        # Expected:
        # For first entry: [sin(pi/2)*cos(0), sin(pi/2)*sin(0), cos(pi/2)] = [1, 0, 0]
        # For second entry: [sin(pi/3)*cos(pi/4), sin(pi/3)*sin(pi/4), cos(pi/3)]
        sin_theta2 = math.sin(math.pi / 3)
        expected2 = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [
                    sin_theta2 * math.cos(math.pi / 4),
                    sin_theta2 * math.sin(math.pi / 4),
                    math.cos(math.pi / 3),
                ],
            ]
        )
        self.assertTrue(torch.allclose(cart, expected2, atol=1e-6))

    def test_sph2_to_cart3_invalid_shape(self):
        # Test that providing a tensor with last dimension != 2 raises ValueError.
        bad_tensor = torch.randn(4, 3)
        with self.assertRaises(ValueError) as context:
            sph2_to_cart3(bad_tensor)
        self.assertIn(
            "The last dimension of spherical_coords must be 2", str(context.exception)
        )

    def test_sph1_to_cart2_valid(self):
        # Test a known conversion:
        # For theta = pi/2,
        # x = cos(pi/2) = 0, y = sin(pi/2) = 1.
        theta = torch.tensor([math.pi / 2])
        sph_coords = theta.unsqueeze(-1)  # shape: (1, 1)
        cart = sph1_to_cart2(sph_coords)
        expected = torch.tensor([[0.0, 1.0]])
        self.assertTrue(torch.allclose(cart, expected, atol=1e-6))

        # Test with a batch of values.
        # For theta = [0, pi/4, pi/2]:
        theta = torch.tensor([0.0, math.pi / 4, math.pi / 2])
        sph_coords = theta.unsqueeze(-1)  # shape: (3, 1)
        cart = sph1_to_cart2(sph_coords)
        expected = torch.tensor(
            [
                [math.cos(0.0), math.sin(0.0)],
                [math.cos(math.pi / 4), math.sin(math.pi / 4)],
                [math.cos(math.pi / 2), math.sin(math.pi / 2)],
            ]
        )
        self.assertTrue(torch.allclose(cart, expected, atol=1e-6))

    def test_sph1_to_cart2_invalid_shape(self):
        # Test that providing a tensor with last dimension != 1 raises ValueError.
        bad_tensor = torch.randn(5, 2)
        with self.assertRaises(ValueError) as context:
            sph1_to_cart2(bad_tensor)
        self.assertIn(
            "The last dimension of spherical_coords must be 1", str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
