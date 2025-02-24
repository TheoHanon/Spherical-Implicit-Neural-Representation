import math
import torch
import unittest
from spherical_inr import (
    SphericalToCartesian,
)


class TestSphericalToCartesian(unittest.TestCase):

    def test_invalid_dimension(self):
        # Test that using an unsupported dimension raises a ValueError.
        with self.assertRaises(ValueError):
            _ = SphericalToCartesian(input_dim=4)

    def test_forward_2d_non_unit(self):
        # Test conversion for 2D with r provided in the input.
        # Example: polar coordinates (r=2, theta=pi/2) should give approximately (0, 2)
        transform = SphericalToCartesian(input_dim=2, unit=False)
        # Create input tensor [r, theta]
        x = torch.tensor([2.0, math.pi / 2])
        output = transform.forward(x)
        expected_x = 2.0 * math.cos(math.pi / 2)  # ~0
        expected_y = 2.0 * math.sin(math.pi / 2)  # 2
        expected = torch.tensor([expected_x, expected_y])
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_forward_3d_non_unit(self):
        # Test conversion for 3D with r provided.
        # Example: spherical coordinates (r=2, theta=pi/4, phi=pi/4)
        transform = SphericalToCartesian(input_dim=3, unit=False)
        x = torch.tensor([2.0, math.pi / 4, math.pi / 4])
        # Expected conversion:
        # x = r * sin(theta) * cos(phi)
        # y = r * sin(theta) * sin(phi)
        # z = r * cos(theta)
        r = 2.0
        theta = math.pi / 4
        phi = math.pi / 4
        expected_x = r * math.sin(theta) * math.cos(phi)
        expected_y = r * math.sin(theta) * math.sin(phi)
        expected_z = r * math.cos(theta)
        expected = torch.tensor([expected_x, expected_y, expected_z])
        output = transform.forward(x)
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_forward_2d_unit(self):
        # Test conversion for 2D when unit flag is True.
        # In this case, input contains only the angle.
        transform = SphericalToCartesian(input_dim=2, unit=True)
        # For an angle of pi/2, expected output is (cos(pi/2), sin(pi/2)) ~ (0, 1)
        x = torch.tensor([math.pi / 2])
        output = transform.forward(x)
        expected = torch.tensor([math.cos(math.pi / 2), math.sin(math.pi / 2)])
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_forward_3d_unit(self):
        # Test conversion for 3D when unit flag is True.
        # In this case, input contains only the angles [theta, phi].
        transform = SphericalToCartesian(input_dim=3, unit=True)
        theta = math.pi / 4
        phi = math.pi / 4
        x = torch.tensor([theta, phi])
        # For unit sphere (r=1):
        expected_x = math.sin(theta) * math.cos(phi)
        expected_y = math.sin(theta) * math.sin(phi)
        expected_z = math.cos(theta)
        expected = torch.tensor([expected_x, expected_y, expected_z])
        output = transform.forward(x)
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
