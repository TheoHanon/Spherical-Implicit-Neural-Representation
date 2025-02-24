import unittest
import torch
import math
from spherical_inr import (
    rsph2_to_cart3,
    sph2_to_cart3,
    rsph1_to_cart2,
    sph1_to_cart2,
)


class TestSphericalToCartesian(unittest.TestCase):

    # Tests for rsph2_to_cart3: expects input with last dim == 3: [r, theta, phi]
    def test_rsph2_to_cart3_valid(self):
        # Single point test
        # Use: r=1, theta=pi/2, phi=0 -> x = 1*sin(pi/2)*cos(0)=1, y = 1*sin(pi/2)*sin(0)=0, z = 1*cos(pi/2)=0
        coords = torch.tensor([1.0, math.pi / 2, 0.0])
        output = rsph2_to_cart3(coords)
        expected = torch.tensor([1.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))

        # Batch test: two points
        coords_batch = torch.tensor(
            [[1.0, math.pi / 2, 0.0], [2.0, math.pi / 2, math.pi]]
        )
        output_batch = rsph2_to_cart3(coords_batch)
        # For the second point: r=2, theta=pi/2, phi=pi -> x=2*sin(pi/2)*cos(pi)= -2, y=2*sin(pi/2)*sin(pi)≈0, z=2*cos(pi/2)=0
        expected_batch = torch.tensor([[1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(output_batch, expected_batch, atol=1e-5))

    def test_rsph2_to_cart3_invalid(self):
        # Input with last dim not equal to 3 should raise a ValueError.
        coords = torch.tensor([1.0, math.pi / 2])  # Only 2 elements.
        with self.assertRaises(ValueError):
            _ = rsph2_to_cart3(coords)

    # Tests for sph2_to_cart3: expects input with last dim == 2: [theta, phi] (unit sphere assumed)
    def test_sph2_to_cart3_valid(self):
        # Single point: theta=pi/2, phi=0 -> x = sin(pi/2)*cos(0)=1, y = sin(pi/2)*sin(0)=0, z = cos(pi/2)=0
        coords = torch.tensor([math.pi / 2, 0.0])
        output = sph2_to_cart3(coords)
        expected = torch.tensor([1.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))

        # Batch test
        coords_batch = torch.tensor([[math.pi / 2, 0.0], [math.pi / 2, math.pi]])
        output_batch = sph2_to_cart3(coords_batch)
        expected_batch = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        self.assertTrue(torch.allclose(output_batch, expected_batch, atol=1e-5))

    def test_sph2_to_cart3_invalid(self):
        # Input with last dim not equal to 2 should raise a ValueError.
        coords = torch.tensor([0.0, math.pi / 2, 0.0])  # 3 elements instead of 2.
        with self.assertRaises(ValueError):
            _ = sph2_to_cart3(coords)

    # Tests for rsph1_to_cart2: expects input with last dim == 2: [r, theta]
    def test_rsph1_to_cart2_valid(self):
        # Single point: r=2, theta=0 -> x = 2*cos(0)=2, y = 2*sin(0)=0.
        coords = torch.tensor([2.0, 0.0])
        output = rsph1_to_cart2(coords)
        expected = torch.tensor([2.0, 0.0])
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))

        # Batch test
        coords_batch = torch.tensor([[2.0, 0.0], [3.0, math.pi / 2]])
        output_batch = rsph1_to_cart2(coords_batch)
        expected_batch = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        self.assertTrue(torch.allclose(output_batch, expected_batch, atol=1e-5))

    def test_rsph1_to_cart2_invalid(self):
        # Input with last dim not equal to 2 should raise a ValueError.
        coords = torch.tensor([1.0])  # Only one element.
        with self.assertRaises(ValueError):
            _ = rsph1_to_cart2(coords)

    # Tests for sph1_to_cart2: expects input with last dim == 1: [theta]
    def test_sph1_to_cart2_valid(self):
        # Single point: theta=pi/2 -> x = cos(pi/2)=0, y = sin(pi/2)=1.
        coords = torch.tensor([math.pi / 2])
        output = sph1_to_cart2(coords)
        expected = torch.tensor([0.0, 1.0])
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))

        # Batch test: two angles.
        coords_batch = torch.tensor([[0.0], [math.pi / 2]])
        output_batch = sph1_to_cart2(coords_batch)
        expected_batch = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]]  # cos(0)=1, sin(0)=0  # cos(pi/2)=0, sin(pi/2)=1
        )
        self.assertTrue(torch.allclose(output_batch, expected_batch, atol=1e-5))

    def test_sph1_to_cart2_invalid(self):
        # Input with last dim not equal to 1 should raise a ValueError.
        coords = torch.tensor([0.0, math.pi / 2])  # Two elements instead of one.
        with self.assertRaises(ValueError):
            _ = sph1_to_cart2(coords)


if __name__ == "__main__":
    unittest.main()
