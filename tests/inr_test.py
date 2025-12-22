import unittest
import torch
from spherical_inr import (
    HerglotzNet,
    SirenNet,
    SphericalSirenNet,
)


class TestHerglotzNet(unittest.TestCase):
    def setUp(self):
        self.L = 3
        self.mlp_sizes = [16, 16]
        self.output_dim = 4

    def test_forward_valid(self):
        net = HerglotzNet(
            num_atoms=self.L * (self.L + 1) // 2,
            mlp_sizes=self.mlp_sizes,
            output_dim=self.output_dim,
        )
        # spherical coords (θ,φ)
        theta = torch.rand(10) * torch.pi
        phi = torch.rand(10) * 2 * torch.pi
        x = torch.stack([theta, phi], dim=-1)
        y = net(x)
        self.assertEqual(y.shape, (10, self.output_dim))

    def test_forward_wrong_input_dim(self):
        net = HerglotzNet(self.L, self.mlp_sizes, self.output_dim)
        x = torch.randn(10, 3)
        with self.assertRaises(ValueError):
            net(x)


class TestSirenNet(unittest.TestCase):
    def setUp(self):
        self.num_atoms = 10
        self.mlp_sizes = [20, 20]
        self.output_dim = 2

    def test_forward_valid(self):
        net = SirenNet(
            num_atoms=self.num_atoms,
            mlp_sizes=self.mlp_sizes,
            output_dim=self.output_dim,
        )
        x = torch.randn(8, 2)
        y = net(x)
        self.assertEqual(y.shape, (8, self.output_dim))

    def test_forward_wrong_input_dim(self):
        net = SirenNet(
            num_atoms=self.num_atoms,
            mlp_sizes=self.mlp_sizes,
            output_dim=self.output_dim,
        )
        x = torch.randn(8, 3)
        with self.assertRaises(RuntimeError):
            net(x)


class TestSphericalSirenNet(unittest.TestCase):
    def setUp(self):
        self.L = 2
        self.mlp_sizes = [10]
        self.output_dim = 3

    def test_forward_valid(self):
        net = SphericalSirenNet(
            num_atoms=self.L * (self.L + 1) // 2,
            mlp_sizes=self.mlp_sizes,
            output_dim=self.output_dim,
        )

        theta = torch.rand(9) * torch.pi
        phi = torch.rand(9) * 2 * torch.pi
        x = torch.stack([theta, phi], dim=-1)
        y = net(x)
        self.assertEqual(y.shape, (9, self.output_dim))

    def test_forward_wrong_input_dim(self):
        net = SphericalSirenNet(self.L, self.mlp_sizes, self.output_dim)
        x = torch.randn(9, 3)
        with self.assertRaises(ValueError):
            net(x)


if __name__ == "__main__":
    unittest.main()
