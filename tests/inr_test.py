import unittest
import torch
from spherical_inr import (
    INR,
    HerglotzNet,
    RegularHerglotzNet,
    IrregularHerglotzNet,
    SirenNet,
    HerglotzSirenNet,
    SphericalSirenNet,
    IrregularSolidSirenNet,
    RegularSolidSirenNet,
)


class TestINR(unittest.TestCase):
    def test_forward_fourier_and_shape(self):
        net = INR(
            num_atoms=6,
            mlp_sizes=[16, 16],
            output_dim=3,
            input_dim=2,
            pe="fourier",
            pe_kwargs={"omega0": 0.5},
            mlp_kwargs={"bias": True},
        )
        x = torch.randn(10, 2)
        y = net(x)
        self.assertEqual(y.shape, (10, 3))

    def test_invalid_pe_name(self):
        with self.assertRaises(ValueError):
            INR(
                num_atoms=4,
                mlp_sizes=[8],
                output_dim=1,
                input_dim=2,
                pe="not_a_pe"
            )


class TestHerglotzNet(unittest.TestCase):
    def setUp(self):
        self.L = 3
        self.mlp_sizes = [16, 16]
        self.output_dim = 4

    def test_forward_valid(self):
        net = HerglotzNet(
            L=self.L,
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


class TestRegularHerglotzNet(unittest.TestCase):
    def setUp(self):
        self.L = 2
        self.mlp_sizes = [12, 12]
        self.output_dim = 3

    def test_forward_valid(self):
        net = RegularHerglotzNet(
            L=self.L,
            mlp_sizes=self.mlp_sizes,
            output_dim=self.output_dim,
        )
        # spherical (r,θ,φ)
        r = torch.rand(5)
        theta = torch.rand(5) * torch.pi
        phi = torch.rand(5) * 2 * torch.pi
        x = torch.stack([r, theta, phi], dim=-1)
        y = net(x)
        self.assertEqual(y.shape, (5, self.output_dim))

    def test_forward_wrong_input_dim(self):
        net = RegularHerglotzNet(self.L, self.mlp_sizes, self.output_dim)
        x = torch.randn(5, 2)
        with self.assertRaises(ValueError):
            net(x)


class TestIrregularHerglotzNet(unittest.TestCase):
    def setUp(self):
        self.L = 2
        self.mlp_sizes = [12, 12]
        self.output_dim = 3

    def test_forward_valid(self):
        net = IrregularHerglotzNet(
            L=self.L,
            mlp_sizes=self.mlp_sizes,
            output_dim=self.output_dim,
        )
        r = torch.rand(5)
        theta = torch.rand(5) * torch.pi
        phi = torch.rand(5) * 2 * torch.pi
        x = torch.stack([r, theta, phi], dim=-1)
        y = net(x)
        self.assertEqual(y.shape, (5, self.output_dim))

    def test_forward_wrong_input_dim(self):
        net = IrregularHerglotzNet(self.L, self.mlp_sizes, self.output_dim)
        x = torch.randn(5, 2)
        with self.assertRaises(ValueError):
            net(x)


class TestSirenNet(unittest.TestCase):
    def setUp(self):
        self.num_atoms = 10
        self.mlp_sizes = [20, 20]
        self.output_dim = 2
        self.input_dim = 5

    def test_forward_valid(self):
        net = SirenNet(
            num_atoms=self.num_atoms,
            mlp_sizes=self.mlp_sizes,
            output_dim=self.output_dim,
            input_dim=self.input_dim,
            pe_kwargs={"omega0": 1.0},
            mlp_kwargs={"omega0": 1.0},
        )
        x = torch.randn(8, self.input_dim)
        y = net(x)
        self.assertEqual(y.shape, (8, self.output_dim))

    def test_forward_wrong_input_dim(self):
        net = SirenNet(
            num_atoms=self.num_atoms,
            mlp_sizes=self.mlp_sizes,
            output_dim=self.output_dim,
            input_dim=self.input_dim,
        )
        x = torch.randn(8, 3)
        with self.assertRaises(RuntimeError):
            net(x)


class TestHerglotzSirenNet(unittest.TestCase):
    def setUp(self):
        self.num_atoms = 6
        self.mlp_sizes = [16, 16]
        self.output_dim = 4
        self.input_dim = 3

    def test_forward_valid(self):
        net = HerglotzSirenNet(
            num_atoms=self.num_atoms,
            mlp_sizes=self.mlp_sizes,
            output_dim=self.output_dim,
            input_dim=self.input_dim,
            pe_kwargs={"seed": 0},
        )
        x = torch.randn(7, self.input_dim)
        y = net(x)
        self.assertEqual(y.shape, (7, self.output_dim))

    def test_forward_wrong_input_dim(self):
        net = HerglotzSirenNet(
            num_atoms=self.num_atoms,
            mlp_sizes=self.mlp_sizes,
            output_dim=self.output_dim,
            input_dim=self.input_dim,
        )
        x = torch.randn(7, 2)
        with self.assertRaises(RuntimeError):
            net(x)


class TestSphericalSirenNet(unittest.TestCase):
    def setUp(self):
        self.L = 2
        self.mlp_sizes = [10]
        self.output_dim = 3

    def test_forward_valid(self):
        net = SphericalSirenNet(
            L=self.L,
            mlp_sizes=self.mlp_sizes,
            output_dim=self.output_dim,
            seed=0,
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


class TestIrregularSolidSirenNet(unittest.TestCase):
    def setUp(self):
        self.L = 3
        self.mlp_sizes = [8]
        self.output_dim = 2

    def test_forward_valid(self):
        net = IrregularSolidSirenNet(
            L=self.L,
            mlp_sizes=self.mlp_sizes,
            output_dim=self.output_dim,
            seed=0,
        )
        r = torch.rand(6)
        theta = torch.rand(6) * torch.pi
        phi = torch.rand(6) * 2 * torch.pi
        x = torch.stack([r, theta, phi], dim=-1)
        y = net(x)
        self.assertEqual(y.shape, (6, self.output_dim))

    def test_forward_wrong_input_dim(self):
        net = IrregularSolidSirenNet(self.L, self.mlp_sizes, self.output_dim)
        x = torch.randn(6, 2)
        with self.assertRaises(ValueError):
            net(x)


class TestRegularSolidSirenNet(unittest.TestCase):
    def setUp(self):
        self.L = 3
        self.mlp_sizes = [8]
        self.output_dim = 2

    def test_forward_valid(self):
        net = RegularSolidSirenNet(
            L=self.L,
            mlp_sizes=self.mlp_sizes,
            output_dim=self.output_dim,
            seed=0,
        )
        r = torch.rand(4)
        theta = torch.rand(4) * torch.pi
        phi = torch.rand(4) * 2 * torch.pi
        x = torch.stack([r, theta, phi], dim=-1)
        y = net(x)
        self.assertEqual(y.shape, (4, self.output_dim))

    def test_forward_wrong_input_dim(self):
        net = RegularSolidSirenNet(self.L, self.mlp_sizes, self.output_dim)
        x = torch.randn(4, 1)
        with self.assertRaises(ValueError):
            net(x)


if __name__ == "__main__":
    unittest.main()
