import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from spherical_inr import HerglotzNet, SolidHerlotzNet, SirenNet


class TestHerglotzNet(unittest.TestCase):
    def setUp(self):
        self.output_dim = 4
        self.num_atoms = 8
        self.mlp_sizes = [16, 16]

    def test_invalid_input_dim(self):
        # Input dim must be 1 or 2, so 3 should raise an error.
        with self.assertRaises(ValueError):
            HerglotzNet(3, self.output_dim, self.num_atoms, self.mlp_sizes)

    def test_forward_with_input_dim_1(self):
        net = HerglotzNet(1, self.output_dim, self.num_atoms, self.mlp_sizes)
        # Create a dummy input tensor with shape (batch, input_dim)
        x = torch.randn(10, 1)
        y = net(x)
        # Expect output shape to be (batch, output_dim)
        self.assertEqual(y.shape, (10, self.output_dim))

    def test_forward_with_input_dim_2(self):
        net = HerglotzNet(2, self.output_dim, self.num_atoms, self.mlp_sizes)
        x = torch.randn(10, 2)
        y = net(x)
        self.assertEqual(y.shape, (10, self.output_dim))


class TestSolidHerlotzNet(unittest.TestCase):
    def setUp(self):
        self.output_dim = 3
        self.num_atoms = 6
        self.mlp_sizes = [12, 12]

    def test_invalid_input_dim(self):
        # Input dim must be 2 or 3, so 1 should raise an error.
        with self.assertRaises(ValueError):
            SolidHerlotzNet(1, self.output_dim, self.num_atoms, self.mlp_sizes)

    def test_invalid_type_parameter(self):
        # The 'type' parameter must be either "R" or "I".
        with self.assertRaises(ValueError):
            SolidHerlotzNet(
                2, self.output_dim, self.num_atoms, self.mlp_sizes, type="X"
            )

    def test_forward_with_input_dim_2(self):
        net = SolidHerlotzNet(
            2, self.output_dim, self.num_atoms, self.mlp_sizes, type="R"
        )
        x = torch.randn(10, 2)
        y = net(x)
        self.assertEqual(y.shape, (10, self.output_dim))

    def test_forward_with_input_dim_3_and_type_I(self):
        net = SolidHerlotzNet(
            3, self.output_dim, self.num_atoms, self.mlp_sizes, type="I"
        )
        x = torch.randn(10, 3)
        y = net(x)
        self.assertEqual(y.shape, (10, self.output_dim))


class TestSirenNet(unittest.TestCase):
    def setUp(self):
        self.input_dim = 5
        self.output_dim = 2
        self.num_atoms = 10
        self.mlp_sizes = [20, 20]

    def test_forward(self):
        net = SirenNet(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_atoms=self.num_atoms,
            mlp_sizes=self.mlp_sizes,
            first_omega0=1.0,
            hidden_omega0=1.0,
        )
        # Create a dummy input tensor with shape (batch, input_dim)
        x = torch.randn(8, self.input_dim)
        y = net(x)
        self.assertEqual(y.shape, (8, self.output_dim))


if __name__ == "__main__":
    unittest.main()
