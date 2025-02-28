import unittest
import torch


from spherical_inr import (
    PositionalEncoding,
    RegularHerglotzPE,
    IregularHerglotzPE,
    FourierPE,
)


class TestPositionalEncodingAbstract(unittest.TestCase):
    def test_cannot_instantiate_abstract_class(self):
        # Attempting to instantiate PositionalEncoding directly should fail.
        with self.assertRaises(TypeError):
            _ = PositionalEncoding(num_atoms=4, input_dim=3)


class TestHerglotzPE(unittest.TestCase):
    def setUp(self):
        self.num_atoms = 5
        self.input_dim = 3  # Valid: >= 2
        self.omega0 = 2.0
        self.bias = True
        self.seed = 42

    def test_invalid_input_dim(self):
        # Input dimension below 2 should raise ValueError.
        with self.assertRaises(ValueError):
            _ = RegularHerglotzPE(
                num_atoms=self.num_atoms,
                input_dim=1,
                bias=self.bias,
                omega0=self.omega0,
                seed=self.seed,
            )

    def test_forward_output_shape_and_type(self):
        pe = RegularHerglotzPE(
            num_atoms=self.num_atoms,
            input_dim=self.input_dim,
            bias=self.bias,
            omega0=self.omega0,
            seed=self.seed,
        )
        batch_size = 4
        # Create an input tensor of shape (batch_size, input_dim)
        x = torch.randn(batch_size, self.input_dim)
        output = pe(x)
        # The forward method returns a real-valued tensor
        self.assertEqual(output.shape, (batch_size, self.num_atoms))
        self.assertTrue(torch.is_floating_point(output))


class TestIregularHerglotzPE(unittest.TestCase):
    def setUp(self):
        self.num_atoms = 4
        self.input_dim = 3  # Valid: >=2
        self.omega0 = 1.5
        self.bias = False
        self.seed = 123

    def test_forward_output_shape_and_type(self):
        pe = IregularHerglotzPE(
            num_atoms=self.num_atoms,
            input_dim=self.input_dim,
            bias=self.bias,
            omega0=self.omega0,
            seed=self.seed,
        )
        batch_size = 6
        # Use nonzero input to avoid division by zero.
        x = torch.randn(batch_size, self.input_dim) + 1.0
        output = pe(x)
        self.assertEqual(output.shape, (batch_size, self.num_atoms))
        self.assertTrue(torch.is_floating_point(output))


class TestFourierPE(unittest.TestCase):
    def setUp(self):
        self.num_atoms = 7
        self.input_dim = 4
        self.bias = True
        self.omega0 = 3.0
        self.seed = 7

    def test_forward_output_shape_and_range(self):
        pe = FourierPE(
            num_atoms=self.num_atoms,
            input_dim=self.input_dim,
            bias=self.bias,
            omega0=self.omega0,
            seed=self.seed,
        )
        batch_size = 8
        x = torch.randn(batch_size, self.input_dim)
        output = pe(x)
        self.assertEqual(output.shape, (batch_size, self.num_atoms))
        # Since FourierPE applies torch.sin, output should be in [-1, 1]
        self.assertTrue(torch.all(output <= 1.0))
        self.assertTrue(torch.all(output >= -1.0))


if __name__ == "__main__":
    unittest.main()
