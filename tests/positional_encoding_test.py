import torch
import unittest
from spherical_inr import HerglotzPE, IregularHerglotzPE, FourierPE


class TestHerglotzPE(unittest.TestCase):

    def test_invalid_input_dim(self):
        # HerglotzPE requires input_dim >= 2.
        with self.assertRaises(ValueError):
            _ = HerglotzPE(num_atoms=4, input_dim=1)

    def test_forward_output_shape(self):
        # Test that the forward pass produces an output of shape (batch_size, num_atoms)
        num_atoms = 4
        input_dim = 2
        batch_size = 5
        seed = 123
        pe = HerglotzPE(num_atoms=num_atoms, input_dim=input_dim, seed=seed, omega0=1.0)
        x = torch.randn(batch_size, input_dim)
        output = pe(x)
        self.assertEqual(output.shape, (batch_size, num_atoms))
        self.assertTrue(torch.all(torch.isfinite(output)))


class TestIregularHerglotzPE(unittest.TestCase):

    def test_forward_output_shape(self):
        # IregularHerglotzPE uses a slightly different forward pass.
        num_atoms = 3
        input_dim = 3
        batch_size = 4
        seed = 789
        pe = IregularHerglotzPE(
            num_atoms=num_atoms, input_dim=input_dim, seed=seed, omega0=1.0
        )
        x = torch.randn(batch_size, input_dim)
        output = pe(x)
        self.assertEqual(output.shape, (batch_size, num_atoms))
        self.assertTrue(torch.all(torch.isfinite(output)))


class TestFourierPE(unittest.TestCase):

    def test_forward_output_shape(self):
        # FourierPE applies a linear transformation followed by sine.
        num_atoms = 6
        input_dim = 3
        batch_size = 7
        seed = 101
        omega0 = 2.0
        pe = FourierPE(
            num_atoms=num_atoms, input_dim=input_dim, seed=seed, omega0=omega0
        )
        x = torch.randn(batch_size, input_dim)
        output = pe(x)
        self.assertEqual(output.shape, (batch_size, num_atoms))
        self.assertTrue(torch.all(torch.isfinite(output)))


if __name__ == "__main__":
    unittest.main()
