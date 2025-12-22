import unittest
import torch

from spherical_inr.positional_encoding import (
    HerglotzPE,
    FourierPE,
    SphericalHarmonicsPE,
)


class TestHerglotzPE(unittest.TestCase):
    def setUp(self) -> None:

        self.num_atoms = 5
        self.L_init = 15

    def test_forward_num_atoms(self):
        pe = HerglotzPE(num_atoms=self.num_atoms, L_init=self.L_init)
        x = torch.randn(2, 3)
        y = pe(x)
        self.assertEqual(y.shape, (2, 5))

    def test_forward_rot_num_atoms(self):
        pe = HerglotzPE(num_atoms=self.num_atoms, L_init=self.L_init, rot=True)
        x = torch.randn(2, 3)
        y = pe(x)
        self.assertEqual(y.shape, (2, 5))

    def test_forward_L(self):
        L = 2
        pe = HerglotzPE(num_atoms=self.num_atoms, L_init=self.L_init)
        x = torch.randn(3, 3)
        y = pe(x)
        self.assertEqual(y.shape, (3, self.num_atoms))


class TestFourierPE(unittest.TestCase):
    def test_forward(self):
        pe = FourierPE(num_atoms=6, input_dim=4, bias=True, omega0=2.0)
        x = torch.randn(3, 4)
        y = pe(x)
        self.assertEqual(y.shape, (3, 6))
        self.assertTrue(torch.all(y <= 1) and torch.all(y >= -1))


class TestSphericalHarmonicsPE(unittest.TestCase):
    def test_forward_L(self):
        L = 2
        num_atoms = (L + 1) ** 2
        pe = SphericalHarmonicsPE(num_atoms=num_atoms)
        theta = torch.rand(7) * torch.pi
        phi = torch.rand(7) * 2 * torch.pi
        x = torch.stack([theta, phi], dim=-1)
        y = pe(x)
        self.assertEqual(y.shape, (7, num_atoms))


if __name__ == "__main__":
    unittest.main()
