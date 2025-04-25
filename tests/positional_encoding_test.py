import unittest
import torch

from spherical_inr.positional_encoding import (
    _PositionalEncoding,
    get_positional_encoding,
    HerglotzPE,
    RegularHerglotzPE,
    IrregularHerglotzPE,
    FourierPE,
    SphericalHarmonicsPE,
    RegularSolidHarmonicsPE,
    IrregularSolidHarmonicsPE,
)


class TestAbstractBase(unittest.TestCase):
    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            _ = _PositionalEncoding(num_atoms=4, input_dim=3)


class TestGetPositionalEncoding(unittest.TestCase):
    def test_valid_names(self):
        for name, cls in [
            ("herglotz", HerglotzPE),
            ("regular_herglotz", RegularHerglotzPE),
            ("irregular_herglotz", IrregularHerglotzPE),
            ("fourier", FourierPE),
            ("spherical_harmonics", SphericalHarmonicsPE),
            ("solid_harmonics", RegularSolidHarmonicsPE),
            ("irregular_solid_harmonics", IrregularSolidHarmonicsPE),
        ]:
            pe = get_positional_encoding(name, num_atoms=4, input_dim=3, L=1, seed=0, omega0=1.0)
            self.assertIsInstance(pe, cls)

    def test_invalid_name(self):
        with self.assertRaises(ValueError):
            get_positional_encoding("no_such_pe")


class TestHerglotzPE(unittest.TestCase):
    def test_forward_num_atoms(self):
        pe = HerglotzPE(num_atoms=5, input_dim=3, bias=False, seed=0)
        x = torch.randn(2, 3)
        y = pe(x)
        self.assertEqual(y.shape, (2, 5))

    def test_forward_L(self):
        L = 2
        num_atoms = (L+1)**2
        pe = HerglotzPE(L=L, input_dim=3, bias=True, seed=1)
        x = torch.randn(3, 3)
        y = pe(x)
        self.assertEqual(y.shape, (3, num_atoms))


class TestRegularHerglotzPE(unittest.TestCase):
    def test_forward_L(self):
        L = 1
        num_atoms = (L+1)**2
        pe = RegularHerglotzPE(L=L, seed=2)
        x = torch.randn(4, 3)
        y = pe(x)
        self.assertEqual(y.shape, (4, num_atoms))


class TestIrregularHerglotzPE(unittest.TestCase):
    def test_forward_L(self):
        L = 1
        num_atoms = (L+1)**2
        pe = IrregularHerglotzPE(L=L, seed=3)
        x = torch.randn(5, 3).abs() + 1e-1  # avoid r=0
        y = pe(x)
        self.assertEqual(y.shape, (5, num_atoms))


class TestFourierPE(unittest.TestCase):
    def test_forward(self):
        pe = FourierPE(num_atoms=6, input_dim=4, bias=True, seed=4, omega0=2.0)
        x = torch.randn(3, 4)
        y = pe(x)
        self.assertEqual(y.shape, (3, 6))
        self.assertTrue(torch.all(y <= 1) and torch.all(y >= -1))


class TestSphericalHarmonicsPE(unittest.TestCase):
    def test_forward_L(self):
        L = 2
        num_atoms = (L+1)**2
        pe = SphericalHarmonicsPE(L=L, seed=5)
        theta = torch.rand(7) * torch.pi
        phi = torch.rand(7) * 2 * torch.pi
        x = torch.stack([theta, phi], dim=-1)
        y = pe(x)
        self.assertEqual(y.shape, (7, num_atoms))


class TestRegularSolidHarmonicsPE(unittest.TestCase):
    def test_forward_L(self):
        L = 1
        num_atoms = (L+1)**2
        pe = RegularSolidHarmonicsPE(L=L, seed=6)
        # (r, θ, φ)
        r = torch.rand(5)
        theta = torch.rand(5) * torch.pi
        phi = torch.rand(5) * 2 * torch.pi
        x = torch.stack([r, theta, phi], dim=-1)
        y = pe(x)
        self.assertEqual(y.shape, (5, num_atoms))


class TestIrregularSolidHarmonicsPE(unittest.TestCase):
    def test_forward_L(self):
        L = 1
        num_atoms = (L+1)**2
        pe = IrregularSolidHarmonicsPE(L=L, seed=7)
        r = torch.rand(5) + 1e-1
        theta = torch.rand(5) * torch.pi
        phi = torch.rand(5) * 2 * torch.pi
        x = torch.stack([r, theta, phi], dim=-1)
        y = pe(x)
        self.assertEqual(y.shape, (5, num_atoms))



if __name__ == "__main__":
    unittest.main()
