import unittest
import torch
from spherical_inr.positional_encoding import HerglotzPE


class TestHerglotzPE(unittest.TestCase):

    def test_invalid_input_domain(self):
        with self.assertRaises(ValueError) as context:
            HerglotzPE(num_atoms=5, input_domain="invalid")
        self.assertIn("Unknown input domain", str(context.exception))

    def test_forward_shape_s2_cart3(self):
        # For "s2" or "r3", the transform sph2_to_cart3 expects input with shape (..., 2)
        for domain, dim in zip(["s2", "r3"], [2, 3]):
            with self.subTest(domain="s2"):
                num_atoms = 4
                module = HerglotzPE(
                    num_atoms=num_atoms, omega0=1.0, seed=42, input_domain=domain
                )
                batch_size = 10
                # Provide input of shape (batch_size, 2) so that sph2_to_cart3 works
                x = torch.randn(batch_size, dim)
                out = module.forward(x)
                # The output should have shape (batch_size, num_atoms)
                self.assertEqual(out.shape, (batch_size, num_atoms))
                self.assertTrue(torch.is_floating_point(out))

    def test_forward_shape_s1_cart2(self):
        # For "s1" or "r2", the transform sph1_to_cart2 expects input with shape (..., 1)
        for domain, dim in zip(["s1", "r2"], [1, 2]):
            with self.subTest(domain=domain):
                num_atoms = 3
                module = HerglotzPE(
                    num_atoms=num_atoms, omega0=2.0, seed=123, input_domain=domain
                )
                batch_size = 8
                # Provide input of shape (batch_size, 1) so that sph1_to_cart2 works
                x = torch.randn(batch_size, dim)
                out = module.forward(x)
                self.assertEqual(out.shape, (batch_size, num_atoms))
                self.assertTrue(torch.is_floating_point(out))

    def test_reproducibility_with_seed(self):
        seed = 2021
        num_atoms = 6
        m1 = HerglotzPE(num_atoms=num_atoms, omega0=0.5, seed=seed, input_domain="s2")
        m2 = HerglotzPE(num_atoms=num_atoms, omega0=0.5, seed=seed, input_domain="s2")
        # Check that the registered buffer "A" is identical.
        torch.testing.assert_close(m1.A, m2.A)
        # Compare all learnable parameters.
        torch.testing.assert_close(m1.w_real, m2.w_real)
        torch.testing.assert_close(m1.w_imag, m2.w_imag)
        torch.testing.assert_close(m1.bias_real, m2.bias_real)
        torch.testing.assert_close(m1.bias_imag, m2.bias_imag)

    def test_generate_herglotz_vector_norm(self):
        module = HerglotzPE(num_atoms=1, input_domain="s2")
        expected = torch.tensor(1 / (2**0.5))
        # Test for 3D vector
        vec3 = module.generate_herglotz_vector(dim=3)
        real_norm = torch.norm(vec3.real)
        imag_norm = torch.norm(vec3.imag)
        torch.testing.assert_close(real_norm, expected, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(imag_norm, expected, rtol=1e-3, atol=1e-3)
        # Check that the real and imaginary parts are orthogonal: dot(a_R, a_I) == 0
        dot3 = torch.dot(vec3.real, vec3.imag)
        torch.testing.assert_close(dot3, torch.tensor(0.0), atol=1e-3, rtol=1e-3)

        # Test for 2D vector
        vec2 = module.generate_herglotz_vector(dim=2)
        real_norm = torch.norm(vec2.real)
        imag_norm = torch.norm(vec2.imag)
        torch.testing.assert_close(real_norm, expected, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(imag_norm, expected, rtol=1e-3, atol=1e-3)
        # Check that the real and imaginary parts are orthogonal: dot(a_R, a_I) == 0
        dot2 = torch.dot(vec2.real, vec2.imag)
        torch.testing.assert_close(dot2, torch.tensor(0.0), atol=1e-3, rtol=1e-3)

    def test_forward_complexity(self):
        # Test that the forward pass applies the intended complex arithmetic.
        num_atoms = 2
        module = HerglotzPE(num_atoms=num_atoms, omega0=1.0, seed=0, input_domain="r3")
        # Manually set parameters to known values.
        module.w_real.data.fill_(0.5)
        module.w_imag.data.fill_(0.0)
        module.bias_real.data.fill_(0.0)
        module.bias_imag.data.fill_(0.0)

        batch_size = 4
        x = torch.randn(batch_size, 3)
        out = module.forward(x)
        self.assertEqual(out.shape, (batch_size, num_atoms))
        self.assertTrue(torch.is_floating_point(out))


if __name__ == "__main__":
    unittest.main()
