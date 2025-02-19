import unittest
import torch
from spherical_inr.inr import HerglotzNet


class TestHerglotzNet(unittest.TestCase):
    def setUp(self):
        # Parameters for the HerglotzNet
        self.num_atoms = 16
        self.hidden_layers = 2
        self.hidden_features = 32
        self.out_features = 8
        self.batch_size = 4
        # Dummy input tensor (shape here does not matter as we replace pe with DummyPE)
        self.input_tensor = torch.randn(self.batch_size, 2)

    def test_forward_without_outermost_linear(self):
        """
        Test forward pass with sine activation after the last linear layer.
        """
        model = HerglotzNet(
            num_atoms=self.num_atoms,
            hidden_layers=self.hidden_layers,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            outermost_linear=False,
        )

        output = model(self.input_tensor)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.out_features),
            "Output shape should match (batch_size, out_features).",
        )

    def test_forward_with_outermost_linear(self):
        """
        Test forward pass when the last layer is a plain linear layer (outermost_linear=True).
        """
        model = HerglotzNet(
            num_atoms=self.num_atoms,
            hidden_layers=self.hidden_layers,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            outermost_linear=True,
        )

        output = model(self.input_tensor)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.out_features),
            "Output shape should match (batch_size, out_features) when using outermost linear layer.",
        )

    def test_forward_numeric(self):
        """
        Ensure that the forward pass produces valid numeric values (not NaN or Inf).
        """
        model = HerglotzNet(
            num_atoms=self.num_atoms,
            hidden_layers=self.hidden_layers,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            outermost_linear=False,
        )

        output = model(self.input_tensor)
        self.assertFalse(
            torch.isnan(output).any(), "Output should not contain NaN values."
        )
        self.assertFalse(
            torch.isinf(output).any(), "Output should not contain Inf values."
        )


if __name__ == "__main__":
    unittest.main()
