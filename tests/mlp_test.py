import torch
import torch.nn as nn
import numpy as np
import unittest
from spherical_inr import (
    VanillaMLP,
    SineMLP,
)


class TestVanillaMLP(unittest.TestCase):
    def setUp(self):
        self.input_features = 10
        self.output_features = 5
        self.hidden_features = 20
        self.hidden_layers = 3
        self.batch_size = 4

    def test_output_shape_last_linear_true(self):
        # With last_linear True the final layer is linear.
        mlp = VanillaMLP(
            input_features=self.input_features,
            output_features=self.output_features,
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            last_linear=True,
        )
        x = torch.randn(self.batch_size, self.input_features)
        output = mlp(x)
        self.assertEqual(output.shape, (self.batch_size, self.output_features))

    def test_output_shape_last_linear_false(self):
        # With last_linear False, the final layer output is passed through the activation.
        mlp = VanillaMLP(
            input_features=self.input_features,
            output_features=self.output_features,
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            last_linear=False,
        )
        x = torch.randn(self.batch_size, self.input_features)
        output = mlp(x)
        self.assertEqual(output.shape, (self.batch_size, self.output_features))

    def test_activation_effect_on_last_layer(self):
        # Test that when last_linear is False, the output equals the activation applied to the final linear output.
        # For clarity, we set all weights and biases to zero so that linear outputs are zero.
        mlp = VanillaMLP(
            input_features=self.input_features,
            output_features=self.output_features,
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            last_linear=False,  # Activation will be applied on the last layer.
        )
        # Manually set weights and biases of all layers to zero.
        for layer in mlp.hidden_layers:
            nn.init.constant_(layer.weight, 0.0)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
        x = torch.randn(self.batch_size, self.input_features)
        # For zero linear outputs, activation(x) should be activation(0)
        # VanillaMLP uses nn.ReLU as default activation so ReLU(0) = 0.
        output = mlp(x)
        self.assertTrue(torch.allclose(output, torch.zeros_like(output), atol=1e-6))


class TestSineMLP(unittest.TestCase):
    def setUp(self):
        self.input_features = 8
        self.output_features = 4
        self.hidden_features = 16
        self.hidden_layers = 2
        self.batch_size = 3
        self.omega0 = 1.5

    def test_output_shape(self):
        # SineMLP should return an output with shape (batch_size, output_features)
        mlp = SineMLP(
            input_features=self.input_features,
            output_features=self.output_features,
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            omega0=self.omega0,
        )
        x = torch.randn(self.batch_size, self.input_features)
        output = mlp(x)
        self.assertEqual(output.shape, (self.batch_size, self.output_features))
        self.assertTrue(torch.all(torch.isfinite(output)))

    def test_omega0_buffer(self):
        # Check that the omega0 value is registered correctly as a buffer.
        mlp = SineMLP(
            input_features=self.input_features,
            output_features=self.output_features,
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            omega0=self.omega0,
        )
        self.assertTrue(
            torch.allclose(mlp.omega0, torch.tensor(self.omega0, dtype=torch.float32))
        )

    def test_weight_initialization_bounds(self):
        # For each linear layer, verify that the weights are initialized within [-bound, bound]
        mlp = SineMLP(
            input_features=self.input_features,
            output_features=self.output_features,
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            omega0=self.omega0,
        )
        for layer in mlp.hidden_layers:
            fan_in = layer.weight.size(1)
            bound = np.sqrt(6 / fan_in) / self.omega0
            weights = layer.weight.detach().cpu().numpy()
            self.assertTrue(np.all(weights <= bound + 1e-6))
            self.assertTrue(np.all(weights >= -bound - 1e-6))


if __name__ == "__main__":
    unittest.main()
