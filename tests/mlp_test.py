import unittest
import torch
import torch.nn as nn
import numpy as np

from spherical_inr import ReLUMLP, SineMLP


class TestReLUMLP(unittest.TestCase):
    def setUp(self):
        self.input_features = 3
        self.output_features = 2
        self.hidden_sizes = [10, 10]
        self.bias = True
        self.batch_size = 5

        # Create an instance of SineMLP
        self.mlp = ReLUMLP(
            input_features=self.input_features,
            output_features=self.output_features,
            hidden_sizes=self.hidden_sizes,
            bias=self.bias,
        )

    def test_forward_output_shape(self):
        # Create a dummy input tensor of shape (batch_size, input_features)
        x = torch.randn(self.batch_size, self.input_features)
        output = self.mlp(x)
        # Check that the output shape is (batch_size, output_features)
        self.assertEqual(output.shape, (self.batch_size, self.output_features))


class TestSineMLP(unittest.TestCase):

    def setUp(self):
        self.input_features = 3
        self.output_features = 2
        self.hidden_sizes = [10, 10]
        self.bias = True
        self.omega0 = 1.0
        self.batch_size = 5

        # Create an instance of SineMLP
        self.mlp = SineMLP(
            input_features=self.input_features,
            output_features=self.output_features,
            hidden_sizes=self.hidden_sizes,
            bias=self.bias,
            omega0=self.omega0,
        )

    def test_forward_output_shape(self):
        # Create a dummy input tensor of shape (batch_size, input_features)
        x = torch.randn(self.batch_size, self.input_features)
        output = self.mlp(x)
        # Check that the output shape is (batch_size, output_features)
        self.assertEqual(output.shape, (self.batch_size, self.output_features))

    def test_forward_activation_function(self):
        # Since SineMLP applies torch.sin(omega0 * layer(x)) for all layers except the last,
        # we can perform a basic sanity check: the sine function returns values in [-1, 1].
        # We run the forward pass on a fixed input and ensure that intermediate outputs (if extracted)
        # are within the expected range.
        x1 = torch.randn(self.batch_size, self.input_features)
        x2 = x1.clone()
        # Forward pass through all but the last layer manually.
        for layer in self.mlp.hidden_layers[:-1]:
            x1 = torch.sin(self.omega0 * layer(x1))
            self.assertTrue(torch.all(x1 <= 1.0))
            self.assertTrue(torch.all(x1 >= -1.0))
        # Complete the forward pass.
        output = self.mlp(x2)
        # We only check the shape of the final output.
        self.assertEqual(output.shape, (self.batch_size, self.output_features))


if __name__ == "__main__":
    unittest.main()
