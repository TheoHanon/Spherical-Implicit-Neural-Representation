import torch
import torch.nn as nn
import unittest
from spherical_inr import HerglotzNet, SirenNet, PositionalEncoding, MLP, Transform


# Dummy classes for error testing
class DummyPE(PositionalEncoding):
    def __init__(self, num_atoms, input_dim):
        super().__init__(num_atoms=num_atoms, input_dim=input_dim)
        self.num_atoms = num_atoms
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return a dummy tensor with shape (batch, num_atoms)
        batch = x.shape[0]
        return torch.zeros(batch, self.num_atoms)


class DummyMLP(MLP):
    def __init__(self, input_features, output_features):
        super().__init__(input_features=input_features, output_features=output_features)
        self.input_features = input_features
        self.output_features = output_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return a dummy tensor with shape (batch, output_features)
        batch = x.shape[0]
        return torch.zeros(batch, self.output_features)


class DummyTransform(Transform):
    def __init__(self, input_dim):
        super().__init__(input_dim=input_dim)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Identity transformation
        return x


class TestINR(unittest.TestCase):

    def test_inr_forward_with_transform(self):
        # Test INR forward pass using HerglotzNet, which uses a transform.
        # For HerglotzNet, the transform is SphericalToCartesian. For 2D,
        # it expects an input tensor where the first element is the radius and
        # the second element is the angle.
        input_dim = 2
        num_atoms = 10
        hidden_layers = 3
        hidden_features = 16
        output_features = 4
        net = HerglotzNet(
            input_dim=input_dim,
            num_atoms=num_atoms,
            hidden_layers=hidden_layers,
            hidden_features=hidden_features,
            output_features=output_features,
            bias=True,
            pe_omega0=1.0,
            hidden_omega0=1.0,
            seed=42,
            last_linear=True,
        )
        # Construct an input tensor for the transform.
        # For 2D SphericalToCartesian (non-unit mode), each sample must have [r, theta].
        batch = 3
        # Use r=1.0 and theta=0.0, which should map to (1,0) in Cartesian.
        x = torch.tensor([[1.0, 0.0]] * batch)
        output = net(x)
        self.assertEqual(output.shape, (batch, output_features))
        self.assertTrue(torch.all(torch.isfinite(output)))

    def test_inr_forward_without_transform(self):
        # Test INR forward pass using SirenNet, which does not use a transform.
        input_dim = 2
        num_atoms = 8
        hidden_layers = 2
        hidden_features = 16
        output_features = 3
        net = SirenNet(
            input_dim=input_dim,
            num_atoms=num_atoms,
            hidden_layers=hidden_layers,
            hidden_features=hidden_features,
            output_features=output_features,
            bias=True,
            pe_omega0=1.0,
            hidden_omega0=1.0,
            seed=24,
            last_linear=True,
        )
        # Since transform is None, the input to the positional encoding is directly x.
        batch = 4
        x = torch.randn(batch, input_dim)
        output = net(x)
        self.assertEqual(output.shape, (batch, output_features))
        self.assertTrue(torch.all(torch.isfinite(output)))


if __name__ == "__main__":
    unittest.main()
