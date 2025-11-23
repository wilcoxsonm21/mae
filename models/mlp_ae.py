"""MLP-based standard autoencoder."""

import torch
import torch.nn as nn
from .base import BaseAutoencoder


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""

    def __init__(self, layer_dims, activation='relu', dropout=0.0):
        """Initialize MLP.

        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            dropout: Dropout probability
        """
        super().__init__()

        # Create activation function
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'sigmoid':
            act_fn = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            # Don't add activation and dropout after the last layer
            if i < len(layer_dims) - 2:
                layers.append(act_fn)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.net(x)


class MLPAE(BaseAutoencoder):
    """MLP-based autoencoder.

    This is a standard autoencoder with MLP encoder and decoder.
    The architecture is symmetric by default but can be customized.
    """

    def __init__(self, input_dim, latent_dim, hidden_dims=None,
                 activation='relu', dropout=0.0):
        """Initialize MLP autoencoder.

        Args:
            input_dim: Dimension of input data
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions for encoder
                        (decoder will be symmetric). Default: [512, 256, 128]
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            dropout: Dropout probability
        """
        super().__init__(input_dim, latent_dim)

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # Encoder: input -> hidden layers -> latent
        encoder_dims = [input_dim] + hidden_dims + [latent_dim]
        self.encoder = MLP(encoder_dims, activation, dropout)

        # Decoder: latent -> hidden layers (reversed) -> input
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        self.decoder = MLP(decoder_dims, activation, dropout)

    def encode(self, x):
        """Encode input to latent representation.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        return self.encoder(x)

    def decode(self, z):
        """Decode latent representation to reconstruction.

        Args:
            z: Latent tensor of shape (batch_size, latent_dim)

        Returns:
            Reconstructed output of shape (batch_size, input_dim)
        """
        return self.decoder(z)
