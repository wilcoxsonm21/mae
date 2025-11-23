"""Base class for all autoencoder models."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseAutoencoder(ABC, nn.Module):
    """Abstract base class for autoencoder architectures.

    This class defines the interface that all autoencoder models must implement,
    allowing for easy extensibility to more complex architectures in the future.
    """

    def __init__(self, input_dim, latent_dim):
        """Initialize the base autoencoder.

        Args:
            input_dim: Dimension of input data
            latent_dim: Dimension of latent space
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    @abstractmethod
    def encode(self, x):
        """Encode input to latent representation.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        pass

    @abstractmethod
    def decode(self, z):
        """Decode latent representation to reconstruction.

        Args:
            z: Latent tensor of shape (batch_size, latent_dim)

        Returns:
            Reconstructed output of shape (batch_size, input_dim)
        """
        pass

    def forward(self, x):
        """Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Dictionary containing:
                - 'reconstruction': Reconstructed output
                - 'latent': Latent representation
        """
        z = self.encode(x)
        reconstruction = self.decode(z)
        return {
            'reconstruction': reconstruction,
            'latent': z
        }
