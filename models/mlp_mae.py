"""MLP-based masked autoencoder."""

import torch
import torch.nn as nn
from .base import BaseAutoencoder
from .mlp_ae import MLP
from data.utils import create_random_mask, apply_mask


class MLPMAE(BaseAutoencoder):
    """MLP-based masked autoencoder.

    This autoencoder learns to reconstruct masked inputs, similar to
    the Masked Autoencoder (MAE) approach. The model is trained to
    predict the original values at masked positions.
    """

    def __init__(self, input_dim, latent_dim, hidden_dims=None,
                 activation='relu', dropout=0.0, mask_ratio=0.5):
        """Initialize MLP masked autoencoder.

        Args:
            input_dim: Dimension of input data
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions for encoder
                        (decoder will be symmetric). Default: [512, 256, 128]
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            dropout: Dropout probability
            mask_ratio: Fraction of input features to mask during training
        """
        super().__init__(input_dim, latent_dim)

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.mask_ratio = mask_ratio

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

    def forward(self, x, mask=None):
        """Forward pass through the masked autoencoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            mask: Optional binary mask of shape (batch_size, input_dim).
                  If None, a random mask will be generated.
                  1 indicates masked positions.

        Returns:
            Dictionary containing:
                - 'reconstruction': Reconstructed output
                - 'latent': Latent representation
                - 'mask': The mask used (for loss computation)
                - 'masked_input': The masked input
        """
        # Create mask if not provided
        if mask is None:
            mask = create_random_mask(
                x.shape[0],
                x.shape[1],
                self.mask_ratio,
                device=x.device
            )

        # Apply mask to input
        masked_input = apply_mask(x, mask, mask_value=0.0)

        # Encode and decode
        z = self.encode(masked_input)
        reconstruction = self.decode(z)

        return {
            'reconstruction': reconstruction,
            'latent': z,
            'mask': mask,
            'masked_input': masked_input
        }
