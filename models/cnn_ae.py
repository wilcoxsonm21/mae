"""Simple CNN-based autoencoder without skip connections."""

import torch
import torch.nn as nn
from .base import BaseAutoencoder


class CNNAutoencoder(BaseAutoencoder):
    """Simple CNN autoencoder without skip connections.

    This is a standard encoder-decoder architecture with a bottleneck.
    No skip connections are used, forcing all information to flow through
    the latent bottleneck.
    """

    def __init__(self, input_dim, latent_dim, base_channels=64, dropout=0.0,
                 image_size=32, in_channels=1):
        """Initialize CNN autoencoder.

        Args:
            input_dim: Total input dimension (image_size * image_size * in_channels)
            latent_dim: Dimension of bottleneck latent space
            base_channels: Number of channels in first layer
            dropout: Dropout probability
            image_size: Size of square input images (default: 32)
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        """
        super().__init__(input_dim, latent_dim)

        self.image_size = image_size
        self.base_channels = base_channels
        self.in_channels = in_channels

        # Encoder: 32x32 -> 16x16 -> 8x8 -> 4x4
        # Simple convolutions with stride 2 for downsampling
        self.encoder = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),

            # 16x16 -> 8x8
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),

            # 8x8 -> 4x4
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),

            # 4x4 -> 2x2 (bottleneck)
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )

        # Latent projection
        bottleneck_size = 2 * 2 * base_channels * 8  # 2x2 spatial, 512 channels
        self.to_latent = nn.Linear(bottleneck_size, latent_dim)
        self.from_latent = nn.Linear(latent_dim, bottleneck_size)

        # Decoder: 2x2 -> 4x4 -> 8x8 -> 16x16 -> 32x32
        # Transpose convolutions for upsampling
        self.decoder = nn.Sequential(
            # 2x2 -> 4x4
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4,
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2,
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(base_channels * 2, base_channels,
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(base_channels, in_channels,
                             kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x):
        """Encode input to latent representation.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        batch_size = x.shape[0]

        # Reshape to image format
        x = x.view(batch_size, self.in_channels, self.image_size, self.image_size)

        # Encode
        x = self.encoder(x)

        # Flatten and project to latent
        x = x.view(batch_size, -1)
        z = self.to_latent(x)

        return z

    def decode(self, z):
        """Decode latent representation to reconstruction.

        Args:
            z: Latent tensor of shape (batch_size, latent_dim)

        Returns:
            Reconstructed output of shape (batch_size, input_dim)
        """
        batch_size = z.shape[0]

        # Project from latent and reshape
        x = self.from_latent(z)
        x = x.view(batch_size, self.base_channels * 8, 2, 2)

        # Decode
        x = self.decoder(x)

        # Flatten to match input format
        x = x.view(batch_size, -1)

        return x

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
