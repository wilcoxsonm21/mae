"""UNet-based masked autoencoder for 2D images."""

import torch
import torch.nn as nn
from .base import BaseAutoencoder
from .unet_ae import ConvBlock
from data.utils import create_random_mask, apply_mask


class UNetMAE(BaseAutoencoder):
    """UNet-based masked autoencoder for 2D images.

    This autoencoder learns to reconstruct masked inputs.
    Designed for 32x32 images (like CIFAR-10/MNIST).
    Uses skip connections from encoder to decoder.
    """

    def __init__(self, input_dim, latent_dim, base_channels=64, dropout=0.0,
                 image_size=32, mask_ratio=0.5):
        """Initialize UNet masked autoencoder.

        Args:
            input_dim: Total input dimension (image_size * image_size)
            latent_dim: Dimension of bottleneck latent space
            base_channels: Number of channels in first layer (doubles each downsampling)
            dropout: Dropout probability
            image_size: Size of square input images (default: 32)
            mask_ratio: Fraction of input pixels to mask during training
        """
        super().__init__(input_dim, latent_dim)

        self.image_size = image_size
        self.base_channels = base_channels
        self.mask_ratio = mask_ratio

        # Encoder (downsampling path)
        # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.enc1 = ConvBlock(1, base_channels, dropout)  # 32x32
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2, dropout)  # 16x16
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, dropout)  # 8x8
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck at 4x4
        self.bottleneck_conv = ConvBlock(base_channels * 4, base_channels * 8, dropout)

        # Project to latent space
        bottleneck_size = (image_size // 8) * (image_size // 8) * base_channels * 8  # 4*4*512 for 32x32
        self.to_latent = nn.Linear(bottleneck_size, latent_dim)
        self.from_latent = nn.Linear(latent_dim, bottleneck_size)

        # Decoder (upsampling path)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4, dropout)  # 8x8 (concat with enc3)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2, dropout)  # 16x16 (concat with enc2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels, dropout)  # 32x32 (concat with enc1)

        # Output layer
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def encode(self, x):
        """Encode input to latent representation.

        Args:
            x: Input tensor of shape (batch_size, input_dim) - will be reshaped to (B, 1, H, W)

        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        # Reshape to image format
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, self.image_size, self.image_size)

        # Encoder path with skip connections saved
        e1 = self.enc1(x)
        x = self.pool1(e1)

        e2 = self.enc2(x)
        x = self.pool2(e2)

        e3 = self.enc3(x)
        x = self.pool3(e3)

        # Bottleneck
        x = self.bottleneck_conv(x)

        # Flatten and project to latent space
        x = x.view(batch_size, -1)
        z = self.to_latent(x)

        # Store skip connections for decode
        self._skip_connections = [e1, e2, e3]

        return z

    def decode(self, z):
        """Decode latent representation to reconstruction.

        Args:
            z: Latent tensor of shape (batch_size, latent_dim)

        Returns:
            Reconstructed output of shape (batch_size, input_dim)
        """
        batch_size = z.shape[0]

        # Project from latent space and reshape
        x = self.from_latent(z)
        bottleneck_spatial_size = self.image_size // 8
        x = x.view(batch_size, self.base_channels * 8, bottleneck_spatial_size, bottleneck_spatial_size)

        # Decoder path with skip connections
        e1, e2, e3 = self._skip_connections

        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)  # Skip connection
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)  # Skip connection
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)  # Skip connection
        x = self.dec1(x)

        # Output
        x = self.out_conv(x)

        # Flatten back to (batch_size, input_dim)
        x = x.view(batch_size, -1)

        return x

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
