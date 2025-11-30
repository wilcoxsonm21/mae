"""Transformer-based Autoencoder."""

import torch
import torch.nn as nn
from .base import BaseAutoencoder
from .transformer_blocks import (
    PatchEmbedding,
    PositionalEmbedding,
    TransformerBlock,
    PatchReconstruction,
    initialize_weights
)


class TransformerAE(BaseAutoencoder):
    """Transformer-based Autoencoder with patch-based processing.

    Uses Vision Transformer (ViT) architecture for encoding and decoding images.
    All patches are processed (no masking).

    Args:
        input_dim: Flattened input dimension (e.g., 1024 for 32x32 grayscale)
        latent_dim: Dimension of latent representation
        image_size: Size of square input images (default: 32)
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        patch_size: Size of square patches (default: 4)
        embed_dim: Dimension of patch embeddings (default: 384)
        num_heads: Number of attention heads (default: 6)
        encoder_depth: Number of encoder transformer blocks (default: 6)
        decoder_depth: Number of decoder transformer blocks (default: 3)
        mlp_ratio: Ratio of MLP hidden dimension to embed_dim (default: 4)
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        image_size=32,
        in_channels=1,
        patch_size=4,
        embed_dim=384,
        num_heads=6,
        encoder_depth=6,
        decoder_depth=3,
        mlp_ratio=4,
        dropout=0.0
    ):
        super().__init__(input_dim, latent_dim)

        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth

        # Calculate number of patches
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2

        # Verify input_dim matches image dimensions
        expected_input_dim = in_channels * image_size * image_size
        assert input_dim == expected_input_dim, \
            f"input_dim ({input_dim}) must equal in_channels * image_size^2 ({expected_input_dim})"

        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)

        # Positional embedding
        self.pos_embed = PositionalEmbedding(self.num_patches, embed_dim)

        # Encoder transformer blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(encoder_depth)
        ])

        # Encoder normalization
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Latent projection layers (for downstream tasks)
        self.to_latent = nn.Linear(embed_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, embed_dim)

        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(decoder_depth)
        ])

        # Decoder normalization
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # Patch reconstruction
        self.patch_recon = PatchReconstruction(embed_dim, patch_size, in_channels, image_size)

        # Initialize weights
        self.apply(initialize_weights)

    def encode(self, x):
        """Encode input to latent representation.

        Args:
            x: (batch_size, input_dim) - flattened input images

        Returns:
            z: (batch_size, latent_dim) - latent representation
        """
        batch_size = x.shape[0]

        # Reshape to image format
        # (B, input_dim) -> (B, in_channels, image_size, image_size)
        x = x.view(batch_size, self.in_channels, self.image_size, self.image_size)

        # Convert to patch embeddings
        # (B, C, H, W) -> (B, num_patches, embed_dim)
        patches = self.patch_embed(x)

        # Add positional embeddings
        patches = patches + self.pos_embed()

        # Process through encoder
        for block in self.encoder_blocks:
            patches = block(patches)

        # Normalize
        patches = self.encoder_norm(patches)

        # Store encoded patches for decoder (preserve spatial structure)
        self._encoded_patches = patches

        # Global average pooling over all patches for downstream tasks
        # (B, num_patches, embed_dim) -> (B, embed_dim)
        pooled = patches.mean(dim=1)

        # Project to latent dimension
        # (B, embed_dim) -> (B, latent_dim)
        z = self.to_latent(pooled)

        return z

    def decode(self, z):
        """Decode latent representation to reconstruction.

        Uses the stored encoded patches directly, preserving spatial structure.

        Args:
            z: (batch_size, latent_dim) - latent representation (not used for reconstruction,
                                          only returned for downstream tasks)

        Returns:
            reconstruction: (batch_size, input_dim) - flattened reconstructed images
        """
        # Use the encoded patches directly (preserve spatial information)
        # The latent z is only for downstream tasks, not for reconstruction
        x = self._encoded_patches

        # Decoder processes the spatially-ordered encoded patches
        # No need to add positional embeddings again - they're already in the encoded patches
        for block in self.decoder_blocks:
            x = block(x)

        # Normalize
        x = self.decoder_norm(x)

        # Reconstruct patches to image
        # (B, num_patches, embed_dim) -> (B, input_dim)
        reconstruction = self.patch_recon(x)

        return reconstruction

    def forward(self, x):
        """Forward pass through the autoencoder.

        Args:
            x: (batch_size, input_dim) - flattened input images

        Returns:
            Dictionary with keys:
                - reconstruction: (batch_size, input_dim) - reconstructed images
                - latent: (batch_size, latent_dim) - latent representations
        """
        z = self.encode(x)
        reconstruction = self.decode(z)

        return {
            'reconstruction': reconstruction,
            'latent': z
        }
