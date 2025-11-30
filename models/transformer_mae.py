"""Transformer-based Masked Autoencoder."""

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
from data.utils import create_patch_mask


class TransformerMAE(BaseAutoencoder):
    """Transformer-based Masked Autoencoder with patch-based processing.

    Uses Masked Autoencoder (MAE) architecture:
    - Encoder only processes visible (unmasked) patches for efficiency
    - Decoder processes all patches with learned mask tokens at masked positions
    - Loss is computed only on masked patches

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
        mask_ratio: Ratio of patches to mask (default: 0.75)
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
        dropout=0.0,
        mask_ratio=0.75
    ):
        super().__init__(input_dim, latent_dim)

        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.mask_ratio = mask_ratio

        # Calculate number of patches
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.patch_dim = patch_size * patch_size * in_channels

        # Verify input_dim matches image dimensions
        expected_input_dim = in_channels * image_size * image_size
        assert input_dim == expected_input_dim, \
            f"input_dim ({input_dim}) must equal in_channels * image_size^2 ({expected_input_dim})"

        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)

        # Positional embedding
        self.pos_embed = PositionalEmbedding(self.num_patches, embed_dim)

        # Learned mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

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

        # Store encoder output for decoding
        self._encoder_output = None
        self._patch_mask = None

    def encode_with_mask(self, x, patch_mask):
        """Encode input with masking - only processes visible patches.

        Args:
            x: (batch_size, input_dim) - flattened input images
            patch_mask: (batch_size, num_patches) - binary mask, 1=masked, 0=visible

        Returns:
            encoder_output: (batch_size, num_visible_patches, embed_dim) - encoded visible patches
            patch_mask: (batch_size, num_patches) - binary mask at patch level
        """
        batch_size = x.shape[0]

        # Reshape to image format
        # (B, input_dim) -> (B, in_channels, image_size, image_size)
        x = x.view(batch_size, self.in_channels, self.image_size, self.image_size)

        # Convert to patch embeddings (WITHOUT positional embeddings yet)
        # (B, C, H, W) -> (B, num_patches, embed_dim)
        patches = self.patch_embed(x)

        # Separate visible and masked patches
        # We'll process each sample in the batch separately to handle variable lengths
        visible_patches_list = []
        visible_indices_list = []

        for i in range(batch_size):
            # Get mask for this sample
            sample_mask = patch_mask[i]  # (num_patches,)
            visible_indices = (sample_mask == 0).nonzero(as_tuple=True)[0]  # Indices of visible patches
            visible_indices_list.append(visible_indices)

            # Extract visible patches (WITHOUT adding positional embeddings here)
            sample_patches = patches[i, visible_indices, :]  # (num_visible, embed_dim)

            # Add positional embeddings for visible patches based on their spatial position
            sample_pos = self.pos_embed().squeeze(0)[visible_indices, :]  # (num_visible, embed_dim)
            sample_patches = sample_patches + sample_pos

            visible_patches_list.append(sample_patches)

        # Stack visible patches (this will have variable length across batch)
        # For simplicity, we'll pad to the max length in the batch
        max_visible = max(vp.shape[0] for vp in visible_patches_list)

        # Create padded tensor
        visible_patches = torch.zeros(batch_size, max_visible, self.embed_dim, device=x.device)
        attention_mask = torch.zeros(batch_size, max_visible, device=x.device)

        for i, vp in enumerate(visible_patches_list):
            num_visible = vp.shape[0]
            visible_patches[i, :num_visible, :] = vp
            attention_mask[i, :num_visible] = 1

        # Process through encoder (note: we're not using attention_mask in this simple implementation)
        for block in self.encoder_blocks:
            visible_patches = block(visible_patches)

        # Normalize
        visible_patches = self.encoder_norm(visible_patches)

        # Store for decoder (with attention mask to know which are real)
        self._attention_mask = attention_mask
        self._visible_indices_list = visible_indices_list

        return visible_patches, patch_mask

    def encode(self, x, patch_mask=None):
        """Encode input to latent representation.

        Args:
            x: (batch_size, input_dim) - flattened input images
            patch_mask: (batch_size, num_patches) - optional binary mask at patch level

        Returns:
            z: (batch_size, latent_dim) - latent representation
        """
        if patch_mask is None:
            # If no mask provided, process all patches (like TransformerAE)
            batch_size = x.shape[0]
            x = x.view(batch_size, self.in_channels, self.image_size, self.image_size)
            patches = self.patch_embed(x)
            patches = patches + self.pos_embed()

            for block in self.encoder_blocks:
                patches = block(patches)

            patches = self.encoder_norm(patches)
            pooled = patches.mean(dim=1)
            self._patch_mask = None
            self._encoder_output = patches
            self._attention_mask = None
            self._visible_indices_list = None  # All patches are visible
        else:
            # Encode with masking
            encoder_output, patch_mask = self.encode_with_mask(x, patch_mask)
            self._encoder_output = encoder_output
            self._patch_mask = patch_mask

            # Global average pooling over visible patches only
            # Use attention mask to only pool over real (non-padded) patches
            attention_mask = self._attention_mask.unsqueeze(-1)  # (B, max_visible, 1)
            masked_sum = (encoder_output * attention_mask).sum(dim=1)  # (B, embed_dim)
            count = attention_mask.sum(dim=1)  # (B, 1)
            pooled = masked_sum / (count + 1e-8)  # (B, embed_dim)

        # Project to latent dimension
        z = self.to_latent(pooled)

        return z

    def decode(self, z):
        """Decode latent representation to reconstruction.

        Uses stored encoder output and patch mask to create full sequence
        with mask tokens at masked positions.

        Args:
            z: (batch_size, latent_dim) - latent representation

        Returns:
            reconstruction: (batch_size, input_dim) - flattened reconstructed images
        """
        batch_size = z.shape[0]

        # Project from latent to embedding dimension
        x = self.from_latent(z)

        if self._patch_mask is None:
            # No masking, use stored encoded patches directly (like TransformerAE)
            # The encoded patches already have positional embeddings from encode
            x = self._encoder_output
        else:
            # Create full sequence with mask tokens at correct spatial positions
            full_sequence = torch.zeros(batch_size, self.num_patches, self.embed_dim, device=z.device)

            # Fill in encoder outputs at visible positions and mask tokens at masked positions
            # The encoder outputs already have positional embeddings baked in
            # Mask tokens need positional embeddings based on their spatial position
            for i in range(batch_size):
                sample_mask = self._patch_mask[i]  # (num_patches,)
                visible_indices = self._visible_indices_list[i]  # Actual indices of visible patches
                masked_indices = (sample_mask == 1).nonzero(as_tuple=True)[0]

                # Get encoder output for this sample (only real, non-padded values)
                num_visible = len(visible_indices)
                encoder_out = self._encoder_output[i, :num_visible, :]  # (num_visible, embed_dim)

                # Place encoder outputs at their correct spatial positions
                # These already have positional embeddings from the encoder
                full_sequence[i, visible_indices, :] = encoder_out

                # Place mask tokens at masked positions WITH their spatial positional embeddings
                # mask_token is [1, 1, embed_dim], squeeze to [embed_dim]
                mask_token_base = self.mask_token.squeeze(0).squeeze(0)  # (embed_dim,)

                # Add positional embeddings to mask tokens based on their spatial position
                masked_pos = self.pos_embed().squeeze(0)[masked_indices, :]  # (num_masked, embed_dim)
                masked_tokens_with_pos = mask_token_base.unsqueeze(0) + masked_pos  # (num_masked, embed_dim)

                full_sequence[i, masked_indices, :] = masked_tokens_with_pos

            # Now full_sequence has spatially-ordered patches:
            # - Visible positions: encoded patches (with pos emb from encoder)
            # - Masked positions: mask tokens (with pos emb added here)
            # All patches are in their correct spatial positions

            x = full_sequence

        # Process through decoder
        for block in self.decoder_blocks:
            x = block(x)

        # Normalize
        x = self.decoder_norm(x)

        # Reconstruct patches to image
        reconstruction = self.patch_recon(x)

        return reconstruction

    def forward(self, x, patch_mask=None):
        """Forward pass through the masked autoencoder.

        Args:
            x: (batch_size, input_dim) - flattened input images
            patch_mask: (batch_size, num_patches) - optional binary mask (1=masked, 0=visible)

        Returns:
            Dictionary with keys:
                - reconstruction: (batch_size, input_dim) - reconstructed images
                - latent: (batch_size, latent_dim) - latent representations
                - patch_mask: (batch_size, num_patches) - patch mask used
        """
        # Generate patch mask if not provided
        if patch_mask is None:
            patch_mask = create_patch_mask(
                x.shape[0], self.num_patches,
                mask_ratio=self.mask_ratio,
                device=x.device
            )

        # Encode with patch mask (encoder only sees visible patches)
        z = self.encode(x, patch_mask=patch_mask)

        # Decode to reconstruction
        reconstruction = self.decode(z)

        return {
            'reconstruction': reconstruction,
            'latent': z,
            'patch_mask': patch_mask
        }
