"""Masked reconstruction loss for masked autoencoders."""

import torch
import torch.nn as nn
from .base import BaseObjective


class MaskedReconstructionLoss(BaseObjective):
    """Masked reconstruction loss for masked autoencoders.

    Computes reconstruction error only at masked positions,
    following the MAE (Masked Autoencoder) approach.
    """

    def __init__(self, loss_type='mse', reduction='mean', predict_all=False):
        """Initialize masked reconstruction loss.

        Args:
            loss_type: Type of loss ('mse', 'l1', 'smooth_l1')
            reduction: Reduction method ('mean', 'sum')
            predict_all: If True, compute loss on all positions (not just masked).
                        This can be useful for some training scenarios.
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.predict_all = predict_all

        # Create base loss function (we'll apply masking manually)
        if loss_type == 'mse':
            self.base_loss = lambda x, y: (x - y) ** 2
        elif loss_type == 'l1':
            self.base_loss = lambda x, y: torch.abs(x - y)
        elif loss_type == 'smooth_l1':
            self.base_loss = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def _expand_patch_mask_to_pixels(self, patch_mask, input_dim):
        """Expand patch-level mask to pixel-level mask.

        Args:
            patch_mask: (batch_size, num_patches) - binary mask at patch level
            input_dim: Total number of pixels (channels * height * width)

        Returns:
            pixel_mask: (batch_size, input_dim) - binary mask at pixel level
        """
        batch_size, num_patches = patch_mask.shape

        # Infer image dimensions (assume square image, single channel for simplicity)
        # For multi-channel, this still works as we repeat per-pixel
        num_patches_per_side = int(num_patches ** 0.5)
        image_size = int(input_dim ** 0.5)
        patch_size = image_size // num_patches_per_side
        in_channels = input_dim // (image_size * image_size)

        # Reshape patch mask to 2D grid
        # (B, num_patches) -> (B, num_patches_h, num_patches_w)
        patch_mask_2d = patch_mask.view(batch_size, num_patches_per_side, num_patches_per_side)

        # Expand each patch to its full pixel size using repeat_interleave
        # (B, num_patches_h, num_patches_w) -> (B, H, W)
        pixel_mask_2d = patch_mask_2d.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)

        # Expand for channels and flatten
        # (B, H, W) -> (B, C, H, W) -> (B, input_dim)
        pixel_mask = pixel_mask_2d.unsqueeze(1).expand(-1, in_channels, -1, -1)
        pixel_mask = pixel_mask.reshape(batch_size, input_dim)

        return pixel_mask

    def forward(self, model_output, target, **kwargs):
        """Compute masked reconstruction loss.

        Args:
            model_output: Dictionary from model.forward() containing:
                         - 'reconstruction': Reconstructed output
                         - 'mask': Binary mask at pixel level (1 for masked positions)
                         OR 'patch_mask': Binary mask at patch level (for transformer MAE)
            target: Ground truth target tensor
            **kwargs: Additional arguments (unused)

        Returns:
            Dictionary containing:
                - 'loss': Total loss value
                - 'reconstruction_loss': Reconstruction loss (same as 'loss')
                - 'masked_loss': Loss on masked positions only
                - 'unmasked_loss': Loss on unmasked positions only
        """
        reconstruction = model_output['reconstruction']

        # Check for patch_mask (transformer MAE) or pixel mask
        patch_mask = model_output.get('patch_mask', None)
        mask = model_output.get('mask', None)

        # If we have a patch mask, expand it to pixel level
        if patch_mask is not None and mask is None:
            mask = self._expand_patch_mask_to_pixels(patch_mask, target.shape[1])

        # Compute element-wise loss
        elementwise_loss = self.base_loss(reconstruction, target)

        if mask is not None and not self.predict_all:
            # Compute loss only on masked positions
            masked_loss_values = elementwise_loss * mask
            unmasked_loss_values = elementwise_loss * (1 - mask)

            # Reduce based on reduction method
            if self.reduction == 'mean':
                # Average over masked positions
                masked_loss = masked_loss_values.sum() / (mask.sum() + 1e-8)
                unmasked_loss = unmasked_loss_values.sum() / ((1 - mask).sum() + 1e-8)
                loss = masked_loss  # Primary loss is on masked positions
            elif self.reduction == 'sum':
                masked_loss = masked_loss_values.sum()
                unmasked_loss = unmasked_loss_values.sum()
                loss = masked_loss
            else:
                raise ValueError(f"Unknown reduction: {self.reduction}")
        else:
            # No masking, compute loss on all positions
            if self.reduction == 'mean':
                loss = elementwise_loss.mean()
            elif self.reduction == 'sum':
                loss = elementwise_loss.sum()
            else:
                raise ValueError(f"Unknown reduction: {self.reduction}")
            masked_loss = loss
            unmasked_loss = torch.tensor(0.0, device=loss.device)

        return {
            'loss': loss,
            'reconstruction_loss': loss,
            'masked_loss': masked_loss,
            'unmasked_loss': unmasked_loss
        }
