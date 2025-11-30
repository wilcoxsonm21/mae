"""Utility functions for data handling."""

import torch


def create_random_mask(batch_size, input_dim, mask_ratio=0.5, device='cpu'):
    """Create a random binary mask for masked autoencoding (pixel-level).

    Args:
        batch_size: Number of samples in the batch
        input_dim: Input dimension
        mask_ratio: Fraction of features to mask
        device: Device to create the mask on

    Returns:
        Binary mask tensor of shape (batch_size, input_dim)
        where 1 indicates masked positions
    """
    mask = torch.rand(batch_size, input_dim, device=device) < mask_ratio
    return mask.float()


def create_patch_mask(batch_size, num_patches, mask_ratio=0.75, device='cpu'):
    """Create a random binary mask at the patch level for MAE.

    Args:
        batch_size: Number of samples in the batch
        num_patches: Number of patches (e.g., 64 for 8x8 grid)
        mask_ratio: Fraction of patches to mask (default 0.75 per MAE paper)
        device: Device to create the mask on

    Returns:
        Binary mask tensor of shape (batch_size, num_patches)
        where 1 indicates masked patches, 0 indicates visible patches
    """
    num_masked = int(num_patches * mask_ratio)

    # Create mask for each sample in batch
    mask = torch.zeros(batch_size, num_patches, device=device)

    for i in range(batch_size):
        # Randomly select patches to mask
        masked_indices = torch.randperm(num_patches, device=device)[:num_masked]
        mask[i, masked_indices] = 1.0

    return mask


def apply_mask(x, mask, mask_value=0.0):
    """Apply a mask to input data.

    Args:
        x: Input tensor of shape (batch_size, input_dim)
        mask: Binary mask of shape (batch_size, input_dim)
        mask_value: Value to use for masked positions

    Returns:
        Masked input tensor
    """
    return x * (1 - mask) + mask_value * mask
