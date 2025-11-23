"""Utility functions for data handling."""

import torch


def create_random_mask(batch_size, input_dim, mask_ratio=0.5, device='cpu'):
    """Create a random binary mask for masked autoencoding.

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
