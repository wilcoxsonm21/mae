"""Evaluation metrics for autoencoders."""

import torch
import torch.nn as nn


def reconstruction_loss(reconstruction, target, loss_type='mse'):
    """Compute reconstruction loss.

    Args:
        reconstruction: Reconstructed tensor
        target: Target tensor
        loss_type: Type of loss ('mse', 'l1')

    Returns:
        Scalar loss value
    """
    if loss_type == 'mse':
        return nn.functional.mse_loss(reconstruction, target)
    elif loss_type == 'l1':
        return nn.functional.l1_loss(reconstruction, target)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def latent_variance(latent):
    """Compute variance of latent representations.

    This metric helps assess if the latent space is being used effectively.

    Args:
        latent: Latent tensor of shape (batch_size, latent_dim)

    Returns:
        Mean variance across latent dimensions
    """
    return latent.var(dim=0).mean()


def reconstruction_error_per_sample(reconstruction, target, loss_type='mse'):
    """Compute per-sample reconstruction error.

    Args:
        reconstruction: Reconstructed tensor of shape (batch_size, input_dim)
        target: Target tensor of shape (batch_size, input_dim)
        loss_type: Type of loss ('mse', 'l1')

    Returns:
        Tensor of shape (batch_size,) with per-sample errors
    """
    if loss_type == 'mse':
        errors = ((reconstruction - target) ** 2).mean(dim=1)
    elif loss_type == 'l1':
        errors = torch.abs(reconstruction - target).mean(dim=1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return errors


def masked_reconstruction_accuracy(reconstruction, target, mask, threshold=0.1):
    """Compute reconstruction accuracy on masked positions.

    A reconstruction is considered "accurate" if the absolute error
    is below the threshold.

    Args:
        reconstruction: Reconstructed tensor
        target: Target tensor
        mask: Binary mask (1 for masked positions)
        threshold: Threshold for considering a reconstruction accurate

    Returns:
        Accuracy on masked positions (fraction of accurate reconstructions)
    """
    errors = torch.abs(reconstruction - target)
    masked_errors = errors * mask
    accurate = (masked_errors < threshold).float() * mask

    # Compute accuracy on masked positions
    accuracy = accurate.sum() / (mask.sum() + 1e-8)
    return accuracy
