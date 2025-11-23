"""Standard reconstruction loss for autoencoders."""

import torch
import torch.nn as nn
from .base import BaseObjective


class ReconstructionLoss(BaseObjective):
    """Standard reconstruction loss for autoencoders.

    Computes the reconstruction error between input and output.
    Supports different loss types (MSE, L1, smooth L1).
    """

    def __init__(self, loss_type='mse', reduction='mean'):
        """Initialize reconstruction loss.

        Args:
            loss_type: Type of loss ('mse', 'l1', 'smooth_l1')
            reduction: Reduction method ('mean', 'sum')
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

        # Create loss function
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, model_output, target, **kwargs):
        """Compute reconstruction loss.

        Args:
            model_output: Dictionary from model.forward() containing 'reconstruction'
            target: Ground truth target tensor
            **kwargs: Additional arguments (unused)

        Returns:
            Dictionary containing:
                - 'loss': Total loss value
                - 'reconstruction_loss': Reconstruction loss (same as 'loss')
        """
        reconstruction = model_output['reconstruction']
        loss = self.loss_fn(reconstruction, target)

        return {
            'loss': loss,
            'reconstruction_loss': loss
        }
