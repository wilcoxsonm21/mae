"""Probe models for downstream evaluation of latent representations."""

import torch
import torch.nn as nn


class LatentProbe(nn.Module):
    """Probe that predicts a property from latent codes.

    Supports two architectures:
    - 'linear': Single linear layer (tests linear separability)
    - 'mlp': 3-layer MLP with hidden layers (tests non-linear separability)
    """

    def __init__(self, latent_dim, hidden_dim=64, output_dim=1, task_type='regression',
                 dropout=0.1, probe_type='mlp'):
        """Initialize latent probe.

        Args:
            latent_dim: Dimension of encoder latent space
            hidden_dim: Hidden layer size (only used for MLP)
            output_dim: Output dimension (1 for regression, num_classes for classification)
            task_type: 'regression' or 'classification'
            dropout: Dropout probability for regularization (only used for MLP)
            probe_type: 'linear' or 'mlp'
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.task_type = task_type
        self.probe_type = probe_type

        if probe_type == 'linear':
            # Single linear layer (no hidden layers)
            self.net = nn.Linear(latent_dim, output_dim)
        elif probe_type == 'mlp':
            # 3-layer MLP with 2 ReLUs
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            raise ValueError(f"Unknown probe_type: {probe_type}. Choose 'linear' or 'mlp'.")

    def forward(self, z):
        """Forward pass.

        Args:
            z: Latent codes of shape (batch_size, latent_dim)

        Returns:
            Predictions of shape (batch_size, output_dim)
        """
        return self.net(z)


class MultiTaskLatentProbe(nn.Module):
    """Multi-task MLP predicting all properties simultaneously.

    Uses a shared trunk with task-specific heads to leverage correlations
    between different generation parameters.
    """

    def __init__(self, latent_dim, shared_dim=128, head_dim=64, dropout=0.1):
        """Initialize multi-task probe.

        Args:
            latent_dim: Dimension of encoder latent space
            shared_dim: Dimension of shared trunk
            head_dim: Dimension of task-specific heads
            dropout: Dropout probability
        """
        super().__init__()

        self.latent_dim = latent_dim

        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, head_dim),
            nn.ReLU()
        )

        # Task-specific heads
        # Regression heads (output_dim=1)
        self.rotation_head = nn.Linear(head_dim, 1)
        self.scale_head = nn.Linear(head_dim, 1)
        self.perspective_x_head = nn.Linear(head_dim, 1)
        self.perspective_y_head = nn.Linear(head_dim, 1)

        # Classification head (4 grid sizes: 2, 4, 8, 16)
        self.grid_size_head = nn.Linear(head_dim, 4)

    def forward(self, z):
        """Forward pass.

        Args:
            z: Latent codes of shape (batch_size, latent_dim)

        Returns:
            Dictionary of predictions for each task:
                - 'rotation': (batch_size, 1)
                - 'scale': (batch_size, 1)
                - 'perspective_x': (batch_size, 1)
                - 'perspective_y': (batch_size, 1)
                - 'grid_size': (batch_size, 4) - logits
        """
        # Shared representation
        shared_repr = self.shared(z)

        # Task-specific predictions
        return {
            'rotation': self.rotation_head(shared_repr),
            'scale': self.scale_head(shared_repr),
            'perspective_x': self.perspective_x_head(shared_repr),
            'perspective_y': self.perspective_y_head(shared_repr),
            'grid_size': self.grid_size_head(shared_repr)  # Logits
        }
