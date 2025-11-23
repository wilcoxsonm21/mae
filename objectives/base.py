"""Base class for training objectives."""

from abc import ABC, abstractmethod
import torch.nn as nn


class BaseObjective(ABC, nn.Module):
    """Abstract base class for training objectives.

    This class defines the interface for all training objectives,
    separating loss computation from model architecture.
    """

    def __init__(self):
        """Initialize the base objective."""
        super().__init__()

    @abstractmethod
    def forward(self, model_output, target, **kwargs):
        """Compute the loss.

        Args:
            model_output: Dictionary of model outputs (from model.forward())
            target: Ground truth target tensor
            **kwargs: Additional arguments specific to the objective

        Returns:
            Dictionary containing:
                - 'loss': Total loss value (scalar tensor)
                - Additional metrics specific to the objective
        """
        pass

    def get_name(self):
        """Get the name of the objective.

        Returns:
            String name of the objective
        """
        return self.__class__.__name__
