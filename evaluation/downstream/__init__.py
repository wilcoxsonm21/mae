"""Downstream evaluation for latent representations."""

from .probes import LatentProbe, MultiTaskLatentProbe
from .trainer import ProbeTrainer

__all__ = ['LatentProbe', 'MultiTaskLatentProbe', 'ProbeTrainer']
