"""Utilities for hashing configurations and managing experiment directories."""

import hashlib
import json
from pathlib import Path


def compute_config_hash(config, exclude_keys=None):
    """Compute a deterministic hash of a configuration.

    Args:
        config: Configuration dictionary
        exclude_keys: List of top-level keys to exclude from hashing
                     (e.g., ['experiment_name', 'checkpoint_dir', 'wandb_project'])

    Returns:
        String hash (8 characters)
    """
    if exclude_keys is None:
        exclude_keys = ['experiment_name', 'checkpoint_dir', 'wandb_project',
                       'visualization_frequency', 'device']

    # Create a copy and remove excluded keys
    config_copy = config.copy()
    for key in exclude_keys:
        config_copy.pop(key, None)

    # Convert to JSON string with sorted keys for determinism
    config_str = json.dumps(config_copy, sort_keys=True)

    # Compute SHA256 hash and take first 8 characters
    hash_obj = hashlib.sha256(config_str.encode('utf-8'))
    config_hash = hash_obj.hexdigest()[:8]

    return config_hash


def get_experiment_dir(config, base_dir='trained_models'):
    """Get the experiment directory based on config hash.

    Args:
        config: Configuration dictionary
        base_dir: Base directory for all experiments

    Returns:
        Path object for the experiment directory
    """
    config_hash = compute_config_hash(config)
    exp_dir = Path(base_dir) / config_hash
    return exp_dir


def get_config_summary(config):
    """Get a human-readable summary of key config parameters.

    Args:
        config: Configuration dictionary

    Returns:
        String summary
    """
    summary_parts = []

    # Model info
    model_type = config.get('model', {}).get('type', 'unknown')
    summary_parts.append(f"model={model_type}")

    # Dataset info
    dataset_name = config.get('dataset', {}).get('dataset_name', 'unknown')
    summary_parts.append(f"data={dataset_name}")

    # Key hyperparameters
    model_params = config.get('model', {}).get('params', {})
    latent_dim = model_params.get('latent_dim', 'unknown')
    summary_parts.append(f"latent={latent_dim}")

    optimizer = config.get('optimizer', {})
    lr = optimizer.get('lr', 'unknown')
    summary_parts.append(f"lr={lr}")

    return "_".join(summary_parts)
