"""Utilities for saving and loading experiment results."""

import json
import shutil
from pathlib import Path
from .config_hash import compute_config_hash, get_experiment_dir


def save_results(config, results, best_model_path, final_model_path, base_dir='trained_models'):
    """Save experiment results with config hash.

    Args:
        config: Configuration dictionary
        results: Dictionary of results (train and val metrics)
        best_model_path: Path to best model checkpoint
        final_model_path: Path to final model checkpoint
        base_dir: Base directory for saving results

    Returns:
        Path to the experiment directory
    """
    exp_dir = get_experiment_dir(config, base_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = exp_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Save results
    results_path = exp_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Copy model checkpoints
    if best_model_path and Path(best_model_path).exists():
        shutil.copy(best_model_path, exp_dir / 'best_model.pt')

    if final_model_path and Path(final_model_path).exists():
        shutil.copy(final_model_path, exp_dir / 'final_model.pt')

    # Save metadata
    metadata = {
        'config_hash': compute_config_hash(config),
        'experiment_complete': True
    }
    metadata_path = exp_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Results saved to: {exp_dir}")
    return exp_dir


def load_results(config, base_dir='trained_models'):
    """Load experiment results if they exist.

    Args:
        config: Configuration dictionary
        base_dir: Base directory for experiments

    Returns:
        Dictionary of results or None if not found
    """
    exp_dir = get_experiment_dir(config, base_dir)
    results_path = exp_dir / 'results.json'

    if not results_path.exists():
        return None

    with open(results_path, 'r') as f:
        results = json.load(f)

    return results


def experiment_exists(config, base_dir='trained_models'):
    """Check if experiment has already been run.

    Args:
        config: Configuration dictionary
        base_dir: Base directory for experiments

    Returns:
        Boolean indicating if experiment exists and is complete
    """
    exp_dir = get_experiment_dir(config, base_dir)
    metadata_path = exp_dir / 'metadata.json'

    if not metadata_path.exists():
        return False

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata.get('experiment_complete', False)
    except Exception:
        return False


def get_experiment_summary(config, base_dir='trained_models'):
    """Get a summary of experiment results.

    Args:
        config: Configuration dictionary
        base_dir: Base directory for experiments

    Returns:
        Dictionary with summary information
    """
    exp_dir = get_experiment_dir(config, base_dir)
    config_hash = compute_config_hash(config)

    if not experiment_exists(config, base_dir):
        return {
            'config_hash': config_hash,
            'exists': False
        }

    results = load_results(config, base_dir)

    return {
        'config_hash': config_hash,
        'exists': True,
        'exp_dir': str(exp_dir),
        'train_loss': results['train'].get('objective_loss', None),
        'val_loss': results['val'].get('objective_loss', None),
        'train_reconstruction_mse': results['train'].get('reconstruction_loss_mse', None),
        'val_reconstruction_mse': results['val'].get('reconstruction_loss_mse', None)
    }
