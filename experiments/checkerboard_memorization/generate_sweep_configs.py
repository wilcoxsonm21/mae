#!/usr/bin/env python3
"""
Generate sweep configs for dataset size experiment.

Creates configs varying n_samples for both Transformer AE and MAE models.
"""

import os
import yaml
from pathlib import Path

# Dataset sizes to sweep
N_SAMPLES = [50, 100, 200, 500, 1000, 2000, 10000, 100000, 1000000]

# Base configs
BASE_CONFIGS = {
    'transformer_ae': 'configs/transformer_ae.yaml',
    'transformer_mae': 'configs/transformer_mae.yaml',
}

# Output directory for sweep configs
SWEEP_CONFIG_DIR = Path(__file__).parent / 'configs' / 'sweep'


def load_base_config(config_path: str) -> dict:
    """Load a YAML config file."""
    full_path = Path(__file__).parent / config_path
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)


def generate_sweep_config(base_config: dict, model_name: str, n_samples: int) -> dict:
    """Generate a sweep config with modified n_samples."""
    config = base_config.copy()

    # Deep copy nested dicts
    config['dataset'] = base_config['dataset'].copy()
    config['model'] = base_config['model'].copy()
    config['model']['params'] = base_config['model']['params'].copy()

    # Update experiment name
    config['experiment_name'] = f"{model_name}_sweep_n{n_samples}"

    # Update n_samples
    config['dataset']['n_samples'] = n_samples

    # Enable early stopping on overfit
    config['early_stop_overfit'] = {
        'enabled': True,
        'patience': 10
    }

    # Update checkpoint directory
    config['checkpoint_dir'] = f"./checkpoints/checkerboard_memorization/sweep/{model_name}_n{n_samples}"

    return config


def main():
    # Create output directory
    SWEEP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    generated_configs = []

    for model_name, config_path in BASE_CONFIGS.items():
        print(f"Loading base config: {config_path}")
        base_config = load_base_config(config_path)

        for n_samples in N_SAMPLES:
            sweep_config = generate_sweep_config(base_config, model_name, n_samples)

            # Output filename
            output_filename = f"{model_name}_n{n_samples}.yaml"
            output_path = SWEEP_CONFIG_DIR / output_filename

            # Write config
            with open(output_path, 'w') as f:
                yaml.dump(sweep_config, f, default_flow_style=False, sort_keys=False)

            print(f"  Generated: {output_path}")
            generated_configs.append(str(output_path))

    print(f"\nGenerated {len(generated_configs)} configs in {SWEEP_CONFIG_DIR}")
    return generated_configs


if __name__ == '__main__':
    main()
