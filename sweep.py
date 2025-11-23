"""Hyperparameter sweep runner with caching and result plotting."""

import argparse
import yaml
import copy
import numpy as np
from pathlib import Path

# Import training function and utilities
from main import train, load_config
from utils import experiment_exists, load_results, compute_config_hash


def set_nested_value(config, key_path, value):
    """Set a value in a nested dictionary using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Path to key in dot notation (e.g., 'model.params.latent_dim')
        value: Value to set

    Example:
        set_nested_value(config, 'model.params.latent_dim', 64)
    """
    keys = key_path.split('.')
    current = config

    # Navigate to the nested location
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Set the value
    current[keys[-1]] = value


def get_nested_value(config, key_path):
    """Get a value from a nested dictionary using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Path to key in dot notation

    Returns:
        The value at the specified path
    """
    keys = key_path.split('.')
    current = config

    for key in keys:
        current = current[key]

    return current


def generate_sweep_configs(base_config, sweep_param, sweep_values):
    """Generate configurations for sweep.

    Args:
        base_config: Base configuration dictionary
        sweep_param: Parameter to sweep (dot notation, e.g., 'model.params.latent_dim')
        sweep_values: List of values to sweep over (quantitative or categorical)

    Returns:
        List of configuration dictionaries
    """
    configs = []

    for value in sweep_values:
        # Create a deep copy of base config
        config = copy.deepcopy(base_config)

        # Set the sweep parameter
        set_nested_value(config, sweep_param, value)

        # Update experiment name to include sweep info
        param_short = sweep_param.split('.')[-1]
        config['experiment_name'] = f"{base_config.get('experiment_name', 'sweep')}_{param_short}={value}"

        configs.append(config)

    return configs


def run_sweep(base_config_path, sweep_param, sweep_values,
              base_results_dir='trained_models',
              wandb_log=False,
              force_rerun=False):
    """Run hyperparameter sweep.

    Args:
        base_config_path: Path to base configuration file
        sweep_param: Parameter to sweep (dot notation)
        sweep_values: List of values to sweep over
        base_results_dir: Directory to save results
        wandb_log: Whether to log to wandb
        force_rerun: If True, rerun even if cached results exist

    Returns:
        Dictionary mapping values to results
    """
    # Load base config
    base_config = load_config(base_config_path)

    # Generate sweep configs
    print(f"\nGenerating sweep over parameter: {sweep_param}")
    print(f"Values: {sweep_values}")
    sweep_configs = generate_sweep_configs(base_config, sweep_param, sweep_values)

    print(f"\nTotal experiments: {len(sweep_configs)}")

    # Run experiments
    sweep_results = {}

    for i, (value, config) in enumerate(zip(sweep_values, sweep_configs)):
        print(f"\n{'='*80}")
        print(f"Experiment {i+1}/{len(sweep_configs)}: {sweep_param} = {value}")
        print(f"{'='*80}")

        # Check if experiment already exists
        config_hash = compute_config_hash(config)
        if not force_rerun and experiment_exists(config, base_results_dir):
            print(f"Experiment already exists (hash: {config_hash}). Loading cached results...")
            results = load_results(config, base_results_dir)
            sweep_results[value] = results
        else:
            print(f"Running experiment (hash: {config_hash})...")
            # Run training with hash-based saving
            results = train(
                config,
                wandb_log=wandb_log,
                use_hash_dir=True,
                base_results_dir=base_results_dir
            )
            sweep_results[value] = results

    print(f"\n{'='*80}")
    print("Sweep complete!")
    print(f"{'='*80}\n")

    return sweep_results, sweep_param


def print_sweep_summary(sweep_results, sweep_param):
    """Print summary of sweep results.

    Args:
        sweep_results: Dictionary mapping values to results
        sweep_param: Parameter that was swept
    """
    print("\nSweep Results Summary:")
    print(f"Parameter: {sweep_param}")
    print(f"\n{'Value':<20} {'Train Loss':<15} {'Val Loss':<15} {'Val Recon MSE':<15}")
    print("-" * 65)

    for value, results in sweep_results.items():
        train_loss = results['train'].get('objective_loss', float('nan'))
        val_loss = results['val'].get('objective_loss', float('nan'))
        val_recon = results['val'].get('reconstruction_loss_mse', float('nan'))

        print(f"{str(value):<20} {train_loss:<15.6f} {val_loss:<15.6f} {val_recon:<15.6f}")

    # Find best value
    best_value = min(sweep_results.keys(),
                    key=lambda v: sweep_results[v]['val'].get('objective_loss', float('inf')))
    best_val_loss = sweep_results[best_value]['val'].get('objective_loss', float('nan'))

    print(f"\nBest {sweep_param}: {best_value} (Val Loss: {best_val_loss:.6f})")


def main():
    """Main entry point for sweep runner."""
    parser = argparse.ArgumentParser(description='Run hyperparameter sweeps')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to base configuration file')
    parser.add_argument('--sweep-config', type=str, required=True,
                       help='Path to sweep configuration file')
    parser.add_argument('--results-dir', type=str, default='trained_models',
                       help='Directory to save results')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable wandb logging')
    parser.add_argument('--force', action='store_true',
                       help='Force rerun even if cached results exist')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots of sweep results')

    args = parser.parse_args()

    # Load sweep configuration
    with open(args.sweep_config, 'r') as f:
        sweep_config = yaml.safe_load(f)

    sweep_param = sweep_config['parameter']
    sweep_values = sweep_config['values']

    # Determine if quantitative or categorical
    is_quantitative = all(isinstance(v, (int, float)) for v in sweep_values)
    if is_quantitative:
        print(f"Detected quantitative sweep over {len(sweep_values)} values")
    else:
        print(f"Detected categorical sweep over {len(sweep_values)} values")

    # Run sweep
    sweep_results, sweep_param = run_sweep(
        args.config,
        sweep_param,
        sweep_values,
        args.results_dir,
        args.wandb,
        args.force
    )

    # Print summary
    print_sweep_summary(sweep_results, sweep_param)

    # Generate plots if requested
    if args.plot:
        from evaluation.plotting import plot_sweep_results

        output_dir = Path(args.results_dir) / 'sweep_plots'
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_sweep_results(
            sweep_results,
            sweep_param,
            output_dir,
            is_quantitative=is_quantitative
        )

        print(f"\nPlots saved to: {output_dir}")


if __name__ == '__main__':
    main()
