#!/usr/bin/env python3
"""
Run dataset size sweep experiments for Transformer AE and MAE.

Runs all configs in configs/sweep/ sequentially.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Sweep config directory
SWEEP_CONFIG_DIR = Path(__file__).parent / 'configs' / 'sweep'

# Dataset sizes (for ordering)
N_SAMPLES = [50, 100, 200, 500, 1000, 2000, 10000, 100000, 1000000]
MODELS = ['transformer_ae', 'transformer_mae']


def run_experiment(config_path: str) -> bool:
    """Run a single experiment using main.py."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'main.py'),
        '--config', str(config_path)
    ]

    print(f"\n{'='*60}")
    print(f"Running: {config_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"\nWARNING: Experiment failed with return code {result.returncode}")
        return False
    return True


def main():
    # First, generate configs if they don't exist
    if not SWEEP_CONFIG_DIR.exists() or not list(SWEEP_CONFIG_DIR.glob('*.yaml')):
        print("Generating sweep configs...")
        from generate_sweep_configs import main as generate_configs
        generate_configs()

    # Get all configs in order
    configs_to_run = []
    for model in MODELS:
        for n in N_SAMPLES:
            config_path = SWEEP_CONFIG_DIR / f"{model}_n{n}.yaml"
            if config_path.exists():
                configs_to_run.append(config_path)
            else:
                print(f"WARNING: Config not found: {config_path}")

    print(f"\nFound {len(configs_to_run)} configs to run:")
    for config in configs_to_run:
        print(f"  - {config.name}")

    # Run all experiments
    results = {}
    for config_path in configs_to_run:
        success = run_experiment(config_path)
        results[config_path.name] = success

    # Summary
    print(f"\n{'='*60}")
    print("SWEEP SUMMARY")
    print(f"{'='*60}")

    successes = sum(1 for v in results.values() if v)
    failures = len(results) - successes

    print(f"Total: {len(results)} experiments")
    print(f"Succeeded: {successes}")
    print(f"Failed: {failures}")

    if failures > 0:
        print("\nFailed experiments:")
        for name, success in results.items():
            if not success:
                print(f"  - {name}")

    print("\nTo plot results, run: python plot_sweep_results.py")


if __name__ == '__main__':
    main()
