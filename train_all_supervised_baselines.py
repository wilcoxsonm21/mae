"""Train supervised baselines for all downstream tasks.

This script trains separate encoder+probe models for each downstream task
and compares their performance.
"""

import argparse
import yaml
import subprocess
from pathlib import Path


def train_all_tasks(config_path, no_wandb=False):
    """Train supervised baselines for all tasks.

    Args:
        config_path: Path to base configuration file
        no_wandb: Whether to disable wandb logging
    """
    # All downstream tasks
    tasks = ['rotation', 'scale', 'perspective_x', 'perspective_y', 'grid_size']

    print("="*60)
    print("Training Supervised Baselines for All Tasks")
    print("="*60)
    print(f"\nTasks: {tasks}")
    print(f"Config: {config_path}")
    print(f"Wandb: {'disabled' if no_wandb else 'enabled'}")
    print("\n")

    results = {}

    for task in tasks:
        print("\n" + "="*60)
        print(f"Training Task: {task}")
        print("="*60)

        # Build command
        cmd = [
            'python', 'train_supervised_baseline.py',
            '--config', config_path,
            '--task', task
        ]

        if no_wandb:
            cmd.append('--no-wandb')

        # Run training
        try:
            subprocess.run(cmd, check=True)
            results[task] = 'success'
            print(f"\n✓ Successfully trained baseline for {task}")
        except subprocess.CalledProcessError as e:
            results[task] = 'failed'
            print(f"\n✗ Failed to train baseline for {task}: {e}")

    # Print summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)

    for task, status in results.items():
        status_symbol = "✓" if status == 'success' else "✗"
        print(f"{status_symbol} {task}: {status}")

    print("\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train supervised baselines for all downstream tasks'
    )
    parser.add_argument('--config', type=str,
                       default='configs/supervised_baseline.yaml',
                       help='Path to configuration file')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')

    args = parser.parse_args()

    # Train all tasks
    train_all_tasks(args.config, args.no_wandb)


if __name__ == '__main__':
    main()
