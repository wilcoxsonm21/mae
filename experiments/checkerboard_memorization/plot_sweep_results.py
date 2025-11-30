#!/usr/bin/env python3
"""
Plot results from dataset size sweep experiments.

Creates a plot showing grid_acc vs n_samples for both Transformer AE and MAE.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import csv

# Dataset sizes used in sweep
N_SAMPLES = [50, 100, 200, 500, 1000, 2000, 10000, 100000, 1000000]
MODELS = ['transformer_ae', 'transformer_mae']

# Checkpoint base directory for sweep experiments
CHECKPOINT_BASE = Path(__file__).parent.parent.parent / 'checkpoints' / 'checkerboard_memorization' / 'sweep'

# Output directory
OUTPUT_DIR = Path(__file__).parent


def load_results(checkpoint_dir: Path) -> dict:
    """Load results from a checkpoint directory."""
    results = {
        'train_loss': None,
        'val_loss': None,
        'grid_accuracy': None,
        'epochs_trained': None,
        'early_stopped': False,
    }

    # Load checkpoint
    checkpoint_path = checkpoint_dir / 'best_model.pt'
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            results['train_loss'] = checkpoint.get('train_loss')
            results['train_loss_eval_mode'] = checkpoint.get('train_loss_eval_mode')
            results['val_loss'] = checkpoint.get('val_loss')
            results['epochs_trained'] = checkpoint.get('epoch', 0) + 1

            # Check if early stopped
            config = checkpoint.get('config', {})
            num_epochs = config.get('num_epochs', 300)
            if results['epochs_trained'] < num_epochs:
                results['early_stopped'] = True
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}")

    # Load downstream results
    downstream_path = checkpoint_dir / 'downstream_eval' / 'results.json'
    if downstream_path.exists():
        try:
            with open(downstream_path, 'r') as f:
                downstream = json.load(f)
            results['grid_accuracy'] = downstream.get('grid_size', {}).get('accuracy')
            results['composite_score'] = downstream.get('composite_score')
            results['f1_macro'] = downstream.get('grid_size', {}).get('f1_macro')
        except Exception as e:
            print(f"  Warning: Could not load downstream results: {e}")

    return results


def collect_all_results() -> dict:
    """Collect results from all sweep experiments."""
    all_results = {}

    for model in MODELS:
        all_results[model] = {}
        for n in N_SAMPLES:
            checkpoint_dir = CHECKPOINT_BASE / f"{model}_n{n}"
            if checkpoint_dir.exists():
                print(f"Loading {model} n={n}...")
                results = load_results(checkpoint_dir)
                all_results[model][n] = results
            else:
                print(f"  Missing: {checkpoint_dir}")
                all_results[model][n] = None

    return all_results


def plot_sweep_results(all_results: dict):
    """Create sweep results plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dataset Size Sweep: Transformer AE vs MAE\n(Checkerboard Memorization, 4 grid classes)',
                 fontsize=14, fontweight='bold')

    colors = {'transformer_ae': 'blue', 'transformer_mae': 'orange'}
    labels = {'transformer_ae': 'Transformer AE', 'transformer_mae': 'Transformer MAE'}
    markers = {'transformer_ae': 'o', 'transformer_mae': 's'}

    # Plot 1: Grid Accuracy vs Dataset Size (main result)
    ax = axes[0, 0]
    for model in MODELS:
        n_vals = []
        acc_vals = []
        for n in N_SAMPLES:
            results = all_results[model].get(n)
            if results and results.get('grid_accuracy') is not None:
                n_vals.append(n)
                acc_vals.append(results['grid_accuracy'])
        if n_vals:
            ax.plot(n_vals, acc_vals, f'{markers[model]}-',
                   color=colors[model], label=labels[model], linewidth=2, markersize=8)
    ax.set_xlabel('Dataset Size (n_samples)')
    ax.set_ylabel('Grid Size Classification Accuracy')
    ax.set_title('Grid Accuracy vs Dataset Size')
    ax.set_xscale('log')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Random chance (25%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Train vs Val Loss
    ax = axes[0, 1]
    for model in MODELS:
        n_vals = []
        train_vals = []
        val_vals = []
        for n in N_SAMPLES:
            results = all_results[model].get(n)
            if results and results.get('val_loss') is not None:
                n_vals.append(n)
                train_vals.append(results.get('train_loss_eval_mode') or results.get('train_loss'))
                val_vals.append(results['val_loss'])
        if n_vals:
            ax.plot(n_vals, train_vals, f'{markers[model]}--',
                   color=colors[model], alpha=0.7, label=f'{labels[model]} (train)')
            ax.plot(n_vals, val_vals, f'{markers[model]}-',
                   color=colors[model], label=f'{labels[model]} (val)')
    ax.set_xlabel('Dataset Size (n_samples)')
    ax.set_ylabel('Loss')
    ax.set_title('Train/Val Loss vs Dataset Size')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Epochs trained (early stopping indicator)
    ax = axes[1, 0]
    for model in MODELS:
        n_vals = []
        epoch_vals = []
        early_stops = []
        for n in N_SAMPLES:
            results = all_results[model].get(n)
            if results and results.get('epochs_trained') is not None:
                n_vals.append(n)
                epoch_vals.append(results['epochs_trained'])
                early_stops.append(results.get('early_stopped', False))
        if n_vals:
            ax.plot(n_vals, epoch_vals, f'{markers[model]}-',
                   color=colors[model], label=labels[model], linewidth=2, markersize=8)
            # Mark early stops
            for i, (n, epochs, stopped) in enumerate(zip(n_vals, epoch_vals, early_stops)):
                if stopped:
                    ax.scatter([n], [epochs], color='red', s=100, zorder=5, marker='x')
    ax.set_xlabel('Dataset Size (n_samples)')
    ax.set_ylabel('Epochs Trained')
    ax.set_title('Training Duration (X = early stopped)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Composite Score vs Dataset Size
    ax = axes[1, 1]
    for model in MODELS:
        n_vals = []
        score_vals = []
        for n in N_SAMPLES:
            results = all_results[model].get(n)
            if results and results.get('composite_score') is not None:
                n_vals.append(n)
                score_vals.append(results['composite_score'])
        if n_vals:
            ax.plot(n_vals, score_vals, f'{markers[model]}-',
                   color=colors[model], label=labels[model], linewidth=2, markersize=8)
    ax.set_xlabel('Dataset Size (n_samples)')
    ax.set_ylabel('Composite Score')
    ax.set_title('Downstream Composite Score vs Dataset Size')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / 'sweep_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    return fig


def save_csv(all_results: dict):
    """Save results to CSV."""
    csv_path = OUTPUT_DIR / 'sweep_results.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'n_samples', 'grid_accuracy', 'f1_macro', 'composite_score',
                        'train_loss', 'val_loss', 'epochs_trained', 'early_stopped'])

        for model in MODELS:
            for n in N_SAMPLES:
                results = all_results[model].get(n)
                if results:
                    writer.writerow([
                        model,
                        n,
                        results.get('grid_accuracy', ''),
                        results.get('f1_macro', ''),
                        results.get('composite_score', ''),
                        results.get('train_loss_eval_mode') or results.get('train_loss', ''),
                        results.get('val_loss', ''),
                        results.get('epochs_trained', ''),
                        results.get('early_stopped', '')
                    ])
                else:
                    writer.writerow([model, n, '', '', '', '', '', '', ''])

    print(f"CSV saved to: {csv_path}")


def main():
    print("=" * 60)
    print("Dataset Size Sweep Results")
    print("=" * 60)

    # Collect all results
    print("\nCollecting results...")
    all_results = collect_all_results()

    # Count available results
    total = 0
    available = 0
    for model in MODELS:
        for n in N_SAMPLES:
            total += 1
            if all_results[model].get(n) and all_results[model][n].get('grid_accuracy') is not None:
                available += 1

    print(f"\nFound {available}/{total} experiment results")

    if available == 0:
        print("\nNo results available. Run the sweep first:")
        print("  python run_sweep.py")
        return

    # Print summary table
    print("\n" + "=" * 60)
    print("Grid Accuracy Summary")
    print("=" * 60)
    print(f"{'n_samples':>10} | {'AE':>12} | {'MAE':>12}")
    print("-" * 40)
    for n in N_SAMPLES:
        ae_acc = all_results['transformer_ae'].get(n, {})
        mae_acc = all_results['transformer_mae'].get(n, {})
        ae_str = f"{ae_acc.get('grid_accuracy', 0):.3f}" if ae_acc and ae_acc.get('grid_accuracy') else "N/A"
        mae_str = f"{mae_acc.get('grid_accuracy', 0):.3f}" if mae_acc and mae_acc.get('grid_accuracy') else "N/A"
        print(f"{n:>10} | {ae_str:>12} | {mae_str:>12}")

    # Create plot
    print("\nCreating plot...")
    plot_sweep_results(all_results)

    # Save CSV
    save_csv(all_results)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
