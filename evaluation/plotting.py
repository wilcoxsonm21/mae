"""Plotting utilities for sweep results."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_sweep_results(sweep_results, sweep_param, output_dir,
                       is_quantitative=True, metrics=None):
    """Plot sweep results for different metrics.

    Args:
        sweep_results: Dictionary mapping values to results
        sweep_param: Parameter that was swept
        output_dir: Directory to save plots
        is_quantitative: Whether the sweep parameter is quantitative or categorical
        metrics: List of metrics to plot. If None, plots all common metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default metrics to plot
    if metrics is None:
        metrics = [
            ('objective_loss', 'Objective Loss'),
            ('reconstruction_loss_mse', 'Reconstruction Loss (MSE)'),
            ('reconstruction_loss_l1', 'Reconstruction Loss (L1)'),
            ('latent_variance', 'Latent Variance')
        ]

    # Extract values and sort if quantitative
    values = list(sweep_results.keys())
    if is_quantitative:
        # Sort by numeric value
        sorted_indices = np.argsort(values)
        values = [values[i] for i in sorted_indices]

    # Plot each metric
    for metric_key, metric_name in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Extract metric values
        train_values = []
        val_values = []

        for value in values:
            results = sweep_results[value]
            train_val = results['train'].get(metric_key, None)
            val_val = results['val'].get(metric_key, None)

            train_values.append(train_val if train_val is not None else np.nan)
            val_values.append(val_val if val_val is not None else np.nan)

        # Skip if all values are NaN
        if all(np.isnan(train_values)) and all(np.isnan(val_values)):
            plt.close(fig)
            continue

        # Plot training metric
        if is_quantitative:
            axes[0].plot(values, train_values, 'o-', linewidth=2, markersize=8,
                        label='Train', color='#2E86AB')
            axes[0].set_xlabel(sweep_param.split('.')[-1])
        else:
            x_pos = np.arange(len(values))
            axes[0].bar(x_pos, train_values, color='#2E86AB', alpha=0.7)
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels([str(v) for v in values], rotation=45, ha='right')
            axes[0].set_xlabel(sweep_param.split('.')[-1])

        axes[0].set_ylabel(metric_name)
        axes[0].set_title(f'Training {metric_name}')
        axes[0].grid(True, alpha=0.3)

        # Plot validation metric
        if is_quantitative:
            axes[1].plot(values, val_values, 'o-', linewidth=2, markersize=8,
                        label='Validation', color='#A23B72')
            axes[1].set_xlabel(sweep_param.split('.')[-1])
        else:
            x_pos = np.arange(len(values))
            axes[1].bar(x_pos, val_values, color='#A23B72', alpha=0.7)
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels([str(v) for v in values], rotation=45, ha='right')
            axes[1].set_xlabel(sweep_param.split('.')[-1])

        axes[1].set_ylabel(metric_name)
        axes[1].set_title(f'Validation {metric_name}')
        axes[1].grid(True, alpha=0.3)

        # Highlight best value on validation plot
        if not all(np.isnan(val_values)):
            best_idx = np.nanargmin(val_values) if metric_key.endswith('loss') else np.nanargmax(val_values)
            if is_quantitative:
                axes[1].axvline(values[best_idx], color='red', linestyle='--', alpha=0.5,
                              label=f'Best: {values[best_idx]}')
            else:
                axes[1].axvline(best_idx, color='red', linestyle='--', alpha=0.5,
                              label=f'Best: {values[best_idx]}')
            axes[1].legend()

        fig.suptitle(f'{metric_name} vs {sweep_param}', fontsize=14, y=1.02)
        plt.tight_layout()

        # Save plot
        safe_metric_name = metric_key.replace('_', '-')
        plot_path = output_dir / f'{safe_metric_name}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {plot_path}")
        plt.close(fig)

    # Create combined plot (train + val on same plot)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use objective loss for combined plot
    train_losses = []
    val_losses = []

    for value in values:
        results = sweep_results[value]
        train_val = results['train'].get('objective_loss', None)
        val_val = results['val'].get('objective_loss', None)

        train_losses.append(train_val if train_val is not None else np.nan)
        val_losses.append(val_val if val_val is not None else np.nan)

    if is_quantitative:
        ax.plot(values, train_losses, 'o-', linewidth=2, markersize=8,
               label='Train Loss', color='#2E86AB')
        ax.plot(values, val_losses, 's-', linewidth=2, markersize=8,
               label='Val Loss', color='#A23B72')
        ax.set_xlabel(sweep_param.split('.')[-1], fontsize=12)

        # Highlight best validation value
        if not all(np.isnan(val_losses)):
            best_idx = np.nanargmin(val_losses)
            ax.axvline(values[best_idx], color='red', linestyle='--', alpha=0.5,
                      label=f'Best Val: {values[best_idx]}')
    else:
        x_pos = np.arange(len(values))
        width = 0.35
        ax.bar(x_pos - width/2, train_losses, width, label='Train Loss',
              color='#2E86AB', alpha=0.7)
        ax.bar(x_pos + width/2, val_losses, width, label='Val Loss',
              color='#A23B72', alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(v) for v in values], rotation=45, ha='right')
        ax.set_xlabel(sweep_param.split('.')[-1], fontsize=12)

        # Highlight best validation value
        if not all(np.isnan(val_losses)):
            best_idx = np.nanargmin(val_losses)
            ax.axvline(best_idx, color='red', linestyle='--', alpha=0.5, linewidth=2,
                      label=f'Best Val: {values[best_idx]}')

    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Train vs Validation Loss\nSweep over {sweep_param}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    combined_path = output_dir / 'combined_loss.png'
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined plot: {combined_path}")
    plt.close(fig)


def plot_sweep_comparison(sweep_results_list, sweep_params, labels,
                         output_path, metric='objective_loss'):
    """Compare multiple sweeps on the same plot.

    Args:
        sweep_results_list: List of sweep results dictionaries
        sweep_params: List of parameter names for each sweep
        labels: List of labels for each sweep
        output_path: Path to save the plot
        metric: Metric to compare
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

    for i, (sweep_results, sweep_param, label) in enumerate(zip(sweep_results_list, sweep_params, labels)):
        values = sorted(sweep_results.keys())
        val_values = [sweep_results[v]['val'].get(metric, np.nan) for v in values]

        color = colors[i % len(colors)]
        ax.plot(values, val_values, 'o-', linewidth=2, markersize=8,
               label=label, color=color)

    ax.set_xlabel('Parameter Value', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title('Sweep Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot: {output_path}")
    plt.close(fig)
