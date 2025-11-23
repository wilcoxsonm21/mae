"""Visualization for downstream evaluation."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path

from .metrics import class_to_grid_size


def plot_regression_scatter(predictions, targets, task_name, save_path=None):
    """Plot scatter of predictions vs ground truth for regression task.

    Args:
        predictions: Predicted values (n_samples, 1) or (n_samples,)
        targets: Ground truth values (n_samples, 1) or (n_samples,)
        task_name: Name of the task
        save_path: Path to save figure (if None, returns figure)

    Returns:
        matplotlib figure
    """
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Scatter plot
    ax.scatter(targets, predictions, alpha=0.5, s=20)

    # Add y=x line (perfect predictions)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

    # Labels and title
    ax.set_xlabel('Ground Truth', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title(f'{task_name} Predictions', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig


def plot_all_regression_tasks(predictions_dict, save_dir=None):
    """Plot scatter plots for all regression tasks.

    Args:
        predictions_dict: Dictionary with predictions and targets for each task
        save_dir: Directory to save figures (if None, returns figures)

    Returns:
        Dictionary of figures (if save_dir is None)
    """
    regression_tasks = ['rotation', 'scale', 'perspective_x', 'perspective_y']

    # Create save directory if needed
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    figures = {}

    for task in regression_tasks:
        if task not in predictions_dict:
            continue

        preds = predictions_dict[task]['predictions']
        targets = predictions_dict[task]['targets']

        save_path = f"{save_dir}/{task}_scatter.png" if save_dir else None
        fig = plot_regression_scatter(preds, targets, task.replace('_', ' ').title(), save_path)

        if fig is not None:
            figures[task] = fig

    return figures if not save_dir else None


def plot_confusion_matrix(predictions, targets, task_name='Grid Size',
                         class_names=['2', '4', '8', '16'], save_path=None):
    """Plot confusion matrix for classification task.

    Args:
        predictions: Predicted class indices or logits
        targets: Ground truth class indices
        task_name: Name of the task
        class_names: Names of classes
        save_path: Path to save figure (if None, returns figure)

    Returns:
        matplotlib figure
    """
    # If predictions are logits, get argmax
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    # Flatten
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Compute confusion matrix
    cm = confusion_matrix(targets, predictions)

    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'}, ax=ax)

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'{task_name} Confusion Matrix', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig


def plot_error_histogram(predictions, targets, task_name, bins=50, save_path=None):
    """Plot histogram of prediction errors.

    Args:
        predictions: Predicted values
        targets: Ground truth values
        task_name: Name of the task
        bins: Number of bins for histogram
        save_path: Path to save figure (if None, returns figure)

    Returns:
        matplotlib figure
    """
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Compute errors
    errors = predictions - targets

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogram
    ax.hist(errors, bins=bins, alpha=0.7, edgecolor='black')

    # Add vertical line at zero
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')

    # Labels
    ax.set_xlabel('Prediction Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{task_name} Error Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    text = f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}'
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig


def plot_metrics_comparison(results_dict, save_path=None):
    """Plot comparison of metrics across all tasks.

    Args:
        results_dict: Dictionary of results from evaluate_probes
        save_path: Path to save figure (if None, returns figure)

    Returns:
        matplotlib figure
    """
    regression_tasks = ['rotation', 'scale', 'perspective_x', 'perspective_y']

    # Extract R² scores
    r2_scores = []
    task_labels = []

    for task in regression_tasks:
        if task in results_dict and 'r2' in results_dict[task]:
            r2_scores.append(results_dict[task]['r2'])
            task_labels.append(task.replace('_', ' ').title())

    # Add grid size accuracy
    if 'grid_size' in results_dict and 'accuracy' in results_dict['grid_size']:
        r2_scores.append(results_dict['grid_size']['accuracy'])
        task_labels.append('Grid Size\n(Accuracy)')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot
    colors = ['#1f77b4'] * len(regression_tasks) + ['#ff7f0e']
    bars = ax.bar(task_labels, r2_scores, color=colors[:len(r2_scores)], alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    # Reference line at 0.8
    ax.axhline(0.8, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (0.8)')

    # Labels
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Downstream Task Performance', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    else:
        return fig


def create_all_visualizations(predictions_dict, results_dict, save_dir=None):
    """Create all visualization plots.

    Args:
        predictions_dict: Dictionary with predictions and targets
        results_dict: Dictionary of evaluation results
        save_dir: Directory to save figures (if None, returns figures dict)

    Returns:
        Dictionary of figures (if save_dir is None)
    """
    # Create save directory if needed
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    figures = {}

    # Regression scatter plots
    regression_figs = plot_all_regression_tasks(predictions_dict, save_dir)
    if regression_figs:
        figures.update(regression_figs)

    # Confusion matrix for grid_size
    if 'grid_size' in predictions_dict:
        save_path = f"{save_dir}/grid_size_confusion.png" if save_dir else None
        fig = plot_confusion_matrix(
            predictions_dict['grid_size']['predictions'],
            predictions_dict['grid_size']['targets'],
            save_path=save_path
        )
        if fig is not None:
            figures['grid_size_confusion'] = fig

    # Metrics comparison
    save_path = f"{save_dir}/metrics_comparison.png" if save_dir else None
    fig = plot_metrics_comparison(results_dict, save_path)
    if fig is not None:
        figures['metrics_comparison'] = fig

    return figures if not save_dir else None
