"""Metrics for downstream evaluation."""

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score


def regression_metrics(predictions, targets):
    """Compute regression metrics.

    Args:
        predictions: Predicted values (numpy array or torch tensor)
        targets: Ground truth values (numpy array or torch tensor)

    Returns:
        Dictionary of metrics: mse, mae, r2, correlation
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Flatten to 1D
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Compute metrics
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    # Pearson correlation
    correlation = np.corrcoef(predictions, targets)[0, 1]

    return {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2),
        'correlation': float(correlation)
    }


def classification_metrics(predictions, targets, num_classes=4):
    """Compute classification metrics.

    Args:
        predictions: Predicted class indices or logits (numpy array or torch tensor)
        targets: Ground truth class indices (numpy array or torch tensor)
        num_classes: Number of classes

    Returns:
        Dictionary of metrics: accuracy, f1_macro, f1_per_class
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # If predictions are logits, get argmax
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    # Flatten to 1D
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Compute metrics
    accuracy = accuracy_score(targets, predictions)
    f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)

    # Per-class F1
    f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)

    return {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_per_class': [float(f1) for f1 in f1_per_class]
    }


def grid_size_to_class(grid_sizes, grid_size_options=None):
    """Convert grid sizes to class indices.

    Args:
        grid_sizes: Array of grid sizes
        grid_size_options: List of possible grid sizes. If None, auto-detects from data.

    Returns:
        Array of class indices
    """
    # Convert to numpy if needed
    if isinstance(grid_sizes, torch.Tensor):
        grid_sizes = grid_sizes.detach().cpu().numpy()

    # Auto-detect grid_size_options if not provided
    if grid_size_options is None:
        unique_sizes = sorted(set(int(s) for s in grid_sizes))
        grid_size_options = unique_sizes

    # Create mapping
    size_to_idx = {size: idx for idx, size in enumerate(grid_size_options)}

    # Convert to class indices
    class_indices = np.array([size_to_idx[int(size)] for size in grid_sizes])

    return class_indices


def get_num_grid_classes(grid_sizes):
    """Get the number of unique grid size classes.

    Args:
        grid_sizes: Array of grid sizes

    Returns:
        Number of unique classes
    """
    if isinstance(grid_sizes, torch.Tensor):
        grid_sizes = grid_sizes.detach().cpu().numpy()

    return len(set(int(s) for s in grid_sizes))


def class_to_grid_size(class_indices, grid_size_options=[2, 4, 8, 16]):
    """Convert class indices back to grid sizes.

    Args:
        class_indices: Array of class indices (0, 1, 2, or 3)
        grid_size_options: List of possible grid sizes

    Returns:
        Array of grid sizes (2, 4, 8, or 16)
    """
    # Convert to numpy if needed
    if isinstance(class_indices, torch.Tensor):
        class_indices = class_indices.detach().cpu().numpy()

    return np.array([grid_size_options[int(idx)] for idx in class_indices])


def compute_composite_score(results, weight_r2=0.7, weight_acc=0.3):
    """Compute composite score across all tasks.

    Args:
        results: Dictionary of results from all tasks
        weight_r2: Weight for average R² score
        weight_acc: Weight for grid size accuracy

    Returns:
        Composite score (0 to 1)
    """
    # Average R² across regression tasks
    regression_tasks = ['rotation', 'scale', 'perspective_x', 'perspective_y']
    r2_scores = [results[task]['r2'] for task in regression_tasks if task in results]

    # Clip negative R² values to 0 (negative R² means worse than baseline)
    # R² can be negative when model performs worse than predicting the mean
    # For composite score, we treat negative R² as 0 contribution
    r2_scores_clipped = [max(0.0, r2) for r2 in r2_scores]
    avg_r2 = np.mean(r2_scores_clipped) if r2_scores_clipped else 0.0

    # Grid size accuracy
    grid_accuracy = results.get('grid_size', {}).get('accuracy', 0.0)

    # Composite score
    composite = weight_r2 * avg_r2 + weight_acc * grid_accuracy

    return float(composite)
