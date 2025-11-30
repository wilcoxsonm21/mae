"""Extract and plot results from mask ratio experiments."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import json

def load_checkpoint_results(checkpoint_dir):
    """Load results from a checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Dictionary with train_loss, val_loss, downstream_results
    """
    checkpoint_path = Path(checkpoint_dir) / 'best_model.pt'

    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return None

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    results = {
        'train_loss': checkpoint.get('train_loss', None),
        'train_loss_eval_mode': checkpoint.get('train_loss_eval_mode', None),
        'val_loss': checkpoint.get('val_loss', None),
        'mask_ratio': checkpoint['config']['model']['params']['mask_ratio']
    }

    # Try to load downstream results
    downstream_dir = Path(checkpoint_dir) / 'downstream_eval'
    downstream_results_path = downstream_dir / 'results.json'

    if downstream_results_path.exists():
        with open(downstream_results_path, 'r') as f:
            results['downstream'] = json.load(f)
    else:
        results['downstream'] = None

    return results


def extract_all_results(base_dir='checkpoints/mask_ratio_study'):
    """Extract results from all mask ratio experiments.

    Args:
        base_dir: Base directory containing all experiment checkpoints

    Returns:
        Dictionary with mask ratios as keys and results as values
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Base directory {base_dir} not found")
        return {}

    all_results = {}

    # Find all mask_* directories
    mask_dirs = sorted(base_path.glob('mask_*'))

    for mask_dir in mask_dirs:
        mask_ratio_pct = int(mask_dir.name.replace('mask_', ''))
        mask_ratio = mask_ratio_pct / 100.0

        print(f"Loading results for mask ratio {mask_ratio:.1f}...")

        results = load_checkpoint_results(mask_dir)
        if results is not None:
            all_results[mask_ratio] = results

    return all_results


def plot_results(all_results, save_dir='experiments/mask_ratio_study'):
    """Plot all metrics as a function of mask ratio.

    Args:
        all_results: Dictionary of results from extract_all_results
        save_dir: Directory to save plots
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    # Sort by mask ratio
    mask_ratios = sorted(all_results.keys())

    # Extract metrics
    train_losses = [all_results[mr]['train_loss'] for mr in mask_ratios
                    if all_results[mr]['train_loss'] is not None]
    val_losses = [all_results[mr]['val_loss'] for mr in mask_ratios
                  if all_results[mr]['val_loss'] is not None]

    # Extract downstream metrics (if available)
    composite_scores = []
    rotation_corr = []
    scale_corr = []
    perspective_x_corr = []
    perspective_y_corr = []
    grid_accuracy = []

    for mr in mask_ratios:
        downstream = all_results[mr].get('downstream')
        if downstream:
            composite_scores.append(downstream.get('composite_score', 0))
            rotation_corr.append(downstream.get('rotation', {}).get('correlation', 0))
            scale_corr.append(downstream.get('scale', {}).get('correlation', 0))
            perspective_x_corr.append(downstream.get('perspective_x', {}).get('correlation', 0))
            perspective_y_corr.append(downstream.get('perspective_y', {}).get('correlation', 0))
            grid_accuracy.append(downstream.get('grid_size', {}).get('accuracy', 0))
        else:
            composite_scores.append(None)
            rotation_corr.append(None)
            scale_corr.append(None)
            perspective_x_corr.append(None)
            perspective_y_corr.append(None)
            grid_accuracy.append(None)

    # Convert to percentages for x-axis
    mask_ratios_pct = [mr * 100 for mr in mask_ratios]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('CNN MAE Performance vs Mask Ratio', fontsize=16, fontweight='bold')

    # Plot 1: Train and Validation Loss
    ax = axes[0, 0]
    if train_losses:
        ax.plot(mask_ratios_pct[:len(train_losses)], train_losses, 'o-', linewidth=2, markersize=8, label='Train Loss', color='blue')
    if val_losses:
        ax.plot(mask_ratios_pct[:len(val_losses)], val_losses, 's-', linewidth=2, markersize=8, label='Val Loss', color='red')
    ax.set_xlabel('Mask Ratio (%)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Composite Score
    ax = axes[0, 1]
    valid_composite = [(mr, cs) for mr, cs in zip(mask_ratios_pct, composite_scores) if cs is not None]
    if valid_composite:
        mrs, css = zip(*valid_composite)
        ax.plot(mrs, css, 'o-', linewidth=2, markersize=8, color='green')
        ax.set_xlabel('Mask Ratio (%)', fontsize=12)
        ax.set_ylabel('Composite Score', fontsize=12)
        ax.set_title('Aggregate Downstream Performance', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Plot 3: Grid Size Accuracy
    ax = axes[0, 2]
    valid_grid = [(mr, ga) for mr, ga in zip(mask_ratios_pct, grid_accuracy) if ga is not None]
    if valid_grid:
        mrs, gas = zip(*valid_grid)
        ax.plot(mrs, gas, 'o-', linewidth=2, markersize=8, color='purple')
        ax.set_xlabel('Mask Ratio (%)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Grid Size Classification Accuracy', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Plot 4: Rotation Correlation
    ax = axes[1, 0]
    valid_rot = [(mr, rc) for mr, rc in zip(mask_ratios_pct, rotation_corr) if rc is not None]
    if valid_rot:
        mrs, rcs = zip(*valid_rot)
        ax.plot(mrs, rcs, 'o-', linewidth=2, markersize=8, color='red')
        ax.set_xlabel('Mask Ratio (%)', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.set_title('Rotation Prediction Correlation', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Plot 5: Scale Correlation
    ax = axes[1, 1]
    valid_scale = [(mr, sc) for mr, sc in zip(mask_ratios_pct, scale_corr) if sc is not None]
    if valid_scale:
        mrs, scs = zip(*valid_scale)
        ax.plot(mrs, scs, 'o-', linewidth=2, markersize=8, color='blue')
        ax.set_xlabel('Mask Ratio (%)', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=12)
        ax.set_title('Scale Prediction Correlation', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Plot 6: Perspective Correlations
    ax = axes[1, 2]
    valid_px = [(mr, px) for mr, px in zip(mask_ratios_pct, perspective_x_corr) if px is not None]
    valid_py = [(mr, py) for mr, py in zip(mask_ratios_pct, perspective_y_corr) if py is not None]
    if valid_px:
        mrs, pxs = zip(*valid_px)
        ax.plot(mrs, pxs, 'o-', linewidth=2, markersize=8, color='orange', label='Perspective X')
    if valid_py:
        mrs, pys = zip(*valid_py)
        ax.plot(mrs, pys, 's-', linewidth=2, markersize=8, color='cyan', label='Perspective Y')
    ax.set_xlabel('Mask Ratio (%)', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Perspective Prediction Correlations', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = save_path / 'mask_ratio_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Also save a summary CSV
    csv_path = save_path / 'mask_ratio_results.csv'
    with open(csv_path, 'w') as f:
        # Write header
        f.write('mask_ratio,train_loss,val_loss,composite_score,grid_accuracy,')
        f.write('rotation_corr,scale_corr,perspective_x_corr,perspective_y_corr\n')

        # Write data
        for i, mr in enumerate(mask_ratios):
            mr_pct = mr * 100
            train_loss = all_results[mr]['train_loss']
            val_loss = all_results[mr]['val_loss']
            cs = composite_scores[i] if composite_scores[i] is not None else ''
            ga = grid_accuracy[i] if grid_accuracy[i] is not None else ''
            rc = rotation_corr[i] if rotation_corr[i] is not None else ''
            sc = scale_corr[i] if scale_corr[i] is not None else ''
            px = perspective_x_corr[i] if perspective_x_corr[i] is not None else ''
            py = perspective_y_corr[i] if perspective_y_corr[i] is not None else ''

            f.write(f'{mr_pct},{train_loss},{val_loss},{cs},{ga},{rc},{sc},{px},{py}\n')

    print(f"Results CSV saved to: {csv_path}")

    return fig


def main():
    """Main function to extract and plot results."""
    print("=" * 60)
    print("Extracting Results from Mask Ratio Experiments")
    print("=" * 60)

    # Extract results
    all_results = extract_all_results()

    if not all_results:
        print("\nNo results found. Make sure experiments have been run.")
        return

    print(f"\nFound results for {len(all_results)} experiments")

    # Check if downstream results are available
    has_downstream = any(r.get('downstream') is not None for r in all_results.values())

    if not has_downstream:
        print("\nWarning: No downstream evaluation results found.")
        print("Downstream metrics will not be plotted.")

    # Plot results
    print("\nCreating plots...")
    plot_results(all_results)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
