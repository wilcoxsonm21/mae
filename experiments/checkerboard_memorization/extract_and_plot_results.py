"""Extract and plot results from checkerboard_memorization experiments."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import json

def load_checkpoint_results(checkpoint_dir):
    """Load results from a checkpoint directory."""
    checkpoint_path = Path(checkpoint_dir) / 'best_model.pt'

    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    results = {
        'train_loss': checkpoint.get('train_loss', None),
        'train_loss_eval_mode': checkpoint.get('train_loss_eval_mode', None),
        'val_loss': checkpoint.get('val_loss', None),
        'config': checkpoint.get('config', {})
    }

    # Load downstream results
    downstream_dir = Path(checkpoint_dir) / 'downstream_eval'
    downstream_results_path = downstream_dir / 'results.json'

    if downstream_results_path.exists():
        with open(downstream_results_path, 'r') as f:
            results['downstream'] = json.load(f)
    else:
        results['downstream'] = None

    return results


def extract_all_results(base_dir='checkpoints/checkerboard_memorization'):
    """Extract results from all experiments."""
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Base directory {base_dir} not found")
        return {}

    all_results = {}

    # Look for transformer_ae and transformer_mae
    for model_dir in ['transformer_ae', 'transformer_mae']:
        model_path = base_path / model_dir
        if model_path.exists():
            print(f"Loading results for {model_dir}...")
            results = load_checkpoint_results(model_path)
            if results is not None:
                all_results[model_dir] = results

    return all_results


def plot_comparison(all_results, save_dir='experiments/checkerboard_memorization'):
    """Plot comparison between AE and MAE."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    models = list(all_results.keys())
    if len(models) == 0:
        print("No results to plot")
        return

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Transformer AE vs MAE on Checkerboard Memorization\n(16 grid classes, 2000 samples)',
                 fontsize=14, fontweight='bold')

    colors = {'transformer_ae': 'blue', 'transformer_mae': 'orange'}
    labels = {'transformer_ae': 'Transformer AE', 'transformer_mae': 'Transformer MAE'}

    # Plot 1: Reconstruction Loss
    ax = axes[0, 0]
    for model in models:
        train_loss = all_results[model].get('train_loss', 0)
        val_loss = all_results[model].get('val_loss', 0)
        x = np.arange(2)
        width = 0.35
        offset = -width/2 if model == 'transformer_ae' else width/2
        ax.bar(x + offset, [train_loss, val_loss], width,
               label=labels.get(model, model), color=colors.get(model, 'gray'))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Train Loss', 'Val Loss'])
    ax.set_ylabel('Loss')
    ax.set_title('Reconstruction Loss')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Grid Size Classification Accuracy
    ax = axes[0, 1]
    accuracies = []
    model_names = []
    for model in models:
        downstream = all_results[model].get('downstream', {})
        if downstream and 'grid_size' in downstream:
            acc = downstream['grid_size'].get('accuracy', 0)
            accuracies.append(acc)
            model_names.append(labels.get(model, model))
    if accuracies:
        bars = ax.bar(model_names, accuracies, color=[colors.get(m, 'gray') for m in models])
        ax.set_ylabel('Accuracy')
        ax.set_title('Grid Size Classification (16 classes)')
        ax.set_ylim(0, 1)
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        ax.axhline(y=1/16, color='red', linestyle='--', label='Random chance (6.25%)')
        ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Composite Score
    ax = axes[1, 0]
    scores = []
    model_names = []
    for model in models:
        downstream = all_results[model].get('downstream', {})
        if downstream:
            score = downstream.get('composite_score', 0)
            scores.append(score)
            model_names.append(labels.get(model, model))
    if scores:
        bars = ax.bar(model_names, scores, color=[colors.get(m, 'gray') for m in models])
        ax.set_ylabel('Composite Score')
        ax.set_title('Downstream Composite Score')
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Per-class F1 scores (if available)
    ax = axes[1, 1]
    for model in models:
        downstream = all_results[model].get('downstream', {})
        if downstream and 'grid_size' in downstream:
            f1_per_class = downstream['grid_size'].get('f1_per_class', [])
            if f1_per_class:
                # Grid sizes: 2, 4, 6, ..., 32
                grid_sizes = list(range(2, 34, 2))[:len(f1_per_class)]
                ax.plot(grid_sizes, f1_per_class, 'o-',
                       label=labels.get(model, model), color=colors.get(model, 'gray'))
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score by Grid Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = save_path / 'comparison_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Save CSV
    csv_path = save_path / 'results.csv'
    with open(csv_path, 'w') as f:
        f.write('model,train_loss,val_loss,grid_accuracy,f1_macro,composite_score\n')
        for model in models:
            train_loss = all_results[model].get('train_loss', '')
            val_loss = all_results[model].get('val_loss', '')
            downstream = all_results[model].get('downstream', {})
            if downstream:
                grid_acc = downstream.get('grid_size', {}).get('accuracy', '')
                f1_macro = downstream.get('grid_size', {}).get('f1_macro', '')
                composite = downstream.get('composite_score', '')
            else:
                grid_acc = f1_macro = composite = ''
            f.write(f'{model},{train_loss},{val_loss},{grid_acc},{f1_macro},{composite}\n')

    print(f"Results CSV saved to: {csv_path}")

    return fig


def main():
    """Main function."""
    print("=" * 60)
    print("Extracting Results from Checkerboard Memorization Experiments")
    print("=" * 60)

    all_results = extract_all_results()

    if not all_results:
        print("\nNo results found. Make sure experiments have been run.")
        return

    print(f"\nFound results for {len(all_results)} experiments")

    # Plot results
    print("\nCreating comparison plots...")
    plot_comparison(all_results)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
