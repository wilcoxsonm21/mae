"""Compare downstream performance across different model architectures.

This script loads trained models and evaluates their downstream probe performance
to determine if removing skip connections (CNN models) improves latent representations
compared to UNet models with skip connections.
"""

import torch
import argparse
from pathlib import Path
import yaml
import pandas as pd
from tabulate import tabulate

from models import UNetAE, UNetMAE, CNNAutoencoder, CNNMAE
from data import get_dataset
from evaluation.downstream import ProbeTrainer


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Load a model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model on

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Get model type and params
    model_type = config['model']['type']
    model_params = config['model']['params'].copy()

    # Create model
    if model_type == 'unet_ae':
        model = UNetAE(**model_params)
    elif model_type == 'unet_mae':
        model = UNetMAE(**model_params)
    elif model_type == 'cnn_ae':
        model = CNNAutoencoder(**model_params)
    elif model_type == 'cnn_mae':
        model = CNNMAE(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


def evaluate_model(model, train_loader, val_loader, train_params, val_params,
                   device='cuda', verbose=False, probe_type='mlp'):
    """Run downstream evaluation on a model.

    Args:
        model: Trained model with encode() method
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        train_params: Training generation parameters
        val_params: Validation generation parameters
        device: Device to run on
        verbose: Whether to print progress
        probe_type: 'linear' or 'mlp'

    Returns:
        Dictionary of evaluation results
    """
    probe_trainer = ProbeTrainer(model, device=device)

    # Train probes
    if verbose:
        print(f"Training downstream probes (probe_type={probe_type})...")
    probe_trainer.train_probes(
        train_loader=train_loader,
        val_loader=val_loader,
        train_params=train_params,
        val_params=val_params,
        hidden_dim=64,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=100,
        batch_size=256,
        patience=15,
        verbose=verbose,
        probe_type=probe_type
    )

    # Evaluate probes
    if verbose:
        print("Evaluating downstream probes...")
    results = probe_trainer.evaluate_probes(val_loader, val_params, verbose=verbose)

    return results


def compare_models(model_paths, dataset_config, device='cuda', verbose=True, probe_type='mlp'):
    """Compare downstream performance across multiple models.

    Args:
        model_paths: Dict mapping model names to checkpoint paths
        dataset_config: Dataset configuration dictionary
        device: Device to run on
        verbose: Whether to print progress
        probe_type: 'linear' or 'mlp'

    Returns:
        DataFrame with comparison results
    """
    # Load dataset once (shared across all models)
    print("Loading dataset...")
    data = get_dataset(**dataset_config)
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    train_params = data['train_params']
    val_params = data['val_params']

    # Evaluate each model
    all_results = {}

    for model_name, checkpoint_path in model_paths.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        model, config = load_model_from_checkpoint(checkpoint_path, device=device)

        # Evaluate
        results = evaluate_model(
            model, train_loader, val_loader, train_params, val_params,
            device=device, verbose=verbose, probe_type=probe_type
        )

        all_results[model_name] = results

    # Create comparison table
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}\n")

    # Extract key metrics for comparison
    comparison_data = []

    for model_name, results in all_results.items():
        row = {'Model': model_name}

        # Add composite score
        if 'composite_score' in results:
            row['Composite Score'] = f"{results['composite_score']:.4f}"

        # Add individual task scores
        for task in ['grid_size', 'rotation', 'scale', 'perspective_x', 'perspective_y', 'mean_intensity']:
            if task in results and isinstance(results[task], dict):
                if 'accuracy' in results[task]:
                    row[f'{task}_acc'] = f"{results[task]['accuracy']:.4f}"
                elif 'r2' in results[task]:
                    row[f'{task}_r2'] = f"{results[task]['r2']:.4f}"

        comparison_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(comparison_data)

    # Print table
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

    return df, all_results


def main():
    parser = argparse.ArgumentParser(description='Compare model downstream performance')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--checkpoints-dir', type=str, default='./checkpoints',
                       help='Base directory containing checkpoints')
    parser.add_argument('--probe-type', type=str, default='mlp', choices=['linear', 'mlp'],
                       help='Type of probe to use (linear or mlp)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    checkpoints_dir = Path(args.checkpoints_dir)
    probe_type = args.probe_type

    # Define models to compare
    model_paths = {
        'UNet AE (with skip)': checkpoints_dir / 'ae_checkerboard' / 'best_model.pt',
        'UNet MAE (with skip)': checkpoints_dir / 'mae_checkerboard' / 'best_model.pt',
        'CNN AE (no skip)': checkpoints_dir / 'cnn_ae_checkerboard' / 'best_model.pt',
        'CNN MAE (no skip)': checkpoints_dir / 'cnn_mae_checkerboard' / 'best_model.pt',
    }

    # Check which models exist
    existing_models = {name: path for name, path in model_paths.items() if path.exists()}
    missing_models = {name: path for name, path in model_paths.items() if not path.exists()}

    if missing_models:
        print("Warning: The following models were not found:")
        for name, path in missing_models.items():
            print(f"  - {name}: {path}")
        print()

    if not existing_models:
        print("Error: No model checkpoints found!")
        return

    # Dataset configuration (checkerboard with transforms)
    dataset_config = {
        'dataset_name': 'checkerboard',
        'n_samples': 15000,
        'image_size': 32,
        'train_split': 0.8,
        'batch_size': 256,
        'normalize': True,
        'random_state': 42,
        'grid_sizes': [2, 4, 8, 16],
        'noise_level': 0.01,
        'apply_transforms': True,
        'rotation_range': 15.0,
        'scale_range': [0.8, 1.2],
        'perspective_range': 0.2,
        'return_params': True
    }

    # Run comparison
    print(f"\nProbe type: {probe_type}")
    df, all_results = compare_models(existing_models, dataset_config, device=device, probe_type=probe_type)

    # Save results
    output_path = Path('model_comparison_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Analyze skip connection impact
    print(f"\n{'='*60}")
    print("ANALYSIS: Impact of Skip Connections")
    print(f"{'='*60}\n")

    if 'UNet AE (with skip)' in all_results and 'CNN AE (no skip)' in all_results:
        unet_ae_score = all_results['UNet AE (with skip)'].get('composite_score', 0)
        cnn_ae_score = all_results['CNN AE (no skip)'].get('composite_score', 0)
        ae_improvement = ((cnn_ae_score - unet_ae_score) / unet_ae_score * 100) if unet_ae_score > 0 else 0

        print(f"Autoencoder (AE) Comparison:")
        print(f"  UNet AE (with skip):  {unet_ae_score:.4f}")
        print(f"  CNN AE (no skip):     {cnn_ae_score:.4f}")
        print(f"  Improvement:          {ae_improvement:+.2f}%")
        print()

    if 'UNet MAE (with skip)' in all_results and 'CNN MAE (no skip)' in all_results:
        unet_mae_score = all_results['UNet MAE (with skip)'].get('composite_score', 0)
        cnn_mae_score = all_results['CNN MAE (no skip)'].get('composite_score', 0)
        mae_improvement = ((cnn_mae_score - unet_mae_score) / unet_mae_score * 100) if unet_mae_score > 0 else 0

        print(f"Masked Autoencoder (MAE) Comparison:")
        print(f"  UNet MAE (with skip): {unet_mae_score:.4f}")
        print(f"  CNN MAE (no skip):    {cnn_mae_score:.4f}")
        print(f"  Improvement:          {mae_improvement:+.2f}%")
        print()

    print("Conclusion:")
    if 'CNN AE (no skip)' in all_results or 'CNN MAE (no skip)' in all_results:
        avg_improvement = (ae_improvement + mae_improvement) / 2 if ('UNet AE (with skip)' in all_results and 'CNN AE (no skip)' in all_results and 'UNet MAE (with skip)' in all_results and 'CNN MAE (no skip)' in all_results) else (ae_improvement if 'CNN AE (no skip)' in all_results else mae_improvement)

        if avg_improvement > 5:
            print("  Removing skip connections SIGNIFICANTLY IMPROVED latent representations!")
            print(f"  Average improvement: {avg_improvement:+.2f}%")
        elif avg_improvement > 0:
            print("  Removing skip connections slightly improved latent representations.")
            print(f"  Average improvement: {avg_improvement:+.2f}%")
        else:
            print("  Removing skip connections did not improve latent representations.")
            print(f"  Average change: {avg_improvement:+.2f}%")
            print("  Skip connections may not be the primary issue.")


if __name__ == '__main__':
    main()
