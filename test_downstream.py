"""Test script for downstream evaluation."""

import torch
from data.generators import get_dataset
from models import UNetAE
from evaluation.downstream import ProbeTrainer
from evaluation.downstream.visualize import create_all_visualizations

def test_downstream_evaluation():
    """Test the downstream evaluation pipeline."""
    print("Testing Downstream Evaluation Pipeline")
    print("=" * 60)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate small test dataset
    print("\nGenerating test dataset...")
    dataset = get_dataset(
        dataset_name='checkerboard',
        n_samples=500,
        image_size=32,
        train_split=0.8,
        batch_size=64,
        normalize=True,
        random_state=42,
        grid_sizes=[2, 4, 8, 16],
        apply_transforms=True,
        rotation_range=15.0,
        scale_range=(0.8, 1.2),
        perspective_range=0.2,
        return_params=True
    )

    train_loader = dataset['train_loader']
    val_loader = dataset['val_loader']
    train_params = dataset['train_params']
    val_params = dataset['val_params']

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Parameters: {list(train_params.keys())}")

    # Create a simple model
    print("\nCreating model...")
    model = UNetAE(
        input_dim=1024,
        latent_dim=64,
        base_channels=32,
        dropout=0.0,
        image_size=32
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create probe trainer
    print("\nCreating probe trainer...")
    probe_trainer = ProbeTrainer(model, device=device)

    # Train probes
    print("\nTraining probes (reduced epochs for testing)...")
    probe_trainer.train_probes(
        train_loader=train_loader,
        val_loader=val_loader,
        train_params=train_params,
        val_params=val_params,
        hidden_dim=32,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=20,  # Reduced for testing
        batch_size=64,
        patience=5,
        verbose=True
    )

    # Evaluate
    print("\nEvaluating probes...")
    results = probe_trainer.evaluate_probes(val_loader, val_params, verbose=True)

    # Get predictions
    print("\nGetting predictions for visualization...")
    predictions_dict = probe_trainer.get_predictions(val_loader, val_params)

    # Create visualizations
    print("\nCreating visualizations...")
    figures = create_all_visualizations(predictions_dict, results, save_dir='./test_downstream_viz')

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("Visualizations saved to: ./test_downstream_viz/")
    print("=" * 60)

    return results


if __name__ == '__main__':
    results = test_downstream_evaluation()
