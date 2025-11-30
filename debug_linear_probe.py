"""Debug linear probe predictions to understand why R² is so negative."""

import torch
import numpy as np
from pathlib import Path

from models import UNetMAE
from data import get_dataset
from evaluation.downstream import ProbeTrainer

# Load UNet MAE model
checkpoint_path = Path('checkpoints/mae_checkerboard/best_model.pt')
checkpoint = torch.load(checkpoint_path, map_location='cuda')
config = checkpoint['config']

# Create model
model_params = config['model']['params'].copy()
model = UNetMAE(**model_params)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda()
model.eval()

# Load dataset
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

print("Loading dataset...")
data = get_dataset(**dataset_config)
train_loader = data['train_loader']
val_loader = data['val_loader']
train_params = data['train_params']
val_params = data['val_params']

# Create probe trainer and train LINEAR probes
print("\nTraining LINEAR probes on UNet MAE (with skip connections)...")
probe_trainer = ProbeTrainer(model, device='cuda')

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
    verbose=True,
    probe_type='linear'  # LINEAR PROBE
)

print("\n" + "="*80)
print("Evaluating probes and analyzing predictions...")
print("="*80)

# Get validation latents and targets
val_latents = probe_trainer.extract_latents(val_loader).cuda()
val_targets_dict = probe_trainer.prepare_targets(val_params)

# Analyze each regression task
for task in ['rotation', 'scale', 'perspective_x', 'perspective_y']:
    if task not in probe_trainer.probes:
        continue

    print(f"\n{'='*80}")
    print(f"Task: {task}")
    print(f"{'='*80}")

    probe = probe_trainer.probes[task]
    probe.eval()

    targets = val_targets_dict[task].cuda()
    with torch.no_grad():
        predictions = probe(val_latents)

    # Convert to numpy
    preds_np = predictions.cpu().numpy().flatten()
    targets_np = targets.cpu().numpy().flatten()

    # Compute statistics
    print(f"\nTarget statistics:")
    print(f"  Mean: {targets_np.mean():.4f}")
    print(f"  Std:  {targets_np.std():.4f}")
    print(f"  Min:  {targets_np.min():.4f}")
    print(f"  Max:  {targets_np.max():.4f}")

    print(f"\nPrediction statistics:")
    print(f"  Mean: {preds_np.mean():.4f}")
    print(f"  Std:  {preds_np.std():.4f}")
    print(f"  Min:  {preds_np.min():.4f}")
    print(f"  Max:  {preds_np.max():.4f}")

    # Compute R²
    from sklearn.metrics import r2_score, mean_squared_error
    mse = mean_squared_error(targets_np, preds_np)
    r2 = r2_score(targets_np, preds_np)

    # Compute baseline (always predict mean)
    baseline_pred = np.full_like(targets_np, targets_np.mean())
    baseline_mse = mean_squared_error(targets_np, baseline_pred)

    print(f"\nMetrics:")
    print(f"  MSE (model):    {mse:.6f}")
    print(f"  MSE (baseline): {baseline_mse:.6f}")
    print(f"  R²:             {r2:.6f}")
    print(f"  MSE ratio:      {mse/baseline_mse:.2f}x baseline")

    # Show sample predictions
    print(f"\nSample predictions (first 10):")
    print(f"  Target: {targets_np[:10]}")
    print(f"  Pred:   {preds_np[:10]}")
    print(f"  Error:  {(preds_np[:10] - targets_np[:10])}")

    # Check if predictions are constant
    if preds_np.std() < 1e-6:
        print(f"\n⚠️  WARNING: Predictions are essentially constant!")
        print(f"    This means the probe learned to output a fixed value.")
        print(f"    The linear probe completely failed to learn this task.")

    # Check if predictions are in wrong range
    target_range = targets_np.max() - targets_np.min()
    pred_range = preds_np.max() - preds_np.min()
    if abs(target_range - pred_range) > target_range:
        print(f"\n⚠️  WARNING: Prediction range differs greatly from target range!")
        print(f"    Target range: {target_range:.4f}")
        print(f"    Pred range:   {pred_range:.4f}")
