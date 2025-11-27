"""Check what the supervised baseline is actually predicting."""

import torch
import numpy as np
from data import get_dataset
from train_supervised_baseline import SupervisedEncoder
from evaluation.downstream.probes import LatentProbe
from pathlib import Path

print("Loading trained model...")
print("="*60)

# Check if model exists
model_path = Path("checkpoints/supervised_baseline/best_model_mean_intensity.pt")
if not model_path.exists():
    print(f"Model not found at {model_path}")
    print("Please train the model first:")
    print("  python train_supervised_baseline.py --config configs/supervised_baseline.yaml --task mean_intensity")
    exit(1)

# Load checkpoint
checkpoint = torch.load(model_path, weights_only=False)
print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"Val loss: {checkpoint['val_loss']:.6f}")

# Generate dataset
dataset = get_dataset(
    dataset_name='checkerboard',
    n_samples=10000,
    image_size=32,
    train_split=0.8,
    batch_size=128,
    normalize=True,
    random_state=42,
    grid_sizes=[2, 4, 8, 16],
    apply_transforms=True,
    rotation_range=15.0,
    scale_range=(0.8, 1.2),
    perspective_range=0.2,
    return_params=True
)

val_loader = dataset['val_loader']
val_params = dataset['val_params']

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = SupervisedEncoder(
    image_size=32,
    base_channels=64,
    latent_dim=64,
    dropout=0.0
).to(device)

probe = LatentProbe(
    latent_dim=64,
    hidden_dim=64,
    output_dim=1,
    task_type='regression',
    dropout=0.1
).to(device)

# Load weights from the full model state dict
full_state_dict = checkpoint['model_state_dict']

# Extract encoder and probe weights
encoder_state = {k.replace('encoder.', ''): v for k, v in full_state_dict.items() if k.startswith('encoder.')}
probe_state = {k.replace('probe.', ''): v for k, v in full_state_dict.items() if k.startswith('probe.')}

encoder.load_state_dict(encoder_state)
probe.load_state_dict(probe_state)

encoder.eval()
probe.eval()

# Get predictions
print("\nGetting predictions...")
all_predictions = []
all_targets = []

with torch.no_grad():
    for batch_idx, (data, _) in enumerate(val_loader):
        data = data.to(device)
        z = encoder(data)
        predictions = probe(z)
        all_predictions.append(predictions.cpu().numpy())

        # Get targets
        batch_size = data.size(0)
        batch_start = batch_idx * val_loader.batch_size
        batch_end = batch_start + batch_size
        targets = val_params['mean_intensity'][batch_start:batch_end]
        all_targets.append(targets)

predictions = np.concatenate(all_predictions).flatten()
targets = np.concatenate(all_targets).flatten()

print(f"\n" + "="*60)
print("Prediction Analysis")
print("="*60)

print(f"\nTargets:")
print(f"  Mean: {targets.mean():.6f}")
print(f"  Std: {targets.std():.6f}")
print(f"  Min: {targets.min():.6f}")
print(f"  Max: {targets.max():.6f}")
print(f"  Range: {targets.max() - targets.min():.6f}")

print(f"\nPredictions:")
print(f"  Mean: {predictions.mean():.6f}")
print(f"  Std: {predictions.std():.6f}")
print(f"  Min: {predictions.min():.6f}")
print(f"  Max: {predictions.max():.6f}")
print(f"  Range: {predictions.max() - predictions.min():.6f}")

print(f"\nMetrics:")
mse = np.mean((predictions - targets)**2)
mae = np.mean(np.abs(predictions - targets))
correlation = np.corrcoef(predictions, targets)[0, 1]
r2 = 1 - (mse / targets.var())
print(f"  MSE: {mse:.6f}")
print(f"  MAE: {mae:.6f}")
print(f"  R²: {r2:.4f}")
print(f"  Correlation: {correlation:.4f}")

print(f"\n" + "="*60)
print("Diagnosis")
print("="*60)

# Check if predicting constant
if predictions.std() < 0.001:
    print("❌ Model is predicting nearly constant values!")
    print(f"   Prediction std ({predictions.std():.6f}) is very low")
elif predictions.std() < targets.std() * 0.5:
    print("⚠️  Model predictions have lower variance than targets")
    print(f"   Prediction std: {predictions.std():.6f}")
    print(f"   Target std: {targets.std():.6f}")
    print(f"   Ratio: {predictions.std() / targets.std():.2f}")
else:
    print("✓ Model is producing varied predictions")

if correlation > 0.9:
    print("✓ Very high correlation - model is learning well!")
elif correlation > 0.7:
    print("✓ Good correlation - model is learning")
elif correlation > 0.3:
    print("⚠️  Moderate correlation - model is learning somewhat")
else:
    print("❌ Low correlation - model may not be learning effectively")

# Show some example predictions
print(f"\n" + "="*60)
print("Sample Predictions (first 10)")
print("="*60)
print("Target     Prediction  Error")
print("-" * 40)
for i in range(min(10, len(targets))):
    error = predictions[i] - targets[i]
    print(f"{targets[i]:+.6f}  {predictions[i]:+.6f}  {error:+.6f}")
