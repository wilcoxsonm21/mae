"""Test if latent codes preserve mean intensity information."""

import torch
import numpy as np
from data import get_dataset
from train_supervised_baseline import SupervisedEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pathlib import Path

print("Testing if latent codes contain mean intensity info...")
print("="*60)

# Generate dataset
dataset = get_dataset(
    dataset_name='checkerboard',
    n_samples=2000,
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

train_loader = dataset['train_loader']
val_loader = dataset['val_loader']
train_params = dataset['train_params']
val_params = dataset['val_params']

print(f"\nDataset info:")
print(f"  Train: {len(train_loader.dataset)}")
print(f"  Val: {len(val_loader.dataset)}")
print(f"  Mean intensity range: [{val_params['mean_intensity'].min():.6f}, {val_params['mean_intensity'].max():.6f}]")
print(f"  Mean intensity std: {val_params['mean_intensity'].std():.6f}")

# Test 1: Can we predict mean intensity directly from pixels?
print(f"\n" + "="*60)
print("Test 1: Direct prediction from raw pixels")
print("="*60)

# Get all data as numpy arrays
train_images = train_loader.dataset.tensors[0].numpy()
val_images = val_loader.dataset.tensors[0].numpy()
train_targets = train_params['mean_intensity']
val_targets = val_params['mean_intensity']

# Simple check: does the mean of the pixels match the stored mean_intensity?
computed_means_train = train_images.mean(axis=1)
max_diff = np.abs(computed_means_train - train_targets).max()
print(f"Max difference between pixel mean and stored mean_intensity: {max_diff:.10f}")
if max_diff < 1e-6:
    print("✓ Mean intensity IS just the mean of the pixels!")
else:
    print("⚠️  Mean intensity doesn't match pixel mean exactly")

# Test 2: Create a random encoder and check if latent preserves mean intensity
print(f"\n" + "="*60)
print("Test 2: Random (untrained) encoder")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_encoder = SupervisedEncoder(
    image_size=32,
    base_channels=64,
    latent_dim=64,
    dropout=0.0
).to(device)

random_encoder.eval()

# Extract latents
train_latents = []
with torch.no_grad():
    for batch, _ in train_loader:
        batch = batch.to(device)
        z = random_encoder(batch)
        train_latents.append(z.cpu().numpy())
train_latents = np.concatenate(train_latents)

val_latents = []
with torch.no_grad():
    for batch, _ in val_loader:
        batch = batch.to(device)
        z = random_encoder(batch)
        val_latents.append(z.cpu().numpy())
val_latents = np.concatenate(val_latents)

print(f"Latent shape: {train_latents.shape}")

# Fit a linear model from latents to mean intensity
linear_model = LinearRegression()
linear_model.fit(train_latents, train_targets)
val_predictions = linear_model.predict(val_latents)

mse = np.mean((val_predictions - val_targets)**2)
r2 = r2_score(val_targets, val_predictions)
correlation = np.corrcoef(val_predictions, val_targets)[0, 1]

print(f"Linear probe on random encoder latents:")
print(f"  MSE: {mse:.6f}")
print(f"  R²: {r2:.4f}")
print(f"  Correlation: {correlation:.4f}")
print(f"  Target variance: {val_targets.var():.6f}")

if r2 > 0.5:
    print("✓ Random encoder preserves mean intensity well!")
elif r2 > 0.1:
    print("⚠️  Random encoder preserves some mean intensity info")
else:
    print("❌ Random encoder loses mean intensity information!")
    print("   This suggests the encoder architecture may not preserve global statistics")

# Test 3: Check if specific latent dimensions correlate with mean intensity
print(f"\n" + "="*60)
print("Test 3: Latent dimension analysis")
print("="*60)

# Find which latent dimensions correlate most with mean intensity
correlations = []
for dim in range(train_latents.shape[1]):
    corr = np.corrcoef(train_latents[:, dim], train_targets)[0, 1]
    correlations.append(corr)

correlations = np.array(correlations)
top_indices = np.argsort(np.abs(correlations))[-5:][::-1]

print(f"Top 5 most correlated latent dimensions:")
for i, idx in enumerate(top_indices, 1):
    print(f"  {i}. Dim {idx}: correlation = {correlations[idx]:.4f}")

max_single_corr = np.max(np.abs(correlations))
print(f"\nMax single-dimension correlation: {max_single_corr:.4f}")

if max_single_corr > 0.5:
    print("✓ Strong correlation in at least one dimension")
elif max_single_corr > 0.2:
    print("⚠️  Moderate correlation, info is somewhat preserved")
else:
    print("❌ Weak correlation across all dimensions")
    print("   Mean intensity info may be distributed across many dimensions")
    print("   or lost during encoding")

print(f"\n" + "="*60)
print("Summary")
print("="*60)
print(f"The issue is likely:")
if r2 < 0.1:
    print("  ❌ The encoder architecture loses global mean information")
    print("  ❌ Convolutional layers focus on local patterns, not global stats")
    print("  💡 Solution: Add global pooling or explicit global statistics to latents")
else:
    print("  ⚠️  Info is preserved but may need a nonlinear probe")
    print("  ⚠️  Or training is getting stuck in local minimum")
