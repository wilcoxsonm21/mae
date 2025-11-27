"""Investigate why mean_intensity is unlearnable.

The hypothesis: After normalization, ALL images have mean ≈ 0,
so "mean_intensity" becomes meaningless!
"""

import numpy as np
from data import get_dataset

print("Investigating the mean_intensity paradox...")
print("="*60)

# Generate dataset
dataset = get_dataset(
    dataset_name='checkerboard',
    n_samples=1000,
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
train_params = dataset['train_params']

# Get the actual images
train_images = train_loader.dataset.tensors[0].numpy()
print(f"\nDataset info:")
print(f"  Shape: {train_images.shape}")
print(f"  Total samples: {len(train_images)}")

# Check what normalization did
print(f"\n" + "="*60)
print("Normalized Image Statistics")
print("="*60)
print(f"Global mean across all pixels: {train_images.mean():.10f}")
print(f"Global std across all pixels: {train_images.std():.10f}")

# Check per-image means
per_image_means = train_images.mean(axis=1)
print(f"\nPer-image means:")
print(f"  Mean of means: {per_image_means.mean():.10f}")
print(f"  Std of means: {per_image_means.std():.10f}")
print(f"  Min: {per_image_means.min():.10f}")
print(f"  Max: {per_image_means.max():.10f}")
print(f"  Range: {per_image_means.max() - per_image_means.min():.10f}")

# Check stored mean_intensity
stored_mean_intensity = train_params['mean_intensity']
print(f"\nStored mean_intensity:")
print(f"  Mean: {stored_mean_intensity.mean():.10f}")
print(f"  Std: {stored_mean_intensity.std():.10f}")
print(f"  Min: {stored_mean_intensity.min():.10f}")
print(f"  Max: {stored_mean_intensity.max():.10f}")
print(f"  Range: {stored_mean_intensity.max() - stored_mean_intensity.min():.10f}")

# Verify they match
max_diff = np.abs(per_image_means - stored_mean_intensity).max()
print(f"\nDifference between computed and stored: {max_diff:.15f}")

# The key question
print(f"\n" + "="*60)
print("THE KEY INSIGHT")
print("="*60)

# What was the normalization applied?
print(f"\nNormalization parameters:")
print(f"  Dataset mean: {dataset.get('mean', 'N/A')}")
print(f"  Dataset std: {dataset.get('std', 'N/A')}")

# Generate UN-normalized dataset to compare
dataset_unnorm = get_dataset(
    dataset_name='checkerboard',
    n_samples=1000,
    image_size=32,
    train_split=0.8,
    batch_size=128,
    normalize=False,  # No normalization!
    random_state=42,
    grid_sizes=[2, 4, 8, 16],
    apply_transforms=True,
    rotation_range=15.0,
    scale_range=(0.8, 1.2),
    perspective_range=0.2,
    return_params=True
)

unnorm_images = dataset_unnorm['train_loader'].dataset.tensors[0].numpy()
unnorm_means = unnorm_images.mean(axis=1)

print(f"\n" + "="*60)
print("UN-normalized Image Statistics")
print("="*60)
print(f"Global mean: {unnorm_images.mean():.10f}")
print(f"Global std: {unnorm_images.std():.10f}")

print(f"\nPer-image means (unnormalized):")
print(f"  Mean: {unnorm_means.mean():.10f}")
print(f"  Std: {unnorm_means.std():.10f}")
print(f"  Min: {unnorm_means.min():.10f}")
print(f"  Max: {unnorm_means.max():.10f}")
print(f"  Range: {unnorm_means.max() - unnorm_means.min():.10f}")

# Compare variances
print(f"\n" + "="*60)
print("Variance Comparison")
print("="*60)
print(f"Normalized per-image mean variance: {per_image_means.var():.10f}")
print(f"UN-normalized per-image mean variance: {unnorm_means.var():.10f}")
print(f"Ratio: {per_image_means.var() / unnorm_means.var():.4f}")

# The smoking gun
print(f"\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

if per_image_means.std() < 0.02:
    print("❌ PROBLEM IDENTIFIED!")
    print("\n   After normalization, all images have mean ≈ 0")
    print("   The 'mean_intensity' target has almost no variance!")
    print(f"   Variance: {per_image_means.var():.10f}")
    print("\n   Why this happens:")
    print("   - Normalization: x_norm = (x - dataset_mean) / dataset_std")
    print("   - This makes the GLOBAL mean = 0")
    print("   - Individual image means become tiny deviations from 0")
    print("   - These deviations are mostly noise from augmentations")
    print("\n   Why models can't learn it:")
    print("   - Signal-to-noise ratio is too low")
    print("   - The task has become: predict tiny noise deviations")
    print("   - Models rationally predict ≈0 (the mean) to minimize MSE")
    print("\n   Conclusion: mean_intensity is NOT a good test metric!")
    print("   It becomes trivial/meaningless after normalization")
else:
    print("✓ Per-image means have reasonable variance")
    print("   The issue is likely something else")

# Suggest alternatives
print(f"\n" + "="*60)
print("Better Test Metrics")
print("="*60)
print("Instead of mean_intensity, try:")
print("  1. Use UN-normalized data (normalize=False)")
print("  2. Test with std/variance of pixels (preserved under normalization)")
print("  3. Test with rotation/scale (geometric properties)")
print("  4. Test with grid_size (categorical, unaffected by normalization)")
