"""Debug metrics for mean_intensity to see why R² is near zero."""

import torch
import numpy as np
from data import get_dataset
from sklearn.metrics import r2_score, mean_squared_error

print("Debugging mean_intensity metrics...")
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

val_params = dataset['val_params']
mean_intensity = val_params['mean_intensity']

print(f"\nTarget (mean_intensity) statistics:")
print(f"  Mean: {mean_intensity.mean():.6f}")
print(f"  Std: {mean_intensity.std():.6f}")
print(f"  Min: {mean_intensity.min():.6f}")
print(f"  Max: {mean_intensity.max():.6f}")
print(f"  Range: {mean_intensity.max() - mean_intensity.min():.6f}")

# Simulate perfect predictions
perfect_predictions = mean_intensity.copy()
print(f"\n--- Perfect Predictions ---")
r2_perfect = r2_score(mean_intensity, perfect_predictions)
mse_perfect = mean_squared_error(mean_intensity, perfect_predictions)
corr_perfect = np.corrcoef(perfect_predictions, mean_intensity)[0, 1]
print(f"R²: {r2_perfect:.4f}")
print(f"MSE: {mse_perfect:.6f}")
print(f"Correlation: {corr_perfect:.4f}")

# Simulate constant predictions (mean)
constant_predictions = np.full_like(mean_intensity, mean_intensity.mean())
print(f"\n--- Constant Predictions (mean baseline) ---")
r2_constant = r2_score(mean_intensity, constant_predictions)
mse_constant = mean_squared_error(mean_intensity, constant_predictions)
print(f"R²: {r2_constant:.4f}")
print(f"MSE: {mse_constant:.6f}")
print(f"Note: This is what R²=0 looks like")

# Simulate slightly noisy predictions
noisy_predictions = mean_intensity + np.random.randn(len(mean_intensity)) * 0.001
print(f"\n--- Noisy Predictions (noise std=0.001) ---")
r2_noisy = r2_score(mean_intensity, noisy_predictions)
mse_noisy = mean_squared_error(mean_intensity, noisy_predictions)
corr_noisy = np.corrcoef(noisy_predictions, mean_intensity)[0, 1]
print(f"R²: {r2_noisy:.4f}")
print(f"MSE: {mse_noisy:.6f}")
print(f"Correlation: {corr_noisy:.4f}")

# Simulate model with MSE=0.0001 (what we saw in training)
# Generate predictions that give MSE ≈ 0.0001
noise_std = np.sqrt(0.0001)
model_predictions = mean_intensity + np.random.randn(len(mean_intensity)) * noise_std
print(f"\n--- Model Predictions (MSE ≈ 0.0001) ---")
r2_model = r2_score(mean_intensity, model_predictions)
mse_model = mean_squared_error(mean_intensity, model_predictions)
corr_model = np.corrcoef(model_predictions, mean_intensity)[0, 1]
print(f"R²: {r2_model:.4f}")
print(f"MSE: {mse_model:.6f}")
print(f"Correlation: {corr_model:.4f}")

print(f"\n" + "="*60)
print("Analysis:")
print("="*60)
print(f"Target variance: {mean_intensity.var():.6f}")
print(f"If MSE = {mse_model:.6f}")
print(f"Then R² = 1 - (MSE / target_var) = 1 - ({mse_model:.6f} / {mean_intensity.var():.6f}) = {1 - mse_model/mean_intensity.var():.4f}")
print(f"\nActual R² calculated: {r2_model:.4f}")
print(f"\n{'✓' if abs(r2_model - (1 - mse_model/mean_intensity.var())) < 0.01 else '✗'} R² calculation is {'correct' if abs(r2_model - (1 - mse_model/mean_intensity.var())) < 0.01 else 'WRONG!'}")

# Check why correlation might be NaN
print(f"\n" + "="*60)
print("Checking for NaN correlation issues:")
print("="*60)
print(f"Predictions std: {model_predictions.std():.6f}")
print(f"Targets std: {mean_intensity.std():.6f}")
if model_predictions.std() < 1e-10 or mean_intensity.std() < 1e-10:
    print("⚠️  WARNING: Very low std can cause numerical issues!")
else:
    print("✓ Standard deviations are reasonable")
