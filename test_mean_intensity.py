"""Quick test to verify mean_intensity metric is working."""

import torch
import numpy as np
from data import get_dataset
from evaluation.downstream import ProbeTrainer
from models import UNetAE

print("Testing mean_intensity metric...")
print("="*60)

# Generate small test dataset
print("\nGenerating dataset with mean_intensity metric...")
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

print(f"\nAvailable parameters: {list(train_params.keys())}")
print(f"Mean intensity range (train): [{train_params['mean_intensity'].min():.4f}, {train_params['mean_intensity'].max():.4f}]")
print(f"Mean intensity std (train): {train_params['mean_intensity'].std():.4f}")

# Verify mean_intensity is actually the mean of the images
print("\nVerifying mean_intensity is correct...")
# Get the actual train tensor (not shuffled)
train_tensor = dataset['train_loader'].dataset.tensors[0]
computed_means = train_tensor[:100].mean(dim=1).numpy()
stored_means = train_params['mean_intensity'][:100]
max_diff = np.abs(computed_means - stored_means).max()
print(f"Max difference between computed and stored mean_intensity: {max_diff:.6f}")

if max_diff < 0.001:
    print("✓ mean_intensity metric is correctly computed!")
else:
    print("✗ WARNING: mean_intensity doesn't match computed values!")

# Test with probe trainer
print("\nTesting with ProbeTrainer...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNetAE(
    input_dim=1024,
    latent_dim=64,
    base_channels=32,
    dropout=0.0,
    image_size=32
).to(device)

probe_trainer = ProbeTrainer(model, device=device)

print(f"Regression tasks recognized: {probe_trainer.regression_tasks}")

if 'mean_intensity' in probe_trainer.regression_tasks:
    print("✓ mean_intensity is recognized as a regression task!")
else:
    print("✗ WARNING: mean_intensity not in regression tasks!")

print("\n" + "="*60)
print("Test complete!")
print("="*60)
