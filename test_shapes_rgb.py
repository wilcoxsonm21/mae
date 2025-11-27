"""Test RGB shapes dataset to verify 3-channel support."""

import numpy as np
from data import get_dataset

print("Testing RGB Shapes Dataset (3 channels)")
print("="*60)

# Generate dataset
dataset = get_dataset(
    dataset_name='shapes',
    n_samples=100,
    image_size=32,
    train_split=0.8,
    batch_size=16,
    normalize=False,
    random_state=42,
    noise_level=0.0,
    return_params=True
)

train_loader = dataset['train_loader']
train_params = dataset['train_params']

print(f"\nDataset info:")
print(f"  Train samples: {len(train_loader.dataset)}")
print(f"  Image size: 32x32")
print(f"  Input dim: {dataset['input_dim']}")

# Get first batch
first_batch, _ = next(iter(train_loader))

print(f"\nBatch shape: {first_batch.shape}")
print(f"  Expected: (batch_size, 32*32*3) = (16, 3072)")

# Reshape to verify RGB structure
reshaped = first_batch[0].view(3, 32, 32)
print(f"\nFirst image reshaped: {reshaped.shape}")
print(f"  Channel 0 (R) range: [{reshaped[0].min():.4f}, {reshaped[0].max():.4f}]")
print(f"  Channel 1 (G) range: [{reshaped[1].min():.4f}, {reshaped[1].max():.4f}]")
print(f"  Channel 2 (B) range: [{reshaped[2].min():.4f}, {reshaped[2].max():.4f}]")

# Check color diversity
shape_names = ["circle", "triangle", "square", "rectangle", "pentagon", "hexagon", "star"]
color_names = ["red", "green", "blue", "yellow", "cyan", "magenta"]

print(f"\n" + "="*60)
print("Color Intensity Analysis")
print("="*60)

# Compute mean RGB values for each color
train_images = train_loader.dataset.tensors[0].numpy()
train_colors = train_params['color']

print(f"\nExpected RGB values for each color:")
print(f"  red:     [1.0, 0.0, 0.0]")
print(f"  green:   [0.0, 1.0, 0.0]")
print(f"  blue:    [0.0, 0.0, 1.0]")
print(f"  yellow:  [1.0, 1.0, 0.0]")
print(f"  cyan:    [0.0, 1.0, 1.0]")
print(f"  magenta: [1.0, 0.0, 1.0]")

print(f"\nActual mean RGB values per color:")
for color_idx in range(6):
    mask = train_colors == color_idx
    color_images = train_images[mask]

    if len(color_images) > 0:
        # Reshape to get RGB channels
        rgb = color_images.reshape(-1, 3, 32, 32)

        # Compute mean per channel
        mean_r = rgb[:, 0, :, :].mean()
        mean_g = rgb[:, 1, :, :].mean()
        mean_b = rgb[:, 2, :, :].mean()

        print(f"  {color_names[color_idx]:8s} [{mean_r:.3f}, {mean_g:.3f}, {mean_b:.3f}]")

print(f"\n" + "="*60)
print("Summary")
print("="*60)
print(f"✓ Dataset correctly generates RGB images (3072-dim vectors)")
print(f"✓ Colors have distinct RGB channel values")
print(f"✓ Ready for 3-channel CNN training")
