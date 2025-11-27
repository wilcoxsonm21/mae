"""Test the shapes dataset - verify diversity and visualize samples."""

import numpy as np
import matplotlib.pyplot as plt
from data import get_dataset

print("Testing Shapes Dataset")
print("="*60)

# Generate dataset
dataset = get_dataset(
    dataset_name='shapes',
    n_samples=1000,
    image_size=32,
    train_split=0.8,
    batch_size=128,
    normalize=False,  # Don't normalize for visualization
    random_state=42,
    noise_level=0.0,
    return_params=True
)

train_loader = dataset['train_loader']
train_params = dataset['train_params']

print(f"\nDataset info:")
print(f"  Train samples: {len(train_loader.dataset)}")
print(f"  Image size: 32x32")

# Get shape and color names
shape_names = ["circle", "triangle", "square", "rectangle", "pentagon", "hexagon", "star"]
color_names = ["red", "green", "blue", "yellow", "cyan", "magenta"]

# Check distribution
print(f"\nShape distribution:")
unique_shapes, counts = np.unique(train_params['shape'], return_counts=True)
for shape_idx, count in zip(unique_shapes, counts):
    print(f"  {shape_names[shape_idx]}: {count}")

print(f"\nColor distribution:")
unique_colors, counts = np.unique(train_params['color'], return_counts=True)
for color_idx, count in zip(unique_colors, counts):
    print(f"  {color_names[color_idx]}: {count}")

# Visualize diversity within each shape
print(f"\n" + "="*60)
print("Visualizing Diversity (10 samples per shape)")
print("="*60)

train_images = train_loader.dataset.tensors[0].numpy()

fig, axes = plt.subplots(7, 10, figsize=(15, 12))
fig.suptitle('Shapes Dataset - Diversity within Each Shape (RGB)', fontsize=16)

for shape_idx in range(7):
    # Find all samples of this shape
    indices = np.where(train_params['shape'] == shape_idx)[0]

    # Take first 10 samples
    samples = indices[:10]

    for i, sample_idx in enumerate(samples):
        # Reshape to RGB image (32, 32, 3)
        img = train_images[sample_idx].reshape(32, 32, 3)
        color_idx = train_params['color'][sample_idx]

        axes[shape_idx, i].imshow(img)
        axes[shape_idx, i].axis('off')

        if i == 0:
            axes[shape_idx, i].set_title(f'{shape_names[shape_idx]}\n({color_names[color_idx]})',
                                         fontsize=10, loc='left')
        else:
            axes[shape_idx, i].set_title(f'{color_names[color_idx]}', fontsize=8)

plt.tight_layout()
plt.savefig('shapes_diversity.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization to: shapes_diversity.png")

# Check size diversity
print(f"\n" + "="*60)
print("Size Diversity Analysis")
print("="*60)

# Compute mean pixel intensity for each image (proxy for size)
mean_intensities = train_images.mean(axis=1)

for shape_idx in range(7):
    indices = np.where(train_params['shape'] == shape_idx)[0]
    shape_intensities = mean_intensities[indices]

    print(f"\n{shape_names[shape_idx]}:")
    print(f"  Mean intensity: {shape_intensities.mean():.4f} ± {shape_intensities.std():.4f}")
    print(f"  Range: [{shape_intensities.min():.4f}, {shape_intensities.max():.4f}]")

    if shape_intensities.std() < 0.02:
        print(f"  ⚠️  WARNING: Low diversity! All {shape_names[shape_idx]}s look similar")
    else:
        print(f"  ✓ Good diversity")

# Additional visualization: all shapes with same color
print(f"\n" + "="*60)
print("Visualizing Color Consistency")
print("="*60)

fig2, axes2 = plt.subplots(6, 7, figsize=(14, 12))
fig2.suptitle('Shapes Dataset - All Shapes for Each Color (RGB)', fontsize=16)

for color_idx in range(6):
    for shape_idx in range(7):
        # Find samples with this color and shape
        indices = np.where((train_params['color'] == color_idx) &
                          (train_params['shape'] == shape_idx))[0]

        if len(indices) > 0:
            sample_idx = indices[0]  # Take first sample
            # Reshape to RGB image (32, 32, 3)
            img = train_images[sample_idx].reshape(32, 32, 3)

            axes2[color_idx, shape_idx].imshow(img)
            axes2[color_idx, shape_idx].axis('off')

            if color_idx == 0:
                axes2[color_idx, shape_idx].set_title(shape_names[shape_idx], fontsize=10)

            if shape_idx == 0:
                axes2[color_idx, shape_idx].set_ylabel(color_names[color_idx],
                                                        fontsize=10, rotation=0,
                                                        ha='right', va='center')
        else:
            axes2[color_idx, shape_idx].axis('off')
            axes2[color_idx, shape_idx].text(0.5, 0.5, 'N/A',
                                            ha='center', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('shapes_by_color.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved color visualization to: shapes_by_color.png")

print(f"\n" + "="*60)
print("Summary")
print("="*60)
print("✓ Dataset generated successfully")
print("✓ Visualizations saved:")
print("  - shapes_diversity.png: Shows 10 variations of each shape")
print("  - shapes_by_color.png: Shows all shapes in each color")
print("\nInspect these images to verify:")
print("  1. Shapes are clearly distinguishable")
print("  2. Colors affect grayscale intensity appropriately")
print("  3. Sufficient size variation within each shape")
