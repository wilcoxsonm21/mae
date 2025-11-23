"""Quick test script to verify 2D image generators."""

import numpy as np
import matplotlib.pyplot as plt
from data.generators import (
    generate_fourier_grid,
    generate_checkerboard,
    generate_radial_pattern,
    generate_gabor_patterns,
    generate_random_fourier,
    generate_noise_images
)

def test_generator(name, generator_func, **kwargs):
    """Test a generator and visualize results."""
    print(f"\nTesting {name}...")

    # Generate a few samples
    images = generator_func(n_samples=16, image_size=32, **kwargs)

    print(f"  Shape: {images.shape}")
    print(f"  Min: {images.min():.3f}, Max: {images.max():.3f}")

    # Reshape to 2D images
    images_2d = images.reshape(-1, 32, 32)

    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()

    for i in range(16):
        axes[i].imshow(images_2d[i], cmap='gray', interpolation='nearest')
        axes[i].axis('off')

    fig.suptitle(f'{name} - 16 samples', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'test_{name.replace(" ", "_").lower()}.png', dpi=100)
    print(f"  Saved to test_{name.replace(' ', '_').lower()}.png")
    plt.close()

if __name__ == '__main__':
    print("Testing 2D Image Generators")
    print("=" * 50)

    # Test all generators
    test_generator("Fourier Grid (Mixed)", generate_fourier_grid,
                   max_freq=8, n_frequencies=3, orientation='mixed', noise_level=0.01)

    test_generator("Fourier Grid (Horizontal)", generate_fourier_grid,
                   max_freq=8, n_frequencies=3, orientation='horizontal', noise_level=0.01)

    test_generator("Checkerboard (No Transform)", generate_checkerboard,
                   grid_sizes=[2, 4, 8], noise_level=0.01, apply_transforms=False)

    test_generator("Checkerboard (With Transforms)", generate_checkerboard,
                   grid_sizes=[2, 4, 8], noise_level=0.01, apply_transforms=True,
                   rotation_range=15.0, scale_range=(0.8, 1.2), perspective_range=0.2)

    test_generator("Radial Pattern", generate_radial_pattern,
                   n_circles=3, noise_level=0.01)

    test_generator("Gabor Patterns", generate_gabor_patterns,
                   n_components=3, noise_level=0.01)

    test_generator("Random Fourier", generate_random_fourier,
                   n_frequencies=10, max_freq=16, noise_level=0.01)

    test_generator("Gaussian Noise", generate_noise_images,
                   noise_type='gaussian')

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("Check the generated PNG files to visualize the patterns.")
