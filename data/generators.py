"""2D image pattern generators using Fourier space signals."""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def generate_fourier_grid(n_samples, image_size=32, max_freq=8, n_frequencies=3,
                          orientation='mixed', noise_level=0.01, random_state=42):
    """Generate 2D grid patterns by combining Fourier frequencies.

    Creates images by placing impulses in Fourier space at specific frequencies,
    then inverse transforming to get spatial domain grid patterns.

    Args:
        n_samples: Number of images to generate
        image_size: Size of square images (height=width)
        max_freq: Maximum frequency component (1 to image_size//2)
        n_frequencies: Number of frequency components to combine
        orientation: 'horizontal', 'vertical', 'diagonal', or 'mixed'
        noise_level: Amount of random noise to add (0-1)
        random_state: Random seed for reproducibility

    Returns:
        numpy array of shape (n_samples, image_size * image_size)
    """
    np.random.seed(random_state)
    images = []

    for i in range(n_samples):
        # Create empty frequency spectrum
        freq_spectrum = np.zeros((image_size, image_size), dtype=np.complex128)

        # Add random frequency components based on orientation
        for _ in range(n_frequencies):
            freq_x = np.random.randint(1, max_freq + 1)
            freq_y = np.random.randint(1, max_freq + 1)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(0.5, 1.0)

            if orientation == 'horizontal':
                freq_y = 0
            elif orientation == 'vertical':
                freq_x = 0
            elif orientation == 'diagonal':
                freq_y = freq_x if np.random.rand() > 0.5 else -freq_x
            # For 'mixed', use both freq_x and freq_y as generated

            # Add frequency component (and its conjugate for real-valued output)
            freq_spectrum[freq_y, freq_x] = amplitude * np.exp(1j * phase)
            freq_spectrum[-freq_y, -freq_x] = amplitude * np.exp(-1j * phase)

        # Inverse FFT to get spatial domain image
        image = np.fft.ifft2(np.fft.ifftshift(freq_spectrum)).real

        # Add noise
        if noise_level > 0:
            image += np.random.randn(image_size, image_size) * noise_level

        # Flatten to 1D vector
        images.append(image.flatten())

    images = np.array(images, dtype=np.float32)
    return images


def generate_checkerboard(n_samples, image_size=32, grid_sizes=None,
                         noise_level=0.01, apply_transforms=True,
                         rotation_range=15.0, scale_range=(0.8, 1.2),
                         perspective_range=0.2, random_state=42,
                         return_params=False):
    """Generate checkerboard patterns with varying grid sizes and homographic transformations.

    Args:
        n_samples: Number of images to generate
        image_size: Size of square images
        grid_sizes: List of possible grid sizes (cells per row/col), None for random
        noise_level: Amount of random noise to add
        apply_transforms: Whether to apply random homographic transformations
        rotation_range: Max rotation angle in degrees (±rotation_range)
        scale_range: Range of random scaling factors (min_scale, max_scale)
        perspective_range: Amount of perspective distortion (0-1, 0=none)
        random_state: Random seed
        return_params: If True, also return generation parameters for each sample

    Returns:
        If return_params=False:
            numpy array of shape (n_samples, image_size * image_size)
        If return_params=True:
            tuple of (images, params_dict) where params_dict contains:
                - 'grid_size': (n_samples,) array of grid sizes
                - 'rotation': (n_samples,) array of rotation angles in degrees
                - 'scale': (n_samples,) array of scale factors
                - 'perspective_x': (n_samples,) array of x-axis perspective
                - 'perspective_y': (n_samples,) array of y-axis perspective
    """
    np.random.seed(random_state)
    images = []

    # Storage for generation parameters
    if return_params:
        params = {
            'grid_size': [],
            'rotation': [],
            'scale': [],
            'perspective_x': [],
            'perspective_y': []
        }

    if grid_sizes is None:
        grid_sizes = [2, 4, 8, 16]

    for i in range(n_samples):
        # Random grid size for each sample
        grid_size = np.random.choice(grid_sizes)

        # Initialize transform parameters (will be set if transforms are applied)
        angle_deg = 0.0
        scale_factor = 1.0
        persp_x = 0.0
        persp_y = 0.0

        # Create larger canvas for transformations (to avoid edge artifacts)
        canvas_size = int(image_size * 1.5) if apply_transforms else image_size

        # Calculate cell size on the canvas
        cell_size = canvas_size // grid_size

        # Create checkerboard pattern in spatial domain
        image = np.zeros((canvas_size, canvas_size), dtype=np.float32)

        for row in range(grid_size):
            for col in range(grid_size):
                # Determine if this cell should be filled (checkerboard pattern)
                if (row + col) % 2 == 0:
                    y_start = row * cell_size
                    y_end = min((row + 1) * cell_size, canvas_size)
                    x_start = col * cell_size
                    x_end = min((col + 1) * cell_size, canvas_size)
                    image[y_start:y_end, x_start:x_end] = 1.0

        # Apply random homographic transformation
        if apply_transforms:
            # Create coordinate grid
            y_coords, x_coords = np.meshgrid(
                np.arange(image_size),
                np.arange(image_size),
                indexing='ij'
            )

            # Center coordinates
            center = image_size / 2
            x_centered = x_coords - center
            y_centered = y_coords - center

            # Random rotation
            angle_deg = np.random.uniform(-rotation_range, rotation_range)
            angle = angle_deg * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_rot = x_centered * cos_a - y_centered * sin_a
            y_rot = x_centered * sin_a + y_centered * cos_a

            # Random scale
            scale_factor = np.random.uniform(scale_range[0], scale_range[1])
            x_scaled = x_rot * scale_factor
            y_scaled = y_rot * scale_factor

            # Random perspective/shear distortion
            if perspective_range > 0:
                # Add slight perspective warping
                persp_x = np.random.uniform(-perspective_range, perspective_range)
                persp_y = np.random.uniform(-perspective_range, perspective_range)

                # Apply perspective scaling based on position
                scale_x = 1 + persp_x * (y_centered / (image_size / 2))
                scale_y = 1 + persp_y * (x_centered / (image_size / 2))

                x_scaled = x_scaled * scale_x
                y_scaled = y_scaled * scale_y

            # Map back to canvas coordinates
            x_mapped = x_scaled + canvas_size / 2
            y_mapped = y_scaled + canvas_size / 2

            # Interpolate to get transformed image
            # Clip coordinates to valid range
            x_mapped = np.clip(x_mapped, 0, canvas_size - 1)
            y_mapped = np.clip(y_mapped, 0, canvas_size - 1)

            # Bilinear interpolation
            x0 = np.floor(x_mapped).astype(int)
            x1 = np.minimum(x0 + 1, canvas_size - 1)
            y0 = np.floor(y_mapped).astype(int)
            y1 = np.minimum(y0 + 1, canvas_size - 1)

            wx = x_mapped - x0
            wy = y_mapped - y0

            # Interpolate
            image_transformed = (
                image[y0, x0] * (1 - wx) * (1 - wy) +
                image[y0, x1] * wx * (1 - wy) +
                image[y1, x0] * (1 - wx) * wy +
                image[y1, x1] * wx * wy
            )

            image = image_transformed
        else:
            # Just crop to image_size if no transforms
            image = image[:image_size, :image_size]

        # Add Gaussian noise
        if noise_level > 0:
            image += np.random.randn(image_size, image_size) * noise_level

        images.append(image.flatten())

        # Store generation parameters
        if return_params:
            params['grid_size'].append(grid_size)
            params['rotation'].append(angle_deg)
            params['scale'].append(scale_factor)
            params['perspective_x'].append(persp_x)
            params['perspective_y'].append(persp_y)

    images = np.array(images, dtype=np.float32)

    if return_params:
        # Convert parameter lists to numpy arrays
        params_array = {k: np.array(v, dtype=np.float32) for k, v in params.items()}
        return images, params_array
    else:
        return images


def generate_radial_pattern(n_samples, image_size=32, n_circles=3,
                           noise_level=0.01, random_state=42):
    """Generate concentric circle patterns.

    Args:
        n_samples: Number of images to generate
        image_size: Size of square images
        n_circles: Number of concentric circles
        noise_level: Amount of random noise to add
        random_state: Random seed

    Returns:
        numpy array of shape (n_samples, image_size * image_size)
    """
    np.random.seed(random_state)
    images = []

    # Create coordinate grids
    y, x = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing='ij')
    center = image_size / 2

    for i in range(n_samples):
        # Random parameters for this sample
        radius_scale = np.random.uniform(0.5, 1.5)
        n_circles_sample = np.random.randint(2, n_circles + 1)

        # Distance from center
        r = np.sqrt((x - center)**2 + (y - center)**2)

        # Create radial pattern
        image = np.zeros((image_size, image_size))
        for j in range(n_circles_sample):
            freq = (j + 1) * 2 * np.pi * radius_scale / image_size
            image += np.cos(freq * r)

        # Add noise
        if noise_level > 0:
            image += np.random.randn(image_size, image_size) * noise_level

        images.append(image.flatten())

    images = np.array(images, dtype=np.float32)
    return images


def generate_gabor_patterns(n_samples, image_size=32, n_components=3,
                           noise_level=0.01, random_state=42):
    """Generate Gabor-like patterns (oriented sinusoidal gratings).

    Args:
        n_samples: Number of images to generate
        image_size: Size of square images
        n_components: Number of Gabor components to combine
        noise_level: Amount of random noise to add
        random_state: Random seed

    Returns:
        numpy array of shape (n_samples, image_size * image_size)
    """
    np.random.seed(random_state)
    images = []

    # Create coordinate grids
    y, x = np.meshgrid(np.linspace(-1, 1, image_size),
                       np.linspace(-1, 1, image_size), indexing='ij')

    for i in range(n_samples):
        image = np.zeros((image_size, image_size))

        for _ in range(n_components):
            # Random orientation
            theta = np.random.uniform(0, np.pi)

            # Random frequency
            freq = np.random.uniform(2, 8)

            # Random phase
            phase = np.random.uniform(0, 2 * np.pi)

            # Rotate coordinates
            x_rot = x * np.cos(theta) + y * np.sin(theta)

            # Create sinusoidal grating
            amplitude = np.random.uniform(0.5, 1.0)
            grating = amplitude * np.cos(2 * np.pi * freq * x_rot + phase)

            # Optional: Add Gaussian envelope for true Gabor
            # sigma = np.random.uniform(0.2, 0.5)
            # envelope = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            # grating *= envelope

            image += grating

        # Add noise
        if noise_level > 0:
            image += np.random.randn(image_size, image_size) * noise_level

        images.append(image.flatten())

    images = np.array(images, dtype=np.float32)
    return images


def generate_random_fourier(n_samples, image_size=32, n_frequencies=10,
                           max_freq=None, noise_level=0.01, random_state=42):
    """Generate random patterns by combining random Fourier frequencies.

    Args:
        n_samples: Number of images to generate
        image_size: Size of square images
        n_frequencies: Number of random frequency components
        max_freq: Maximum frequency (None = image_size//2)
        noise_level: Amount of random noise to add
        random_state: Random seed

    Returns:
        numpy array of shape (n_samples, image_size * image_size)
    """
    np.random.seed(random_state)
    images = []

    if max_freq is None:
        max_freq = image_size // 2

    for i in range(n_samples):
        # Create empty frequency spectrum
        freq_spectrum = np.zeros((image_size, image_size), dtype=np.complex128)

        # Add random frequency components
        for _ in range(n_frequencies):
            freq_x = np.random.randint(-max_freq, max_freq + 1)
            freq_y = np.random.randint(-max_freq, max_freq + 1)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.exponential(1.0)  # Exponential for variety

            freq_spectrum[freq_y, freq_x] = amplitude * np.exp(1j * phase)

        # Ensure Hermitian symmetry for real-valued output
        freq_spectrum = (freq_spectrum + np.conj(freq_spectrum[::-1, ::-1])) / 2

        # Inverse FFT to get spatial domain image
        image = np.fft.ifft2(np.fft.ifftshift(freq_spectrum)).real

        # Add noise
        if noise_level > 0:
            image += np.random.randn(image_size, image_size) * noise_level

        images.append(image.flatten())

    images = np.array(images, dtype=np.float32)
    return images


def generate_noise_images(n_samples, image_size=32, noise_type='gaussian', random_state=42):
    """Generate random noise images.

    Args:
        n_samples: Number of images to generate
        image_size: Size of square images
        noise_type: 'gaussian' or 'uniform'
        random_state: Random seed

    Returns:
        numpy array of shape (n_samples, image_size * image_size)
    """
    np.random.seed(random_state)

    if noise_type == 'gaussian':
        images = np.random.randn(n_samples, image_size * image_size)
    elif noise_type == 'uniform':
        images = np.random.uniform(-1, 1, (n_samples, image_size * image_size))
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return images.astype(np.float32)


def normalize_data(data):
    """Normalize data to zero mean and unit variance.

    Args:
        data: numpy array of shape (n_samples, input_dim)

    Returns:
        Normalized data, mean, std
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8  # Add small epsilon to avoid division by zero
    normalized_data = (data - mean) / std
    return normalized_data, mean, std


def get_dataset(dataset_name, n_samples=10000, image_size=32, train_split=0.8,
                batch_size=128, normalize=True, random_state=42, **kwargs):
    """Get a 2D image dataset with train/val split.

    Args:
        dataset_name: Name of the dataset:
            - 'fourier_grid': Grid patterns from Fourier frequencies
            - 'checkerboard': Checkerboard patterns
            - 'radial': Concentric circle patterns
            - 'gabor': Oriented sinusoidal gratings
            - 'random_fourier': Random frequency combinations
            - 'noise': Random noise images
        n_samples: Total number of samples
        image_size: Size of square images (height = width)
        train_split: Fraction of data for training
        batch_size: Batch size for DataLoader
        normalize: Whether to normalize the data
        random_state: Random seed
        **kwargs: Additional arguments for specific generators

    Returns:
        Dictionary containing:
            - 'train_loader': Training DataLoader
            - 'val_loader': Validation DataLoader
            - 'input_dim': Input dimension (image_size * image_size)
            - 'image_size': Size of square image
            - 'mean': Data mean (if normalized)
            - 'std': Data std (if normalized)
    """
    # Generate data based on dataset name
    generation_params = None  # Will be populated for datasets that support it

    if dataset_name == 'fourier_grid':
        max_freq = kwargs.get('max_freq', 8)
        n_frequencies = kwargs.get('n_frequencies', 3)
        orientation = kwargs.get('orientation', 'mixed')
        noise_level = kwargs.get('noise_level', 0.01)
        data = generate_fourier_grid(n_samples, image_size, max_freq, n_frequencies,
                                     orientation, noise_level, random_state)

    elif dataset_name == 'checkerboard':
        grid_sizes = kwargs.get('grid_sizes', None)
        noise_level = kwargs.get('noise_level', 0.01)
        apply_transforms = kwargs.get('apply_transforms', True)
        rotation_range = kwargs.get('rotation_range', 15.0)
        scale_range = kwargs.get('scale_range', (0.8, 1.2))
        perspective_range = kwargs.get('perspective_range', 0.2)
        return_params = kwargs.get('return_params', False)

        result = generate_checkerboard(
            n_samples, image_size, grid_sizes, noise_level,
            apply_transforms, rotation_range, scale_range, perspective_range,
            random_state, return_params
        )

        if return_params:
            data, generation_params = result
        else:
            data = result
            generation_params = None

    elif dataset_name == 'radial':
        n_circles = kwargs.get('n_circles', 3)
        noise_level = kwargs.get('noise_level', 0.01)
        data = generate_radial_pattern(n_samples, image_size, n_circles,
                                       noise_level, random_state)

    elif dataset_name == 'gabor':
        n_components = kwargs.get('n_components', 3)
        noise_level = kwargs.get('noise_level', 0.01)
        data = generate_gabor_patterns(n_samples, image_size, n_components,
                                       noise_level, random_state)

    elif dataset_name == 'random_fourier':
        n_frequencies = kwargs.get('n_frequencies', 10)
        max_freq = kwargs.get('max_freq', None)
        noise_level = kwargs.get('noise_level', 0.01)
        data = generate_random_fourier(n_samples, image_size, n_frequencies,
                                       max_freq, noise_level, random_state)

    elif dataset_name == 'noise':
        noise_type = kwargs.get('noise_type', 'gaussian')
        data = generate_noise_images(n_samples, image_size, noise_type, random_state)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: "
                        "'fourier_grid', 'checkerboard', 'radial', 'gabor', "
                        "'random_fourier', 'noise'")

    # Normalize if requested
    mean, std = None, None
    if normalize:
        data, mean, std = normalize_data(data)

    # Split into train and validation
    n_train = int(n_samples * train_split)
    train_data = data[:n_train]
    val_data = data[n_train:]

    # Split generation parameters if available
    train_params = None
    val_params = None
    if generation_params is not None:
        train_params = {k: v[:n_train] for k, v in generation_params.items()}
        val_params = {k: v[n_train:] for k, v in generation_params.items()}

    # Convert to PyTorch tensors
    train_tensor = torch.from_numpy(train_data)
    val_tensor = torch.from_numpy(val_data)

    # Create TensorDatasets (targets are the same as inputs for autoencoders)
    train_dataset = TensorDataset(train_tensor, train_tensor)
    val_dataset = TensorDataset(val_tensor, val_tensor)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    input_dim = image_size * image_size

    result_dict = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'input_dim': input_dim,
        'image_size': image_size,
        'mean': mean,
        'std': std
    }

    # Add generation parameters if available
    if generation_params is not None:
        result_dict['train_params'] = train_params
        result_dict['val_params'] = val_params
        result_dict['generation_params'] = generation_params  # Full params before split

    return result_dict
