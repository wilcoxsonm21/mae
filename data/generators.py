"""Synthetic data generators for autoencoder training."""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_swiss_roll, make_s_curve


def generate_gaussian_mixture(n_samples, input_dim, n_components=5, random_state=42):
    """Generate data from a Gaussian mixture model.

    Args:
        n_samples: Number of samples to generate
        input_dim: Dimension of the data
        n_components: Number of Gaussian components
        random_state: Random seed for reproducibility

    Returns:
        numpy array of shape (n_samples, input_dim)
    """
    np.random.seed(random_state)
    data = []

    samples_per_component = n_samples // n_components
    for i in range(n_components):
        # Random mean for each component
        mean = np.random.randn(input_dim) * 3
        # Random covariance (diagonal for simplicity)
        cov = np.diag(np.random.rand(input_dim) * 0.5 + 0.5)
        component_data = np.random.multivariate_normal(mean, cov, samples_per_component)
        data.append(component_data)

    data = np.vstack(data)
    np.random.shuffle(data)
    return data.astype(np.float32)


def generate_swiss_roll_data(n_samples, noise=0.1, random_state=42):
    """Generate Swiss roll dataset.

    Args:
        n_samples: Number of samples to generate
        noise: Noise level
        random_state: Random seed

    Returns:
        numpy array of shape (n_samples, 3)
    """
    data, _ = make_swiss_roll(n_samples, noise=noise, random_state=random_state)
    return data.astype(np.float32)


def generate_s_curve_data(n_samples, noise=0.1, random_state=42):
    """Generate S-curve dataset.

    Args:
        n_samples: Number of samples to generate
        noise: Noise level
        random_state: Random seed

    Returns:
        numpy array of shape (n_samples, 3)
    """
    data, _ = make_s_curve(n_samples, noise=noise, random_state=random_state)
    return data.astype(np.float32)


def generate_concentric_circles(n_samples, input_dim, n_circles=3, random_state=42):
    """Generate concentric circles/spheres in high dimensions.

    Args:
        n_samples: Number of samples to generate
        input_dim: Dimension of the data
        n_circles: Number of concentric circles
        random_state: Random seed

    Returns:
        numpy array of shape (n_samples, input_dim)
    """
    np.random.seed(random_state)
    data = []

    samples_per_circle = n_samples // n_circles
    for i in range(n_circles):
        # Generate random directions
        directions = np.random.randn(samples_per_circle, input_dim)
        # Normalize to unit sphere
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms
        # Scale to different radii
        radius = (i + 1) * 2.0
        circle_data = directions * radius
        # Add small noise
        circle_data += np.random.randn(samples_per_circle, input_dim) * 0.1
        data.append(circle_data)

    data = np.vstack(data)
    np.random.shuffle(data)
    return data.astype(np.float32)


def generate_uniform_noise(n_samples, input_dim, low=-1.0, high=1.0, random_state=42):
    """Generate uniform random noise.

    Args:
        n_samples: Number of samples to generate
        input_dim: Dimension of the data
        low: Lower bound
        high: Upper bound
        random_state: Random seed

    Returns:
        numpy array of shape (n_samples, input_dim)
    """
    np.random.seed(random_state)
    return np.random.uniform(low, high, (n_samples, input_dim)).astype(np.float32)


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


def get_dataset(dataset_name, n_samples=10000, input_dim=64, train_split=0.8,
                batch_size=128, normalize=True, random_state=42, **kwargs):
    """Get a synthetic dataset with train/val split.

    Args:
        dataset_name: Name of the dataset ('gaussian_mixture', 'swiss_roll',
                                           's_curve', 'concentric_circles', 'uniform')
        n_samples: Total number of samples
        input_dim: Input dimension (used for applicable datasets)
        train_split: Fraction of data for training
        batch_size: Batch size for DataLoader
        normalize: Whether to normalize the data
        random_state: Random seed
        **kwargs: Additional arguments for specific generators

    Returns:
        Dictionary containing:
            - 'train_loader': Training DataLoader
            - 'val_loader': Validation DataLoader
            - 'input_dim': Input dimension
            - 'mean': Data mean (if normalized)
            - 'std': Data std (if normalized)
    """
    # Generate data based on dataset name
    if dataset_name == 'gaussian_mixture':
        n_components = kwargs.get('n_components', 5)
        data = generate_gaussian_mixture(n_samples, input_dim, n_components, random_state)
    elif dataset_name == 'swiss_roll':
        noise = kwargs.get('noise', 0.1)
        data = generate_swiss_roll_data(n_samples, noise, random_state)
        input_dim = 3  # Swiss roll is always 3D
    elif dataset_name == 's_curve':
        noise = kwargs.get('noise', 0.1)
        data = generate_s_curve_data(n_samples, noise, random_state)
        input_dim = 3  # S-curve is always 3D
    elif dataset_name == 'concentric_circles':
        n_circles = kwargs.get('n_circles', 3)
        data = generate_concentric_circles(n_samples, input_dim, n_circles, random_state)
    elif dataset_name == 'uniform':
        low = kwargs.get('low', -1.0)
        high = kwargs.get('high', 1.0)
        data = generate_uniform_noise(n_samples, input_dim, low, high, random_state)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Normalize if requested
    mean, std = None, None
    if normalize:
        data, mean, std = normalize_data(data)

    # Split into train and validation
    n_train = int(n_samples * train_split)
    train_data = data[:n_train]
    val_data = data[n_train:]

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
        num_workers=0  # Can be increased for faster loading
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'input_dim': input_dim,
        'mean': mean,
        'std': std
    }
