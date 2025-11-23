"""Visualization utilities for autoencoders with 2D image support."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.decomposition import PCA


def visualize_data_samples(data, image_size=None, title="Data Samples", max_samples=64):
    """Visualize data samples as 2D images.

    Args:
        data: Data tensor or numpy array of shape (n_samples, input_dim)
        image_size: Size of square images (None = try to infer or show as 1D)
        title: Title for the plot
        max_samples: Maximum number of samples to plot

    Returns:
        matplotlib figure
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # Limit number of samples for visualization
    n_samples = min(data.shape[0], max_samples)
    data = data[:n_samples]

    # Try to reshape to 2D images if image_size is provided
    if image_size is not None:
        # Reshape from (n_samples, H*W) to (n_samples, H, W)
        images = data.reshape(n_samples, image_size, image_size)

        # Create grid of images
        n_cols = min(8, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i in range(n_rows):
            for j in range(n_cols):
                idx = i * n_cols + j
                if idx < n_samples:
                    axes[i, j].imshow(images[idx], cmap='gray', interpolation='nearest')
                    axes[i, j].axis('off')
                else:
                    axes[i, j].axis('off')

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
    else:
        # Fallback to PCA visualization for non-image data
        input_dim = data.shape[1]
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        explained_var = pca.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.5, s=10)
        ax.set_xlabel(f'PC1 ({explained_var[0]:.2%} var)')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.2%} var)')
        ax.set_title(f'{title}\n(PCA projection from {input_dim}D)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    return fig


def visualize_reconstructions(original, reconstruction, image_size=None, mask=None,
                              n_samples=8, title="Reconstructions"):
    """Visualize original images vs reconstructions.

    Args:
        original: Original data tensor of shape (batch_size, input_dim)
        reconstruction: Reconstructed data tensor of shape (batch_size, input_dim)
        image_size: Size of square images (None = auto-detect or 1D)
        mask: Optional mask tensor for MAE
        n_samples: Number of samples to visualize
        title: Title for the plot

    Returns:
        matplotlib figure
    """
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstruction, torch.Tensor):
        reconstruction = reconstruction.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Limit to n_samples
    n_samples = min(n_samples, original.shape[0])
    original = original[:n_samples]
    reconstruction = reconstruction[:n_samples]
    if mask is not None:
        mask = mask[:n_samples]

    # Display as 2D images if image_size is provided
    if image_size is not None:
        # Reshape to 2D images
        orig_images = original.reshape(n_samples, image_size, image_size)
        recon_images = reconstruction.reshape(n_samples, image_size, image_size)

        # Create comparison figure
        n_cols = 3 if mask is None else 4
        fig, axes = plt.subplots(n_samples, n_cols,
                                figsize=(n_cols * 2, n_samples * 2))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Original
            axes[i, 0].imshow(orig_images[i], cmap='gray', interpolation='nearest')
            axes[i, 0].axis('off')
            if i == 0:
                axes[i, 0].set_title('Original', fontsize=10)

            # Reconstruction
            axes[i, 1].imshow(recon_images[i], cmap='gray', interpolation='nearest')
            axes[i, 1].axis('off')
            if i == 0:
                axes[i, 1].set_title('Reconstruction', fontsize=10)

            # Difference/Error
            diff = np.abs(orig_images[i] - recon_images[i])
            im = axes[i, 2].imshow(diff, cmap='hot', interpolation='nearest')
            axes[i, 2].axis('off')
            if i == 0:
                axes[i, 2].set_title('Error', fontsize=10)

            # Mask (if available)
            if mask is not None:
                mask_img = mask[i].reshape(image_size, image_size)
                axes[i, 3].imshow(mask_img, cmap='RdYlGn_r', interpolation='nearest',
                                 vmin=0, vmax=1)
                axes[i, 3].axis('off')
                if i == 0:
                    axes[i, 3].set_title('Mask', fontsize=10)

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

    else:
        # Fallback to heatmap visualization for non-image data
        input_dim = original.shape[1]

        if input_dim <= 10:
            # For low dimensions, show bar plots
            fig, axes = plt.subplots(n_samples, 3 if mask is not None else 2,
                                    figsize=(12 if mask is not None else 8, 2 * n_samples))
            if n_samples == 1:
                axes = axes.reshape(1, -1)

            for i in range(n_samples):
                # Original
                axes[i, 0].bar(range(input_dim), original[i])
                axes[i, 0].set_ylim([original.min() - 0.5, original.max() + 0.5])
                if i == 0:
                    axes[i, 0].set_title('Original')
                if i == n_samples - 1:
                    axes[i, 0].set_xlabel('Dimension')

                # Reconstruction
                axes[i, 1].bar(range(input_dim), reconstruction[i])
                axes[i, 1].set_ylim([original.min() - 0.5, original.max() + 0.5])
                if i == 0:
                    axes[i, 1].set_title('Reconstruction')
                if i == n_samples - 1:
                    axes[i, 1].set_xlabel('Dimension')

                # Mask (if available)
                if mask is not None:
                    axes[i, 2].bar(range(input_dim), mask[i])
                    axes[i, 2].set_ylim([0, 1.2])
                    if i == 0:
                        axes[i, 2].set_title('Mask')
                    if i == n_samples - 1:
                        axes[i, 2].set_xlabel('Dimension')
        else:
            # For high dimensions, show heatmaps
            n_cols = 3 if mask is not None else 2
            fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 8))

            # Original
            im0 = axes[0].imshow(original.T, aspect='auto', cmap='viridis')
            axes[0].set_title('Original')
            axes[0].set_xlabel('Sample')
            axes[0].set_ylabel('Dimension')
            plt.colorbar(im0, ax=axes[0])

            # Reconstruction
            im1 = axes[1].imshow(reconstruction.T, aspect='auto', cmap='viridis')
            axes[1].set_title('Reconstruction')
            axes[1].set_xlabel('Sample')
            axes[1].set_ylabel('Dimension')
            plt.colorbar(im1, ax=axes[1])

            # Mask (if available)
            if mask is not None:
                im2 = axes[2].imshow(mask.T, aspect='auto', cmap='RdYlGn_r')
                axes[2].set_title('Mask (red=masked)')
                axes[2].set_xlabel('Sample')
                axes[2].set_ylabel('Dimension')
                plt.colorbar(im2, ax=axes[2])

        fig.suptitle(title, fontsize=14, y=1.0 if input_dim <= 10 else 1.02)
        plt.tight_layout()

    return fig


def visualize_latent_space(latent, labels=None, title="Latent Space"):
    """Visualize latent space.

    Args:
        latent: Latent representations of shape (n_samples, latent_dim)
        labels: Optional labels for coloring points
        title: Title for the plot

    Returns:
        matplotlib figure
    """
    if isinstance(latent, torch.Tensor):
        latent = latent.detach().cpu().numpy()

    latent_dim = latent.shape[1]

    if latent_dim == 2:
        # Direct 2D visualization
        fig, ax = plt.subplots(figsize=(8, 8))
        if labels is not None:
            scatter = ax.scatter(latent[:, 0], latent[:, 1], c=labels,
                               alpha=0.5, s=10, cmap='viridis')
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(latent[:, 0], latent[:, 1], alpha=0.5, s=10)
        ax.set_xlabel('Latent Dim 1')
        ax.set_ylabel('Latent Dim 2')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    elif latent_dim == 3:
        # 3D visualization
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        if labels is not None:
            scatter = ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2],
                               c=labels, alpha=0.5, s=10, cmap='viridis')
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2], alpha=0.5, s=10)
        ax.set_xlabel('Latent Dim 1')
        ax.set_ylabel('Latent Dim 2')
        ax.set_zlabel('Latent Dim 3')
        ax.set_title(title)
    else:
        # Use PCA to reduce to 2D
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent)
        explained_var = pca.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(8, 8))
        if labels is not None:
            scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels,
                               alpha=0.5, s=10, cmap='viridis')
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5, s=10)
        ax.set_xlabel(f'PC1 ({explained_var[0]:.2%} var)')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.2%} var)')
        ax.set_title(f'{title}\n(PCA projection from {latent_dim}D)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_reconstruction_errors(original, reconstruction, image_size=None,
                                    title="Reconstruction Errors"):
    """Visualize reconstruction errors.

    Args:
        original: Original data tensor
        reconstruction: Reconstructed data tensor
        image_size: Size of square images (None = treat as 1D)
        title: Title for the plot

    Returns:
        matplotlib figure
    """
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstruction, torch.Tensor):
        reconstruction = reconstruction.detach().cpu().numpy()

    # Compute per-sample and per-dimension errors
    errors = np.abs(original - reconstruction)
    per_sample_error = errors.mean(axis=1)
    per_dim_error = errors.mean(axis=0)

    # Create figure with appropriate layout
    if image_size is not None:
        # Show error as image + histogram
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Per-sample error histogram
        axes[0].hist(per_sample_error, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Mean Absolute Error')
        axes[0].set_ylabel('Number of Samples')
        axes[0].set_title('Per-Sample Error Distribution')
        axes[0].axvline(per_sample_error.mean(), color='red', linestyle='--',
                       label=f'Mean: {per_sample_error.mean():.4f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Per-pixel error as 2D image
        per_pixel_error = per_dim_error.reshape(image_size, image_size)
        im = axes[1].imshow(per_pixel_error, cmap='hot', interpolation='nearest')
        axes[1].set_title('Per-Pixel Error Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])

        # Example error images
        # Pick a few samples with different error levels
        n_examples = min(8, original.shape[0])
        indices = np.linspace(0, original.shape[0]-1, n_examples, dtype=int)
        example_errors = errors[indices].reshape(n_examples, image_size, image_size)

        # Create small grid
        n_cols = min(4, n_examples)
        n_rows = (n_examples + n_cols - 1) // n_cols

        # Remove third axis and create subplots
        axes[2].axis('off')
        gs = axes[2].get_gridspec()
        # Create new GridSpec in place of axes[2]
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        inner_gs = GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs[2])

        for i in range(n_examples):
            row = i // n_cols
            col = i % n_cols
            ax = fig.add_subplot(inner_gs[row, col])
            ax.imshow(example_errors[i], cmap='hot', interpolation='nearest')
            ax.axis('off')
            ax.set_title(f'{per_sample_error[indices[i]]:.3f}', fontsize=8)

    else:
        # Original 1D visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Per-sample error histogram
        axes[0].hist(per_sample_error, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Mean Absolute Error')
        axes[0].set_ylabel('Number of Samples')
        axes[0].set_title('Per-Sample Reconstruction Error')
        axes[0].axvline(per_sample_error.mean(), color='red', linestyle='--',
                       label=f'Mean: {per_sample_error.mean():.4f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Per-dimension error
        axes[1].bar(range(len(per_dim_error)), per_dim_error)
        axes[1].set_xlabel('Dimension')
        axes[1].set_ylabel('Mean Absolute Error')
        axes[1].set_title('Per-Dimension Reconstruction Error')
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def log_visualizations_to_wandb(model, train_loader, val_loader, device,
                                image_size=None, step=None, max_samples=1000):
    """Create and log visualizations to wandb.

    Args:
        model: Trained model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run on
        image_size: Size of square images (None = treat as 1D data)
        step: Current training step
        max_samples: Maximum samples for visualization
    """
    model.eval()

    with torch.no_grad():
        # Get a batch from train and val
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        train_data, _ = train_batch
        val_data, _ = val_batch

        train_data = train_data.to(device)
        val_data = val_data.to(device)

        # Forward pass
        train_output = model(train_data)
        val_output = model(val_data)

        # Create visualizations
        vis_dict = {}

        # 1. Dataset samples
        fig_train_samples = visualize_data_samples(
            train_data, image_size, "Training Data Samples", max_samples=64
        )
        vis_dict['data/train_samples'] = wandb.Image(fig_train_samples)
        plt.close(fig_train_samples)

        fig_val_samples = visualize_data_samples(
            val_data, image_size, "Validation Data Samples", max_samples=64
        )
        vis_dict['data/val_samples'] = wandb.Image(fig_val_samples)
        plt.close(fig_val_samples)

        # 2. Train reconstructions
        train_mask = train_output.get('mask', None)
        fig_train_recon = visualize_reconstructions(
            train_data, train_output['reconstruction'], image_size, train_mask,
            n_samples=8, title="Training Reconstructions"
        )
        vis_dict['reconstructions/train'] = wandb.Image(fig_train_recon)
        plt.close(fig_train_recon)

        # 3. Val reconstructions
        val_mask = val_output.get('mask', None)
        fig_val_recon = visualize_reconstructions(
            val_data, val_output['reconstruction'], image_size, val_mask,
            n_samples=8, title="Validation Reconstructions"
        )
        vis_dict['reconstructions/val'] = wandb.Image(fig_val_recon)
        plt.close(fig_val_recon)

        # 4. Latent space visualization
        train_latents = []
        val_latents = []
        n_batches = min(5, len(train_loader))

        for i, (data, _) in enumerate(train_loader):
            if i >= n_batches:
                break
            data = data.to(device)
            output = model(data)
            train_latents.append(output['latent'])

        for i, (data, _) in enumerate(val_loader):
            if i >= n_batches:
                break
            data = data.to(device)
            output = model(data)
            val_latents.append(output['latent'])

        train_latents = torch.cat(train_latents, dim=0)
        val_latents = torch.cat(val_latents, dim=0)

        fig_train_latent = visualize_latent_space(
            train_latents, title="Training Latent Space"
        )
        vis_dict['latent/train'] = wandb.Image(fig_train_latent)
        plt.close(fig_train_latent)

        fig_val_latent = visualize_latent_space(
            val_latents, title="Validation Latent Space"
        )
        vis_dict['latent/val'] = wandb.Image(fig_val_latent)
        plt.close(fig_val_latent)

        # 5. Reconstruction errors
        fig_train_errors = visualize_reconstruction_errors(
            train_data, train_output['reconstruction'], image_size,
            "Training Reconstruction Errors"
        )
        vis_dict['errors/train'] = wandb.Image(fig_train_errors)
        plt.close(fig_train_errors)

        fig_val_errors = visualize_reconstruction_errors(
            val_data, val_output['reconstruction'], image_size,
            "Validation Reconstruction Errors"
        )
        vis_dict['errors/val'] = wandb.Image(fig_val_errors)
        plt.close(fig_val_errors)

        # Log to wandb
        if step is not None:
            wandb.log(vis_dict, step=step)
        else:
            wandb.log(vis_dict)

    model.train()
