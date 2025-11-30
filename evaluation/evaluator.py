"""Evaluation framework for autoencoders."""

import torch
from tqdm import tqdm
from .metrics import (
    reconstruction_loss,
    latent_variance,
    reconstruction_error_per_sample,
    masked_reconstruction_accuracy
)


class Evaluator:
    """Evaluator for autoencoder models.

    Provides a flexible framework for evaluating models on various metrics.
    Easy to extend with additional metrics.
    """

    def __init__(self, model, device='cpu'):
        """Initialize evaluator.

        Args:
            model: Autoencoder model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.metrics = {}

    def add_metric(self, name, metric_fn):
        """Add a custom metric to the evaluator.

        Args:
            name: Name of the metric
            metric_fn: Function that takes (model_output, target) and returns a scalar
        """
        self.metrics[name] = metric_fn

    def _expand_patch_mask_to_pixels(self, patch_mask, input_dim):
        """Expand patch-level mask to pixel-level mask.

        Args:
            patch_mask: (batch_size, num_patches) - binary mask at patch level
            input_dim: Total number of pixels (channels * height * width)

        Returns:
            pixel_mask: (batch_size, input_dim) - binary mask at pixel level
        """
        batch_size, num_patches = patch_mask.shape

        # Infer image dimensions (assume square image)
        num_patches_per_side = int(num_patches ** 0.5)
        image_size = int(input_dim ** 0.5)
        patch_size = image_size // num_patches_per_side
        in_channels = input_dim // (image_size * image_size)

        # Reshape patch mask to 2D grid
        patch_mask_2d = patch_mask.view(batch_size, num_patches_per_side, num_patches_per_side)

        # Expand each patch to its full pixel size
        pixel_mask_2d = patch_mask_2d.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)

        # Expand for channels and flatten
        pixel_mask = pixel_mask_2d.unsqueeze(1).expand(-1, in_channels, -1, -1)
        pixel_mask = pixel_mask.reshape(batch_size, input_dim)

        return pixel_mask

    @torch.no_grad()
    def evaluate(self, dataloader, objective=None, verbose=True):
        """Evaluate model on a dataset.

        Args:
            dataloader: DataLoader for evaluation data
            objective: Optional objective function to compute loss
            verbose: Whether to show progress bar

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        # Accumulators for metrics
        total_samples = 0
        total_reconstruction_loss_mse = 0.0
        total_reconstruction_loss_l1 = 0.0
        total_latent_variance = 0.0
        total_objective_loss = 0.0

        # For MAE models
        total_masked_loss = 0.0
        total_unmasked_loss = 0.0
        total_masked_accuracy = 0.0
        has_masking = False

        # Iterate over batches
        iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader

        for batch_idx, (data, target) in enumerate(iterator):
            data = data.to(self.device)
            target = target.to(self.device)
            batch_size = data.size(0)

            # Forward pass
            model_output = self.model(data)
            reconstruction = model_output['reconstruction']
            latent = model_output['latent']

            # Compute basic metrics
            recon_loss_mse = reconstruction_loss(reconstruction, target, 'mse')
            recon_loss_l1 = reconstruction_loss(reconstruction, target, 'l1')
            lat_var = latent_variance(latent)

            total_reconstruction_loss_mse += recon_loss_mse.item() * batch_size
            total_reconstruction_loss_l1 += recon_loss_l1.item() * batch_size
            total_latent_variance += lat_var.item() * batch_size

            # Compute objective loss if provided
            if objective is not None:
                obj_output = objective(model_output, target)
                total_objective_loss += obj_output['loss'].item() * batch_size

                # Check for masked losses
                if 'masked_loss' in obj_output:
                    has_masking = True
                    total_masked_loss += obj_output['masked_loss'].item() * batch_size
                    total_unmasked_loss += obj_output['unmasked_loss'].item() * batch_size

            # Compute masked accuracy if applicable
            if 'mask' in model_output or 'patch_mask' in model_output:
                has_masking = True
                # Get mask (pixel-level or expand from patch-level)
                if 'mask' in model_output:
                    mask = model_output['mask']
                else:
                    # Expand patch mask to pixel level
                    patch_mask = model_output['patch_mask']
                    mask = self._expand_patch_mask_to_pixels(patch_mask, target.shape[1])
                masked_acc = masked_reconstruction_accuracy(reconstruction, target, mask)
                total_masked_accuracy += masked_acc.item() * batch_size

            # Custom metrics
            for metric_name, metric_fn in self.metrics.items():
                if metric_name not in locals():
                    locals()[f'total_{metric_name}'] = 0.0
                metric_value = metric_fn(model_output, target)
                locals()[f'total_{metric_name}'] += metric_value.item() * batch_size

            total_samples += batch_size

        # Compute average metrics
        results = {
            'reconstruction_loss_mse': total_reconstruction_loss_mse / total_samples,
            'reconstruction_loss_l1': total_reconstruction_loss_l1 / total_samples,
            'latent_variance': total_latent_variance / total_samples,
        }

        if objective is not None:
            results['objective_loss'] = total_objective_loss / total_samples

        if has_masking:
            results['masked_loss'] = total_masked_loss / total_samples
            results['unmasked_loss'] = total_unmasked_loss / total_samples
            results['masked_accuracy'] = total_masked_accuracy / total_samples

        # Add custom metrics
        for metric_name in self.metrics.keys():
            results[metric_name] = locals()[f'total_{metric_name}'] / total_samples

        return results

    def evaluate_and_log(self, train_loader, val_loader, objective=None,
                        wandb_logger=None, step=None):
        """Evaluate on both train and validation sets and log results.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            objective: Objective function
            wandb_logger: Weights & Biases logger (optional)
            step: Current training step for logging

        Returns:
            Dictionary with 'train' and 'val' results
        """
        # Evaluate on training set
        train_results = self.evaluate(train_loader, objective, verbose=False)

        # Evaluate on validation set
        val_results = self.evaluate(val_loader, objective, verbose=False)

        # Prepare results
        results = {
            'train': train_results,
            'val': val_results
        }

        # Log to wandb if available
        if wandb_logger is not None:
            log_dict = {}
            for split in ['train', 'val']:
                for metric_name, metric_value in results[split].items():
                    log_dict[f'{split}/{metric_name}'] = metric_value

            if step is not None:
                wandb_logger.log(log_dict, step=step)
            else:
                wandb_logger.log(log_dict)

        return results
