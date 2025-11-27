"""Supervised baseline for downstream tasks.

This script trains an encoder + probe jointly on downstream tasks using fully
supervised learning (no masking). This serves as a baseline to compare against
MAE pretraining + probe fine-tuning.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Import data
from data import get_dataset

# Import probe models
from evaluation.downstream.probes import LatentProbe
from evaluation.downstream.metrics import regression_metrics, classification_metrics, grid_size_to_class, compute_composite_score
from evaluation.downstream.visualize import create_all_visualizations


class SupervisedEncoder(nn.Module):
    """Encoder module extracted from UNet architecture.

    This is the same encoder used in UNetAE/UNetMAE, but standalone
    so it can be trained end-to-end with probes.
    """

    def __init__(self, image_size=32, base_channels=64, latent_dim=64, dropout=0.0, in_channels=1):
        """Initialize supervised encoder.

        Args:
            image_size: Size of square input images
            base_channels: Number of channels in first layer
            latent_dim: Dimension of output latent space
            dropout: Dropout probability
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        """
        super().__init__()

        self.image_size = image_size
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        # ConvBlock definition (same as in UNetAE)
        class ConvBlock(nn.Module):
            def __init__(self, in_channels, out_channels, dropout=0.0):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                if self.dropout is not None:
                    x = self.dropout(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)
                return x

        # Encoder (downsampling path)
        # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.enc1 = ConvBlock(in_channels, base_channels, dropout)  # 32x32
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2, dropout)  # 16x16
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, dropout)  # 8x8
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck at 4x4
        self.bottleneck_conv = ConvBlock(base_channels * 4, base_channels * 8, dropout)

        # Project to latent space
        bottleneck_size = (image_size // 8) * (image_size // 8) * base_channels * 8
        self.to_latent = nn.Linear(bottleneck_size, latent_dim)

    def forward(self, x):
        """Encode input to latent representation.

        Args:
            x: Input tensor of shape (batch_size, input_dim) - will be reshaped to (B, C, H, W)

        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        # Reshape to image format
        batch_size = x.shape[0]
        x = x.view(batch_size, self.in_channels, self.image_size, self.image_size)

        # Encoder path
        x = self.enc1(x)
        x = self.pool1(x)

        x = self.enc2(x)
        x = self.pool2(x)

        x = self.enc3(x)
        x = self.pool3(x)

        # Bottleneck
        x = self.bottleneck_conv(x)

        # Flatten and project to latent space
        x = x.view(batch_size, -1)
        z = self.to_latent(x)

        return z


class SupervisedModel(nn.Module):
    """Encoder + Probe trained end-to-end for downstream tasks."""

    def __init__(self, encoder, probe):
        """Initialize supervised model.

        Args:
            encoder: Encoder module
            probe: Probe module
        """
        super().__init__()
        self.encoder = encoder
        self.probe = probe

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input images (batch_size, input_dim)

        Returns:
            Predictions from probe
        """
        z = self.encoder(x)
        predictions = self.probe(z)
        return predictions


def prepare_targets(params_dict, task, device):
    """Prepare target tensors for a specific task.

    Args:
        params_dict: Dictionary with generation parameters
        task: Task name
        device: Device to put tensors on

    Returns:
        Target tensor
    """
    if task == 'grid_size':
        # Classification
        class_indices = grid_size_to_class(params_dict[task])
        return torch.from_numpy(class_indices).long().to(device)
    else:
        # Regression
        return torch.from_numpy(params_dict[task]).float().unsqueeze(1).to(device)


def train_epoch(model, train_loader, task, criterion, optimizer, device, epoch):
    """Train for one epoch.

    Args:
        model: Supervised model (encoder + probe)
        train_loader: Training DataLoader
        task: Task name
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        epoch: Current epoch number

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for data, targets in pbar:
        data = data.to(device)
        targets = targets.to(device)
        batch_size = data.size(0)

        # For regression tasks, need to unsqueeze targets to (batch_size, 1)
        if task not in ['grid_size', 'shape', 'color'] and len(targets.shape) == 1:
            targets = targets.unsqueeze(1)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(data)
        loss = criterion(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / total_samples
    return avg_loss


@torch.no_grad()
def evaluate(model, val_loader, task, criterion, device):
    """Evaluate model.

    Args:
        model: Supervised model
        val_loader: Validation DataLoader
        task: Task name
        criterion: Loss function
        device: Device

    Returns:
        Dictionary with loss and metrics
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []

    for data, targets in val_loader:
        data = data.to(device)
        targets = targets.to(device)
        batch_size = data.size(0)

        # For regression tasks, need to unsqueeze targets to (batch_size, 1)
        if task not in ['grid_size', 'shape', 'color'] and len(targets.shape) == 1:
            targets = targets.unsqueeze(1)

        # Forward pass
        predictions = model(data)
        loss = criterion(predictions, targets)

        # Track loss
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Store predictions and targets
        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())

    avg_loss = total_loss / total_samples

    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    if task in ['grid_size', 'shape', 'color']:
        # Classification metrics
        if task == 'shape':
            num_classes = 7
        elif task == 'color':
            num_classes = 6
        else:  # grid_size
            num_classes = 4
        metrics = classification_metrics(all_predictions, all_targets, num_classes=num_classes)
    else:
        # Regression metrics
        metrics = regression_metrics(all_predictions, all_targets)

    return {
        'loss': avg_loss,
        'metrics': metrics,
        'predictions': all_predictions.numpy(),
        'targets': all_targets.numpy()
    }


def train_supervised_baseline(config, wandb_log=True):
    """Train supervised baseline for a single task.

    Args:
        config: Configuration dictionary
        wandb_log: Whether to log to wandb

    Returns:
        Final evaluation results
    """
    # Set random seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Initialize wandb
    if wandb_log:
        wandb.init(
            project=config.get('wandb_project', 'supervised-baseline'),
            name=config.get('experiment_name', None),
            config=config
        )

    # Get task first (needed for dataset loading)
    task = config['task']
    print(f"\nTraining on task: {task}")

    # Get dataset
    print("Loading dataset...")
    dataset_config = config['dataset']
    dataset_config['supervised_task'] = task  # Add task for supervised learning
    data = get_dataset(**dataset_config)
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    image_size = data.get('image_size', 32)

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Image size: {image_size}x{image_size}")

    # Create encoder
    encoder_config = config.get('encoder', {})
    latent_dim = encoder_config.get('latent_dim', 64)
    base_channels = encoder_config.get('base_channels', 64)
    dropout = encoder_config.get('dropout', 0.0)
    in_channels = encoder_config.get('in_channels', 1)  # Default to 1 for grayscale

    print("\nCreating encoder...")
    encoder = SupervisedEncoder(
        image_size=image_size,
        base_channels=base_channels,
        latent_dim=latent_dim,
        dropout=dropout,
        in_channels=in_channels
    )

    # Create probe
    probe_config = config.get('probe', {})
    hidden_dim = probe_config.get('hidden_dim', 64)
    probe_dropout = probe_config.get('dropout', 0.1)

    is_classification = task in ['grid_size', 'shape', 'color']

    # Determine output dimension
    if task == 'grid_size':
        output_dim = 4  # 4 grid sizes
    elif task == 'shape':
        output_dim = 7  # 7 shapes
    elif task == 'color':
        output_dim = 6  # 6 colors
    else:
        output_dim = 1  # Regression tasks

    print("Creating probe...")
    probe = LatentProbe(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        task_type='classification' if is_classification else 'regression',
        dropout=probe_dropout
    )

    # Create full model
    model = SupervisedModel(encoder, probe).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in encoder.parameters())
    probe_params = sum(p.numel() for p in probe.parameters())

    print(f"\nModel parameters:")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Probe: {probe_params:,}")
    print(f"  Total: {total_params:,}")

    # Loss function
    if is_classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    # Optimizer
    optimizer_config = config.get('optimizer', {})
    lr = optimizer_config.get('lr', 1e-3)
    weight_decay = optimizer_config.get('weight_decay', 1e-4)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    num_epochs = config.get('num_epochs', 100)
    patience = config.get('patience', 15)

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Task: {task}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Early stopping patience: {patience}")

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    # Create checkpoint directory
    checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints/supervised_baseline'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, task,
            criterion, optimizer, device, epoch
        )

        # Evaluate
        val_results = evaluate(
            model, val_loader, task,
            criterion, device
        )
        val_loss = val_results['loss']

        # Print results
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")

        # Print metrics
        for metric_name, metric_value in val_results['metrics'].items():
            if not isinstance(metric_value, list):
                print(f"  {metric_name}: {metric_value:.4f}")

        # Log to wandb
        if wandb_log:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
            }
            for metric_name, metric_value in val_results['metrics'].items():
                if not isinstance(metric_value, list):
                    log_dict[f'val/{metric_name}'] = metric_value
            wandb.log(log_dict)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': encoder.state_dict(),
                'probe_state_dict': probe.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_results': val_results,
                'config': config
            }
            # Save best model
            checkpoint_path = checkpoint_dir / f'best_model_{task}.pt'
            torch.save(best_state, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Load best model for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state['model_state_dict'])

    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation on Best Model")
    print("="*60)

    final_results = evaluate(
        model, val_loader, task,
        criterion, device
    )

    print(f"\nTask: {task}")
    print(f"Val Loss: {final_results['loss']:.4f}")
    for metric_name, metric_value in final_results['metrics'].items():
        if isinstance(metric_value, list):
            print(f"{metric_name}: {metric_value}")
        else:
            print(f"{metric_name}: {metric_value:.4f}")

    # Log final results to wandb
    if wandb_log:
        log_dict = {
            'final/val_loss': final_results['loss'],
        }
        for metric_name, metric_value in final_results['metrics'].items():
            if not isinstance(metric_value, list):
                log_dict[f'final/{metric_name}'] = metric_value
        wandb.log(log_dict)
        wandb.finish()

    return final_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train supervised baseline for downstream tasks')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')
    parser.add_argument('--task', type=str, default=None,
                       help='Override task from config (rotation, scale, perspective_x, perspective_y, grid_size)')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override task if specified
    if args.task is not None:
        config['task'] = args.task

    # Train
    train_supervised_baseline(config, wandb_log=not args.no_wandb)


if __name__ == '__main__':
    main()
