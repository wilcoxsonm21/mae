"""Main training script for autoencoders."""

import os
import argparse
import yaml
import torch
import torch.optim as optim
import wandb
from pathlib import Path
from tqdm import tqdm

# Import models
from models import UNetAE, UNetMAE

# Import objectives
from objectives import ReconstructionLoss, MaskedReconstructionLoss

# Import data
from data import get_dataset

# Import evaluation
from evaluation import Evaluator, log_visualizations_to_wandb
from evaluation.downstream import ProbeTrainer
from evaluation.downstream.visualize import create_all_visualizations


def load_config(config_path):
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary of configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_model(config, image_size=None):
    """Create model based on config.

    Args:
        config: Configuration dictionary
        image_size: Size of square input images (for UNet models)

    Returns:
        Model instance
    """
    model_type = config['model']['type']
    model_params = config['model']['params'].copy()

    # Add image_size to params if needed and not already specified
    if image_size is not None and 'image_size' not in model_params:
        model_params['image_size'] = image_size

    if model_type == 'unet_ae':
        model = UNetAE(**model_params)
    elif model_type == 'unet_mae':
        model = UNetMAE(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def get_objective(config):
    """Create objective function based on config.

    Args:
        config: Configuration dictionary

    Returns:
        Objective instance
    """
    objective_type = config['objective']['type']
    objective_params = config['objective']['params']

    if objective_type == 'reconstruction':
        objective = ReconstructionLoss(**objective_params)
    elif objective_type == 'masked_reconstruction':
        objective = MaskedReconstructionLoss(**objective_params)
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")

    return objective


def train_epoch(model, train_loader, objective, optimizer, device, epoch, wandb_log=True):
    """Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        objective: Objective function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        wandb_log: Whether to log to wandb

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device)
        target = target.to(device)
        batch_size = data.size(0)

        # Forward pass
        optimizer.zero_grad()
        model_output = model(data)
        loss_output = objective(model_output, target)
        loss = loss_output['loss']

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Log to wandb
        if wandb_log and batch_idx % 10 == 0:
            # Calculate global step for proper chronological logging
            global_step = (epoch - 1) * len(train_loader) + batch_idx
            wandb.log({
                'train/batch_loss': loss.item(),
                'epoch': epoch
            }, step=global_step)

    avg_loss = total_loss / total_samples
    return avg_loss


def run_downstream_evaluation(model, train_loader, val_loader, train_params, val_params,
                              device, wandb_log=False, step=None, save_dir=None):
    """Run downstream evaluation of latent representations.

    Args:
        model: Trained model with encode() method
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        train_params: Training generation parameters
        val_params: Validation generation parameters
        device: Device to run on
        wandb_log: Whether to log to wandb
        step: Current training step for logging
        save_dir: Directory to save visualizations (optional)

    Returns:
        Dictionary of evaluation results
    """
    print("\n" + "="*60)
    print("Running Downstream Evaluation")
    print("="*60)

    # Create probe trainer
    probe_trainer = ProbeTrainer(model, device=device)

    # Train probes
    print("\nTraining downstream probes...")
    probe_trainer.train_probes(
        train_loader=train_loader,
        val_loader=val_loader,
        train_params=train_params,
        val_params=val_params,
        hidden_dim=64,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=100,
        batch_size=256,
        patience=15,
        verbose=True
    )

    # Evaluate probes
    print("\nEvaluating downstream probes...")
    results = probe_trainer.evaluate_probes(val_loader, val_params, verbose=True)

    # Get predictions for visualization
    predictions_dict = probe_trainer.get_predictions(val_loader, val_params)

    # Create visualizations
    print("\nCreating visualizations...")
    figures = create_all_visualizations(predictions_dict, results, save_dir=save_dir)

    # Log to wandb
    if wandb_log:
        log_dict = {}

        # Log metrics
        for task, task_results in results.items():
            if task == 'composite_score':
                log_dict['downstream/composite_score'] = task_results
            elif isinstance(task_results, dict):
                for metric_name, metric_value in task_results.items():
                    if not isinstance(metric_value, list):
                        log_dict[f'downstream/{task}_{metric_name}'] = metric_value

        # Log figures
        if figures:
            for fig_name, fig in figures.items():
                log_dict[f'downstream/{fig_name}'] = wandb.Image(fig)

        if step is not None:
            wandb.log(log_dict, step=step)
        else:
            wandb.log(log_dict)

    print("\n" + "="*60)
    print("Downstream Evaluation Complete")
    print("="*60 + "\n")

    return results


def train(config, wandb_log=True, use_hash_dir=False, base_results_dir='trained_models'):
    """Main training function.

    Args:
        config: Configuration dictionary
        wandb_log: Whether to log to wandb
        use_hash_dir: If True, save checkpoints using config hash in base_results_dir
        base_results_dir: Base directory for hash-based results storage

    Returns:
        Dictionary containing final results
    """
    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)

    # Setup device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Initialize wandb
    if wandb_log:
        wandb.init(
            project=config.get('wandb_project', 'autoencoder-training'),
            name=config.get('experiment_name', None),
            config=config
        )
        # Update config with sweep parameters (if running a sweep)
        # This ensures the config hash includes sweep modifications
        config.update(dict(wandb.config))

    # Get dataset
    print("Loading dataset...")
    dataset_config = config['dataset']
    data = get_dataset(**dataset_config)
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    input_dim = data['input_dim']
    image_size = data.get('image_size', None)  # Get image_size if available (for 2D images)
    train_params = data.get('train_params', None)  # Generation parameters for downstream eval
    val_params = data.get('val_params', None)
    print(f"Dataset loaded: input_dim={input_dim}")
    if image_size is not None:
        print(f"Image size: {image_size}x{image_size}")
    if train_params is not None:
        print(f"Generation parameters available for downstream evaluation")

    # Update model config with input_dim if not specified
    if 'input_dim' not in config['model']['params']:
        config['model']['params']['input_dim'] = input_dim

    # Create model
    print("Creating model...")
    model = get_model(config, image_size=image_size)
    model = model.to(device)
    print(f"Model: {config['model']['type']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create objective
    objective = get_objective(config)
    objective = objective.to(device)
    print(f"Objective: {config['objective']['type']}")

    # Create optimizer
    optimizer_config = config.get('optimizer', {'type': 'adam', 'lr': 1e-3})
    if optimizer_config['type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(optimizer_config.get('lr', 1e-3)),
            weight_decay=float(optimizer_config.get('weight_decay', 0.0))
        )
    elif optimizer_config['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(optimizer_config.get('lr', 1e-3)),
            momentum=float(optimizer_config.get('momentum', 0.9)),
            weight_decay=float(optimizer_config.get('weight_decay', 0.0))
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")

    # Create evaluator
    evaluator = Evaluator(model, device)

    # Log initial visualizations
    if wandb_log:
        print("Creating initial visualizations...")
        log_visualizations_to_wandb(model, train_loader, val_loader, device,
                                   image_size=image_size, step=0)

    # Training loop
    num_epochs = config.get('num_epochs', 100)
    best_val_loss = float('inf')

    # Determine checkpoint directory
    if use_hash_dir:
        from utils import get_experiment_dir
        checkpoint_dir = get_experiment_dir(config, base_results_dir)
    else:
        checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Visualization frequency (how often to log visualizations)
    vis_freq = config.get('visualization_frequency', 10)

    # Logging strategy: Use global_step for all wandb logs to ensure chronological ordering
    # - Batch logs: global_step = (epoch - 1) * len(train_loader) + batch_idx
    # - Epoch logs: global_step = epoch * len(train_loader)
    # This ensures batch logs within an epoch come before the epoch-level evaluation logs

    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        # Train for one epoch
        train_loss = train_epoch(
            model, train_loader, objective, optimizer,
            device, epoch, wandb_log=wandb_log
        )

        # Calculate global step for this epoch (for consistent wandb logging)
        global_step = epoch * len(train_loader)

        # Log train loss from training (for comparison with eval mode losses)
        if wandb_log:
            wandb.log({'train/loss_train_mode': train_loss}, step=global_step)

        # Evaluate (in eval mode)
        eval_results = evaluator.evaluate_and_log(
            train_loader, val_loader, objective,
            wandb_logger=wandb if wandb_log else None,
            step=global_step
        )

        val_loss = eval_results['val']['objective_loss']
        train_loss_eval_mode = eval_results['train']['objective_loss']

        # Print results
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train Loss (train mode): {train_loss:.4f}")
        print(f"  Train Loss (eval mode):  {train_loss_eval_mode:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Reconstruction (MSE): {eval_results['val']['reconstruction_loss_mse']:.4f}")

        # Log visualizations periodically
        if wandb_log and epoch % vis_freq == 0:
            print(f"  Logging visualizations at epoch {epoch}...")
            log_visualizations_to_wandb(model, train_loader, val_loader, device,
                                       image_size=image_size, step=global_step)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")

    # Save final model
    final_checkpoint_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config
    }, final_checkpoint_path)
    print(f"\nTraining complete! Final model saved to {final_checkpoint_path}")

    # Final evaluation
    print("\nFinal Evaluation:")
    final_global_step = num_epochs * len(train_loader)
    final_results = evaluator.evaluate_and_log(
        train_loader, val_loader, objective,
        wandb_logger=wandb if wandb_log else None,
        step=final_global_step
    )

    print("\nTrain Set:")
    for metric, value in final_results['train'].items():
        print(f"  {metric}: {value:.4f}")

    print("\nValidation Set:")
    for metric, value in final_results['val'].items():
        print(f"  {metric}: {value:.4f}")

    # Final visualizations
    if wandb_log:
        print("\nCreating final visualizations...")
        log_visualizations_to_wandb(model, train_loader, val_loader, device,
                                   image_size=image_size, step=final_global_step)

    # Downstream evaluation (if generation parameters available)
    downstream_results = None
    if train_params is not None and val_params is not None:
        downstream_eval_dir = checkpoint_dir / 'downstream_eval'
        downstream_eval_dir.mkdir(exist_ok=True)

        downstream_results = run_downstream_evaluation(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_params=train_params,
            val_params=val_params,
            device=device,
            wandb_log=wandb_log,
            step=final_global_step,
            save_dir=str(downstream_eval_dir)
        )

    if wandb_log:
        wandb.finish()

    # Save results using config hash if requested
    if use_hash_dir:
        from utils import save_results
        save_results(
            config,
            final_results,
            checkpoint_path,
            final_checkpoint_path,
            base_results_dir
        )

    return final_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train autoencoder models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Train
    train(config, wandb_log=not args.no_wandb)


if __name__ == '__main__':
    main()
