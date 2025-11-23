"""Trainer for downstream evaluation probes."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np

from .probes import LatentProbe, MultiTaskLatentProbe
from .metrics import regression_metrics, classification_metrics, grid_size_to_class, compute_composite_score


class ProbeTrainer:
    """Handles training and evaluation of downstream probes."""

    def __init__(self, encoder, device='cpu', latent_dim=None):
        """Initialize probe trainer.

        Args:
            encoder: Trained encoder model (or full autoencoder with .encode() method)
            device: Device to run on
            latent_dim: Latent dimension (inferred if None)
        """
        self.encoder = encoder
        self.device = device
        self.latent_dim = latent_dim

        # Storage for probes
        self.probes = {}

        # Task configuration
        self.regression_tasks = ['rotation', 'scale', 'perspective_x', 'perspective_y']
        self.classification_tasks = ['grid_size']

    @torch.no_grad()
    def extract_latents(self, dataloader):
        """Extract latent codes from encoder.

        Args:
            dataloader: DataLoader with image data

        Returns:
            Tensor of latent codes (n_samples, latent_dim)
        """
        self.encoder.eval()
        latents = []

        for batch, _ in dataloader:
            batch = batch.to(self.device)

            # Get latent codes
            if hasattr(self.encoder, 'encode'):
                z = self.encoder.encode(batch)
            else:
                # Assume encoder is the full model and returns dict with 'latent'
                output = self.encoder(batch)
                z = output['latent']

            latents.append(z.cpu())

        latents = torch.cat(latents, dim=0)

        # Infer latent_dim if not set
        if self.latent_dim is None:
            self.latent_dim = latents.shape[1]

        return latents

    def prepare_targets(self, params_dict):
        """Prepare target tensors from parameter dictionary.

        Args:
            params_dict: Dictionary with generation parameters

        Returns:
            Dictionary of target tensors
        """
        targets = {}

        # Regression targets
        for task in self.regression_tasks:
            if task in params_dict:
                targets[task] = torch.from_numpy(params_dict[task]).float().unsqueeze(1)

        # Classification target (grid_size)
        if 'grid_size' in params_dict:
            class_indices = grid_size_to_class(params_dict['grid_size'])
            targets['grid_size'] = torch.from_numpy(class_indices).long()

        return targets

    def train_single_probe(self, task, train_latents, train_targets, val_latents, val_targets,
                          hidden_dim=64, lr=1e-3, weight_decay=1e-4, epochs=100,
                          batch_size=256, patience=15, verbose=False):
        """Train a single probe for one task.

        Args:
            task: Task name
            train_latents: Training latent codes
            train_targets: Training targets
            val_latents: Validation latent codes
            val_targets: Validation targets
            hidden_dim: Hidden dimension of probe
            lr: Learning rate
            weight_decay: L2 regularization
            epochs: Maximum epochs
            batch_size: Batch size
            patience: Early stopping patience
            verbose: Whether to show progress

        Returns:
            Trained probe model
        """
        # Determine task type
        is_classification = task in self.classification_tasks
        task_type = 'classification' if is_classification else 'regression'

        # Create probe
        if is_classification:
            output_dim = 4  # Grid sizes: 2, 4, 8, 16
        else:
            output_dim = 1

        probe = LatentProbe(
            latent_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            task_type=task_type
        ).to(self.device)

        # Loss function
        if is_classification:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        # Optimizer
        optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

        # Create dataloaders
        train_dataset = TensorDataset(train_latents, train_targets)
        val_dataset = TensorDataset(val_latents, val_targets)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        iterator = tqdm(range(epochs), desc=f"Training {task} probe") if verbose else range(epochs)

        for epoch in iterator:
            # Train
            probe.train()
            train_loss = 0.0
            for batch_latents, batch_targets in train_loader:
                batch_latents = batch_latents.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                predictions = probe(batch_latents)

                if is_classification:
                    loss = criterion(predictions, batch_targets)
                else:
                    loss = criterion(predictions, batch_targets)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_latents.size(0)

            train_loss /= len(train_dataset)

            # Validate
            probe.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_latents, batch_targets in val_loader:
                    batch_latents = batch_latents.to(self.device)
                    batch_targets = batch_targets.to(self.device)

                    predictions = probe(batch_latents)

                    if is_classification:
                        loss = criterion(predictions, batch_targets)
                    else:
                        loss = criterion(predictions, batch_targets)

                    val_loss += loss.item() * batch_latents.size(0)

            val_loss /= len(val_dataset)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = probe.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        # Load best model
        probe.load_state_dict(best_state)

        return probe

    def train_probes(self, train_loader, val_loader, train_params, val_params,
                    hidden_dim=64, lr=1e-3, weight_decay=1e-4, epochs=100,
                    batch_size=256, patience=15, verbose=True):
        """Train probes for all tasks.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            train_params: Training generation parameters
            val_params: Validation generation parameters
            hidden_dim: Hidden dimension for probes
            lr: Learning rate
            weight_decay: L2 regularization
            epochs: Maximum epochs
            batch_size: Batch size for probe training
            patience: Early stopping patience
            verbose: Whether to show progress

        Returns:
            Dictionary of trained probes
        """
        if verbose:
            print("Extracting latent codes...")

        # Extract latent codes
        train_latents = self.extract_latents(train_loader)
        val_latents = self.extract_latents(val_loader)

        if verbose:
            print(f"Latent dimension: {self.latent_dim}")
            print(f"Train latents: {train_latents.shape}")
            print(f"Val latents: {val_latents.shape}")

        # Prepare targets
        train_targets_dict = self.prepare_targets(train_params)
        val_targets_dict = self.prepare_targets(val_params)

        # Train probe for each task
        all_tasks = self.regression_tasks + self.classification_tasks

        for task in all_tasks:
            if task not in train_targets_dict:
                if verbose:
                    print(f"Skipping {task} (no data)")
                continue

            if verbose:
                print(f"\nTraining probe for: {task}")

            probe = self.train_single_probe(
                task=task,
                train_latents=train_latents,
                train_targets=train_targets_dict[task],
                val_latents=val_latents,
                val_targets=val_targets_dict[task],
                hidden_dim=hidden_dim,
                lr=lr,
                weight_decay=weight_decay,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                verbose=verbose
            )

            self.probes[task] = probe

        return self.probes

    @torch.no_grad()
    def evaluate_probes(self, val_loader, val_params, verbose=True):
        """Evaluate all trained probes.

        Args:
            val_loader: Validation DataLoader
            val_params: Validation generation parameters
            verbose: Whether to print results

        Returns:
            Dictionary of evaluation metrics for each task
        """
        if not self.probes:
            raise ValueError("No probes have been trained yet. Call train_probes() first.")

        # Extract latents
        val_latents = self.extract_latents(val_loader).to(self.device)
        val_targets_dict = self.prepare_targets(val_params)

        results = {}

        # Evaluate each probe
        for task, probe in self.probes.items():
            probe.eval()

            if task not in val_targets_dict:
                continue

            targets = val_targets_dict[task].to(self.device)
            predictions = probe(val_latents)

            # Compute metrics
            if task in self.classification_tasks:
                metrics = classification_metrics(predictions, targets)
            else:
                metrics = regression_metrics(predictions, targets)

            results[task] = metrics

            if verbose:
                print(f"\n{task}:")
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, list):
                        print(f"  {metric_name}: {metric_value}")
                    else:
                        print(f"  {metric_name}: {metric_value:.4f}")

        # Compute composite score
        composite = compute_composite_score(results)
        results['composite_score'] = composite

        if verbose:
            print(f"\nComposite Score: {composite:.4f}")

        return results

    @torch.no_grad()
    def get_predictions(self, val_loader, val_params):
        """Get predictions from all probes for visualization.

        Args:
            val_loader: Validation DataLoader
            val_params: Validation generation parameters

        Returns:
            Dictionary with predictions and targets for each task
        """
        if not self.probes:
            raise ValueError("No probes have been trained yet.")

        # Extract latents
        val_latents = self.extract_latents(val_loader).to(self.device)
        val_targets_dict = self.prepare_targets(val_params)

        predictions_dict = {}

        for task, probe in self.probes.items():
            probe.eval()

            if task not in val_targets_dict:
                continue

            targets = val_targets_dict[task]
            predictions = probe(val_latents).cpu()

            predictions_dict[task] = {
                'predictions': predictions.numpy(),
                'targets': targets.numpy()
            }

        return predictions_dict
