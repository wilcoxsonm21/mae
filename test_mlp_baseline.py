"""Test if a simple MLP can learn mean_intensity.

This tests the hypothesis that convolutional encoders lose global statistics
while MLPs can learn them directly from flattened pixels.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from data import get_dataset
from evaluation.downstream.metrics import regression_metrics

print("Testing Simple MLP Baseline for mean_intensity")
print("="*60)


class SimpleMLP(nn.Module):
    """Simple MLP for direct pixel -> mean_intensity prediction."""

    def __init__(self, input_dim=1024, hidden_dims=[256, 128, 64]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Generate dataset
print("Loading dataset...")
dataset = get_dataset(
    dataset_name='checkerboard',
    n_samples=10000,
    image_size=32,
    train_split=0.8,
    batch_size=128,
    normalize=True,
    random_state=42,
    grid_sizes=[2, 4, 8, 16],
    apply_transforms=True,
    rotation_range=15.0,
    scale_range=(0.8, 1.2),
    perspective_range=0.2,
    return_params=True
)

train_loader = dataset['train_loader']
val_loader = dataset['val_loader']
train_params = dataset['train_params']
val_params = dataset['val_params']

print(f"Train samples: {len(train_loader.dataset)}")
print(f"Val samples: {len(val_loader.dataset)}")
print(f"Mean intensity range: [{train_params['mean_intensity'].min():.6f}, {train_params['mean_intensity'].max():.6f}]")
print(f"Mean intensity std: {train_params['mean_intensity'].std():.6f}")

# Create MLP model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model = SimpleMLP(input_dim=1024, hidden_dims=[256, 128, 64]).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"MLP parameters: {total_params:,}")

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Training loop
num_epochs = 50
best_val_loss = float('inf')
patience = 10
patience_counter = 0

print(f"\nTraining for {num_epochs} epochs...")
print("="*60)

for epoch in range(1, num_epochs + 1):
    # Train
    model.train()
    train_loss = 0.0
    train_samples = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        batch_size = data.size(0)

        # Get targets
        batch_start = batch_idx * train_loader.batch_size
        batch_end = batch_start + batch_size
        targets = torch.from_numpy(train_params['mean_intensity'][batch_start:batch_end]).float().unsqueeze(1).to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(data)
        loss = criterion(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_size
        train_samples += batch_size

    train_loss /= train_samples

    # Validate
    model.eval()
    val_loss = 0.0
    val_samples = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(val_loader):
            data = data.to(device)
            batch_size = data.size(0)

            batch_start = batch_idx * val_loader.batch_size
            batch_end = batch_start + batch_size
            targets = torch.from_numpy(val_params['mean_intensity'][batch_start:batch_end]).float().unsqueeze(1).to(device)

            predictions = model(data)
            loss = criterion(predictions, targets)

            val_loss += loss.item() * batch_size
            val_samples += batch_size

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

    val_loss /= val_samples

    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = regression_metrics(all_predictions, all_targets)

    # Print progress
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:2d}/{num_epochs}: "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"R²: {metrics['r2']:.4f}, "
              f"Corr: {metrics['correlation']:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_metrics = metrics
        best_predictions = all_predictions.numpy()
        best_targets = all_targets.numpy()
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

# Final results
print("\n" + "="*60)
print("Final Results (Best Model)")
print("="*60)
print(f"Val Loss (MSE): {best_val_loss:.6f}")
print(f"MAE: {best_metrics['mae']:.6f}")
print(f"R²: {best_metrics['r2']:.4f}")
print(f"Correlation: {best_metrics['correlation']:.4f}")

# Analysis
print("\n" + "="*60)
print("Prediction Statistics")
print("="*60)
predictions_flat = best_predictions.flatten()
targets_flat = best_targets.flatten()

print(f"\nTargets:")
print(f"  Mean: {targets_flat.mean():.6f}")
print(f"  Std: {targets_flat.std():.6f}")
print(f"  Range: [{targets_flat.min():.6f}, {targets_flat.max():.6f}]")

print(f"\nPredictions:")
print(f"  Mean: {predictions_flat.mean():.6f}")
print(f"  Std: {predictions_flat.std():.6f}")
print(f"  Range: [{predictions_flat.min():.6f}, {predictions_flat.max():.6f}]")

print(f"\n" + "="*60)
print("Hypothesis Test Results")
print("="*60)

if best_metrics['r2'] > 0.8:
    print("✅ HYPOTHESIS CONFIRMED!")
    print("   Simple MLP can learn mean_intensity very well")
    print("   R² > 0.8 indicates strong learning")
    print("\n   Conclusion: The convolutional encoder LOSES global information")
    print("   The architecture is the problem, not the data or training setup")
elif best_metrics['r2'] > 0.5:
    print("✅ HYPOTHESIS SUPPORTED")
    print("   Simple MLP can learn mean_intensity reasonably well")
    print("   R² > 0.5 indicates moderate learning")
    print("\n   Conclusion: Convolutional encoder likely loses global information")
elif best_metrics['r2'] > 0.2:
    print("⚠️  HYPOTHESIS PARTIALLY SUPPORTED")
    print("   Simple MLP shows some learning but not great")
    print("   R² > 0.2 indicates weak learning")
    print("\n   Conclusion: Task is learnable but challenging")
else:
    print("❌ HYPOTHESIS REJECTED")
    print("   Simple MLP also struggles with mean_intensity")
    print("   R² < 0.2 indicates very weak learning")
    print("\n   Conclusion: Issue may not be architecture-specific")
    print("   Could be: data issue, task difficulty, or training problem")

# Check if predicting nearly constant
if predictions_flat.std() < targets_flat.std() * 0.5:
    print("\n⚠️  Warning: MLP predictions have low variance")
    print(f"   Prediction std: {predictions_flat.std():.6f}")
    print(f"   Target std: {targets_flat.std():.6f}")
    print(f"   Ratio: {predictions_flat.std() / targets_flat.std():.2f}")
else:
    print("\n✓ MLP predictions have good variance")
    print(f"   Prediction std: {predictions_flat.std():.6f}")
    print(f"   Target std: {targets_flat.std():.6f}")
    print(f"   Ratio: {predictions_flat.std() / targets_flat.std():.2f}")

print("\n" + "="*60)
print("Sample Predictions (first 10)")
print("="*60)
print("Target     Prediction  Error")
print("-" * 40)
for i in range(min(10, len(targets_flat))):
    error = predictions_flat[i] - targets_flat[i]
    print(f"{targets_flat[i]:+.6f}  {predictions_flat[i]:+.6f}  {error:+.6f}")
