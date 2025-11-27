"""Diagnose why shapes dataset is not learning."""

import numpy as np
import torch
import torch.nn as nn
from data import get_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

print("="*60)
print("SHAPES DATASET DIAGNOSTIC")
print("="*60)

# Test 1: Load dataset and check basic stats
print("\nTest 1: Dataset Loading")
print("-"*60)

dataset = get_dataset(
    dataset_name='shapes',
    n_samples=1000,
    image_size=32,
    train_split=0.8,
    batch_size=128,
    normalize=False,  # No normalization first
    random_state=42,
    return_params=True
)

train_loader = dataset['train_loader']
val_loader = dataset['val_loader']
train_params = dataset['train_params']
val_params = dataset['val_params']

# Get all train data
X_train = train_loader.dataset.tensors[0].numpy()
y_train = train_params['shape']

# Get all val data
X_val = val_loader.dataset.tensors[0].numpy()
y_val = val_params['shape']

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
print(f"Train labels shape: {y_train.shape}, Val labels shape: {y_val.shape}")
print(f"Unique train labels: {np.unique(y_train)}")
print(f"Label distribution: {np.bincount(y_train)}")

# Test 2: Check RGB channel statistics
print("\nTest 2: RGB Channel Statistics (Without Normalization)")
print("-"*60)

# Reshape to get RGB channels
X_train_rgb = X_train.reshape(-1, 3, 32, 32)

print(f"Channel 0 (R) - mean: {X_train_rgb[:, 0].mean():.4f}, std: {X_train_rgb[:, 0].std():.4f}")
print(f"Channel 1 (G) - mean: {X_train_rgb[:, 1].mean():.4f}, std: {X_train_rgb[:, 1].std():.4f}")
print(f"Channel 2 (B) - mean: {X_train_rgb[:, 2].mean():.4f}, std: {X_train_rgb[:, 2].std():.4f}")

# Test 3: Can linear classifier solve this?
print("\nTest 3: Linear Classifier (Without Normalization)")
print("-"*60)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)

acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred, average='macro')

print(f"Linear classifier accuracy: {acc:.4f}")
print(f"Linear classifier F1-macro: {f1:.4f}")

if acc < 0.2:
    print("⚠️  WARNING: Even linear classifier fails! Data might be corrupted.")
elif acc < 0.5:
    print("⚠️  WARNING: Linear classifier struggles. Problem might be harder than expected.")
else:
    print("✓ Linear classifier works! Problem is learnable.")

# Test 4: Now with normalization
print("\nTest 4: WITH Normalization")
print("-"*60)

dataset_norm = get_dataset(
    dataset_name='shapes',
    n_samples=1000,
    image_size=32,
    train_split=0.8,
    batch_size=128,
    normalize=True,  # WITH normalization
    random_state=42,
    return_params=True
)

X_train_norm = dataset_norm['train_loader'].dataset.tensors[0].numpy()
X_val_norm = dataset_norm['val_loader'].dataset.tensors[0].numpy()

# Check stats after normalization
print(f"After normalization - mean: {X_train_norm.mean():.6f}, std: {X_train_norm.std():.6f}")

# Try linear classifier on normalized data
clf_norm = LogisticRegression(max_iter=1000, random_state=42)
clf_norm.fit(X_train_norm, y_train)
y_pred_norm = clf_norm.predict(X_val_norm)

acc_norm = accuracy_score(y_val, y_pred_norm)
f1_norm = f1_score(y_val, y_pred_norm, average='macro')

print(f"Linear classifier accuracy (normalized): {acc_norm:.4f}")
print(f"Linear classifier F1-macro (normalized): {f1_norm:.4f}")

if acc_norm < acc * 0.5:
    print("⚠️  CRITICAL: Normalization destroys learnability!")
else:
    print("✓ Normalization preserves information")

# Test 5: Simple neural network
print("\nTest 5: Simple Neural Network (No Normalization)")
print("-"*60)

class SimpleNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet(3072, 7).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train for 20 epochs
model.train()
for epoch in range(20):
    for batch_x, _ in train_loader:
        batch_x = batch_x.to(device)
        # Get labels for this batch
        batch_size = batch_x.size(0)
        batch_idx = epoch * len(train_loader.dataset) // batch_size
        start_idx = batch_idx % len(y_train)
        end_idx = min(start_idx + batch_size, len(y_train))

        batch_y = torch.from_numpy(y_train[start_idx:end_idx]).long().to(device)

        if len(batch_y) != batch_size:
            continue

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Evaluate
model.eval()
all_preds = []
with torch.no_grad():
    for batch_x, _ in val_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)

all_preds = np.array(all_preds[:len(y_val)])
acc_nn = accuracy_score(y_val, all_preds)
f1_nn = f1_score(y_val, all_preds, average='macro')

print(f"Simple NN accuracy: {acc_nn:.4f}")
print(f"Simple NN F1-macro: {f1_nn:.4f}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Linear (no norm):  acc={acc:.4f}, f1={f1:.4f}")
print(f"Linear (norm):     acc={acc_norm:.4f}, f1={f1_norm:.4f}")
print(f"Simple NN (no norm): acc={acc_nn:.4f}, f1={f1_nn:.4f}")

if acc > 0.5 and acc_nn > 0.5:
    print("\n✓ Dataset is learnable!")
    print("✓ Issue is likely with the supervised baseline architecture/training")
else:
    print("\n⚠️  Dataset itself has issues!")
