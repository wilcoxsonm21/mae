# MLP-based Autoencoder and Masked Autoencoder Framework

A flexible, modular framework for training MLP-based autoencoders (AE) and masked autoencoders (MAE) on synthetic datasets.

## Features

- **Modular Architecture**: Clean separation between models, training objectives, and data generation
- **Multiple Model Types**: Standard autoencoder (AE) and masked autoencoder (MAE)
- **Flexible Training Objectives**: Different loss functions with easy extensibility
- **Synthetic Data Generators**: Multiple synthetic datasets for quick experimentation
- **Comprehensive Evaluation**: Built-in metrics with easy-to-extend evaluation framework
- **Rich Visualizations**: Automatic logging of dataset samples, reconstructions, latent spaces, and error distributions to Weights & Biases
- **Hyperparameter Sweeps**: Automatic sweeping with caching, supports both quantitative and categorical parameters
- **YAML Configuration**: All experiments controlled via config files
- **Weights & Biases Integration**: Automatic logging and experiment tracking
- **Checkpointing**: Saves best and final models automatically

## Directory Structure

```
mae_ae_test/
├── configs/
│   ├── default.yaml              # Default configuration
│   ├── examples/
│   │   ├── ae_synthetic.yaml     # Example: AE on synthetic data
│   │   └── mae_synthetic.yaml    # Example: MAE on synthetic data
│   └── sweeps/
│       ├── latent_dim_sweep.yaml # Sweep over latent dimensions
│       ├── learning_rate_sweep.yaml
│       ├── activation_sweep.yaml # Categorical sweep
│       ├── loss_type_sweep.yaml
│       └── mask_ratio_sweep.yaml
├── data/
│   ├── generators.py             # Dataset generators
│   └── utils.py                  # Data utilities
├── models/
│   ├── base.py                   # Base architecture interface
│   ├── mlp_ae.py                 # Standard MLP autoencoder
│   └── mlp_mae.py                # MLP masked autoencoder
├── objectives/
│   ├── base.py                   # Base objective interface
│   ├── reconstruction.py         # Standard reconstruction loss
│   └── masked_reconstruction.py  # MAE-specific objective
├── evaluation/
│   ├── evaluator.py              # Evaluation framework
│   ├── metrics.py                # Metric functions
│   ├── visualization.py          # Visualization utilities
│   └── plotting.py               # Sweep result plotting
├── utils/
│   ├── config_hash.py            # Config hashing utilities
│   └── results.py                # Result storage/loading
├── main.py                       # Training script
├── sweep.py                      # Hyperparameter sweep runner
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Train a Standard Autoencoder

```bash
python main.py --config configs/examples/ae_synthetic.yaml
```

### Train a Masked Autoencoder

```bash
python main.py --config configs/examples/mae_synthetic.yaml
```

### Train without Weights & Biases logging

```bash
python main.py --config configs/default.yaml --no-wandb
```

## Configuration

All experiments are configured via YAML files. Here's a breakdown of the configuration options:

### Model Configuration

```yaml
model:
  type: "mlp_ae"  # Options: "mlp_ae", "mlp_mae"
  params:
    latent_dim: 32
    hidden_dims: [512, 256, 128]  # Hidden layer dimensions
    activation: "relu"  # Options: "relu", "tanh", "sigmoid"
    dropout: 0.0
    mask_ratio: 0.5  # Only for mlp_mae
```

### Training Objective

```yaml
objective:
  type: "reconstruction"  # Options: "reconstruction", "masked_reconstruction"
  params:
    loss_type: "mse"  # Options: "mse", "l1", "smooth_l1"
    reduction: "mean"  # Options: "mean", "sum"
```

### Dataset Configuration

```yaml
dataset:
  dataset_name: "gaussian_mixture"
  # Options: "gaussian_mixture", "swiss_roll", "s_curve",
  #          "concentric_circles", "uniform"
  n_samples: 10000
  input_dim: 64
  train_split: 0.8
  batch_size: 128
  normalize: true
  random_state: 42

  # Dataset-specific parameters:
  n_components: 5  # for gaussian_mixture
  n_circles: 3     # for concentric_circles
  noise: 0.1       # for swiss_roll, s_curve
```

### Available Datasets

1. **gaussian_mixture**: Mixture of Gaussian distributions in high dimensions
2. **swiss_roll**: Classic 3D Swiss roll manifold
3. **s_curve**: 3D S-curve manifold
4. **concentric_circles**: Concentric circles/spheres in high dimensions
5. **uniform**: Uniform random noise

## Models

### Standard Autoencoder (MLPAE)

A symmetric MLP-based autoencoder that learns to reconstruct input data through a compressed latent representation.

- **Use case**: General-purpose dimensionality reduction and data compression
- **Training**: Minimizes reconstruction error on entire input

### Masked Autoencoder (MLPMAE)

An autoencoder that learns to reconstruct randomly masked portions of the input, inspired by masked language models.

- **Use case**: Robust representation learning, denoising
- **Training**: Masks a portion of input features and learns to predict them
- **Advantage**: Can learn more robust representations by forcing the model to infer missing information

## Training Objectives

### Reconstruction Loss

Standard reconstruction loss that minimizes the difference between input and reconstruction.

- **loss_type**: 'mse', 'l1', or 'smooth_l1'
- **Use with**: MLPAE model

### Masked Reconstruction Loss

Specialized loss for masked autoencoders that computes loss only on masked positions.

- **loss_type**: 'mse', 'l1', or 'smooth_l1'
- **predict_all**: If False, only computes loss on masked positions
- **Use with**: MLPMAE model

## Evaluation Metrics

The framework automatically computes and logs:

- **reconstruction_loss_mse**: MSE reconstruction error
- **reconstruction_loss_l1**: L1 reconstruction error
- **latent_variance**: Variance of latent representations (measures latent space utilization)
- **masked_loss**: Reconstruction loss on masked positions (MAE only)
- **unmasked_loss**: Reconstruction loss on unmasked positions (MAE only)
- **masked_accuracy**: Fraction of masked positions reconstructed within threshold (MAE only)

### Adding Custom Metrics

You can easily add custom metrics by modifying the `Evaluator` class:

```python
from evaluation import Evaluator

evaluator = Evaluator(model, device)

# Add custom metric
def my_custom_metric(model_output, target):
    # Your metric computation
    return metric_value

evaluator.add_metric('my_metric', my_custom_metric)
```

## Visual Logging with Weights & Biases

The framework automatically logs comprehensive visualizations to Weights & Biases during training, making it easy to monitor your model's progress and understand what it's learning.

### What Gets Visualized

1. **Dataset Samples**
   - Scatter plots of training and validation data
   - 2D/3D plots for low-dimensional data
   - PCA projections for high-dimensional data

2. **Reconstructions**
   - Original vs reconstructed samples (side-by-side comparison)
   - Visualizes both train and validation sets
   - For MAE: Shows mask visualization (which features were masked)
   - Bar plots for low-dimensional data, heatmaps for high-dimensional data

3. **Latent Space**
   - Visualization of learned latent representations
   - Direct plots for 2D/3D latent spaces
   - PCA projections for higher-dimensional latent spaces
   - Helps assess if latent space is being used effectively

4. **Reconstruction Errors**
   - Per-sample error distribution (histogram)
   - Per-dimension error analysis (bar plot)
   - Identifies which samples/dimensions are harder to reconstruct

### Visualization Frequency

Control how often visualizations are logged with the `visualization_frequency` parameter:

```yaml
# In your config file
visualization_frequency: 10  # Log every 10 epochs
```

Visualizations are logged:
- At initialization (epoch 0) - shows untrained model performance
- Every N epochs during training (where N = `visualization_frequency`)
- At the end of training - final model performance

### Viewing Visualizations

After starting training, navigate to your Weights & Biases dashboard:

1. **Data samples**: `data/train_samples`, `data/val_samples`
2. **Reconstructions**: `reconstructions/train`, `reconstructions/val`
3. **Latent space**: `latent/train`, `latent/val`
4. **Errors**: `errors/train`, `errors/val`

### Disabling Visualizations

To train without wandb logging (no visualizations):

```bash
python main.py --config configs/default.yaml --no-wandb
```

### Tips for Using Visualizations

- **Early in training**: Reconstructions will be poor, latent space may be unstructured
- **During training**: Watch how reconstructions improve and latent space organizes
- **For MAE**: The mask visualization helps verify that masking is working correctly
- **Reconstruction errors**: High per-sample errors may indicate outliers in your data
- **Latent variance**: If all points cluster together in latent space, consider reducing `latent_dim`

## Checkpointing

Checkpoints are automatically saved to the directory specified in `checkpoint_dir`:

- **best_model.pt**: Model with lowest validation loss
- **final_model.pt**: Model at the end of training

Each checkpoint contains:
- Model state dictionary
- Optimizer state dictionary
- Epoch number
- Validation loss
- Full configuration

### Loading a Checkpoint

```python
import torch
from models import MLPAE

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')

# Create model from config
config = checkpoint['config']
model = MLPAE(**config['model']['params'])

# Load state
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Hyperparameter Sweeps

The framework includes a powerful sweep system that allows you to run experiments across different hyperparameter values with automatic caching and visualization.

### Features

- **Automatic Caching**: Results are saved using config hashes - if you run the same configuration twice, it uses cached results
- **Smart Storage**: All models and results stored in `trained_models/` organized by config hash
- **Quantitative & Categorical**: Supports both numeric sweeps (e.g., learning rate) and categorical sweeps (e.g., activation functions)
- **Automatic Plotting**: Generates comparison plots for all metrics
- **Result Summary**: Prints best configurations automatically

### Running a Sweep

1. **Create a base config** (or use an existing one)
2. **Create a sweep config** defining the parameter and values to sweep
3. **Run the sweep**

```bash
python sweep.py --config configs/examples/ae_synthetic.yaml \
                --sweep-config configs/sweeps/latent_dim_sweep.yaml \
                --plot
```

### Sweep Configuration Format

Create a YAML file specifying the parameter and values:

**Quantitative Sweep** (e.g., `latent_dim_sweep.yaml`):
```yaml
parameter: "model.params.latent_dim"
values: [8, 16, 32, 64, 128]
```

**Categorical Sweep** (e.g., `activation_sweep.yaml`):
```yaml
parameter: "model.params.activation"
values: ["relu", "tanh", "sigmoid"]
```

The `parameter` field uses dot notation to specify nested config values:
- `model.params.latent_dim` → Updates latent dimension
- `optimizer.lr` → Updates learning rate
- `model.params.activation` → Updates activation function
- `objective.params.loss_type` → Updates loss function

### Example Sweeps

The framework includes several example sweep configs:

**Quantitative Sweeps:**
- `latent_dim_sweep.yaml`: Sweep over latent space dimensions [8, 16, 32, 64, 128]
- `learning_rate_sweep.yaml`: Sweep over learning rates [0.0001, 0.0003, 0.001, 0.003, 0.01]
- `mask_ratio_sweep.yaml`: Sweep over MAE mask ratios [0.3, 0.4, 0.5, 0.6, 0.7]

**Categorical Sweeps:**
- `activation_sweep.yaml`: Compare activation functions ["relu", "tanh", "sigmoid"]
- `loss_type_sweep.yaml`: Compare loss functions ["mse", "l1", "smooth_l1"]

### Command Line Options

```bash
python sweep.py --config <base_config> \
                --sweep-config <sweep_config> \
                [--results-dir <dir>] \       # Default: trained_models
                [--wandb] \                    # Enable wandb logging
                [--force] \                    # Rerun even if cached
                [--plot]                       # Generate plots
```

### Caching and Config Hashing

Each experiment configuration is hashed to create a unique identifier. Results are stored in:
```
trained_models/
└── <config_hash>/
    ├── config.json          # Full configuration
    ├── results.json         # Final metrics
    ├── metadata.json        # Experiment metadata
    ├── best_model.pt        # Best model checkpoint
    └── final_model.pt       # Final model checkpoint
```

If you run the same config again, the framework automatically detects it and loads cached results instead of retraining. Use `--force` to override this behavior.

### Visualization

With `--plot` flag, the sweep generates:

1. **Individual Metric Plots**: Train and validation for each metric
   - `objective-loss.png`
   - `reconstruction-loss-mse.png`
   - `latent-variance.png`
   - etc.

2. **Combined Loss Plot**: Train vs validation loss on same plot
   - `combined_loss.png`

For **quantitative sweeps**, plots show line graphs with the best value highlighted.

For **categorical sweeps**, plots show bar charts for easy comparison.

### Example Workflow

```bash
# 1. Run a learning rate sweep
python sweep.py --config configs/examples/ae_synthetic.yaml \
                --sweep-config configs/sweeps/learning_rate_sweep.yaml \
                --plot

# Output shows:
# - Which experiments are cached vs newly run
# - Summary table of results
# - Best learning rate and its validation loss

# 2. Later, run the same sweep again (uses cache)
python sweep.py --config configs/examples/ae_synthetic.yaml \
                --sweep-config configs/sweeps/learning_rate_sweep.yaml \
                --plot

# All results loaded from cache instantly!

# 3. Try a different sweep
python sweep.py --config configs/examples/mae_synthetic.yaml \
                --sweep-config configs/sweeps/mask_ratio_sweep.yaml \
                --plot --wandb
```

### Creating Custom Sweeps

Create your own sweep config for any parameter:

```yaml
# configs/sweeps/my_custom_sweep.yaml
parameter: "model.params.dropout"
values: [0.0, 0.1, 0.2, 0.3, 0.5]
```

Then run:
```bash
python sweep.py --config configs/default.yaml \
                --sweep-config configs/sweeps/my_custom_sweep.yaml \
                --plot
```

### Tips for Sweeps

1. **Start with wider ranges**: Use logarithmic spacing for learning rates
2. **Check cached results**: Review `trained_models/` to see what's already computed
3. **Use --force sparingly**: Only when you've changed code and need to rerun
4. **Combine with wandb**: Use `--wandb` for detailed tracking of individual runs
5. **Quantitative vs Categorical**: The plotting automatically adapts to your sweep type

## Extending the Framework

### Adding a New Model Architecture

1. Create a new file in `models/` (e.g., `my_model.py`)
2. Inherit from `BaseAutoencoder` in `models/base.py`
3. Implement `encode()` and `decode()` methods
4. Add to `models/__init__.py`
5. Register in `main.py` `get_model()` function

### Adding a New Training Objective

1. Create a new file in `objectives/` (e.g., `my_objective.py`)
2. Inherit from `BaseObjective` in `objectives/base.py`
3. Implement `forward()` method
4. Add to `objectives/__init__.py`
5. Register in `main.py` `get_objective()` function

### Adding a New Dataset

1. Add generator function to `data/generators.py`
2. Follow the signature: `generate_X(n_samples, input_dim, ...)`
3. Add case to `get_dataset()` function
4. Document parameters in config

## Default Parameters

The framework uses sensible defaults:

- **Optimizer**: Adam (lr=1e-3)
- **Batch size**: 128
- **Epochs**: 100
- **Hidden dimensions**: [512, 256, 128]
- **Latent dimension**: 32
- **Activation**: ReLU
- **MAE mask ratio**: 0.5
- **Train/val split**: 80/20

## Tips for Best Results

1. **Normalize your data**: Set `normalize: true` in dataset config
2. **Start simple**: Begin with the default config and adjust gradually
3. **Monitor latent_variance**: Low values indicate underutilized latent space
4. **Experiment with mask_ratio**: For MAE, try values between 0.3-0.7
5. **Use appropriate hidden_dims**: Adjust based on input_dim and latent_dim
6. **Learning rate**: If training is unstable, reduce lr; if too slow, increase it

## Troubleshooting

### Training loss not decreasing
- Try reducing learning rate
- Check if data is normalized
- Ensure input_dim matches your dataset

### Model not using latent space (low latent_variance)
- Reduce latent_dim
- Add regularization (weight_decay)
- Increase model capacity (more/larger hidden layers)

### Out of memory errors
- Reduce batch_size
- Reduce n_samples
- Use smaller hidden_dims

## Citation

If you use this framework in your research, please cite:

```
@software{mae_ae_framework,
  title={MLP-based Autoencoder and Masked Autoencoder Framework},
  author={Your Name},
  year={2025}
}
```

## License

MIT License - feel free to use and modify for your own projects.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
