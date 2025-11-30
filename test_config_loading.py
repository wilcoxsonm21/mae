"""Test that transformer models can be loaded through config files."""

import yaml
import torch
from main import get_model, get_objective, get_dataset

def test_config_loading(config_path):
    """Test loading a model through its config file."""
    print(f"\nTesting config: {config_path}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"  Model type: {config['model']['type']}")

    # Load dataset to get input_dim (like the real training code does)
    dataset_config = config['dataset']
    data = get_dataset(**dataset_config)
    input_dim = data['input_dim']
    image_size = data.get('image_size', None)

    # Add input_dim to model config if not specified (like main.py does)
    if 'input_dim' not in config['model']['params']:
        config['model']['params']['input_dim'] = input_dim

    # Create model
    model = get_model(config, image_size=image_size)
    print(f"  ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create objective
    objective = get_objective(config)
    print(f"  ✓ Objective created: {config['objective']['type']}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, input_dim)

    with torch.no_grad():
        output = model(x)

    # Check output format
    assert 'reconstruction' in output, "Missing 'reconstruction' in output"
    assert 'latent' in output, "Missing 'latent' in output"

    print(f"  ✓ Forward pass successful")
    print(f"    - reconstruction shape: {output['reconstruction'].shape}")
    print(f"    - latent shape: {output['latent'].shape}")

    # Test with objective
    with torch.no_grad():
        loss_output = objective(output, x)

    # Handle both dict and tensor returns
    if isinstance(loss_output, dict):
        loss_value = loss_output.get('total_loss', loss_output.get('loss', list(loss_output.values())[0]))
    else:
        loss_value = loss_output

    print(f"  ✓ Objective computed successfully: loss = {loss_value.item():.4f}")

    return True

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Transformer Model Config Loading")
    print("=" * 70)

    configs = [
        "configs/examples/transformer_ae_checkerboard.yaml",
        "configs/examples/transformer_mae_checkerboard.yaml"
    ]

    try:
        for config_path in configs:
            test_config_loading(config_path)

        print("\n" + "=" * 70)
        print("✓ All config loading tests passed!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
