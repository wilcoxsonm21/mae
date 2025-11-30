"""Generate configuration files for mask ratio experiment."""
import yaml
import numpy as np
from pathlib import Path

# Read base config
base_config_path = "configs/examples/cnn_mae_checkerboard.yaml"
with open(base_config_path, 'r') as f:
    base_config = yaml.safe_load(f)

# Create output directory for configs
config_dir = Path("experiments/mask_ratio_study/configs")
config_dir.mkdir(exist_ok=True)

# Generate mask ratios from 10% to 90% (9 experiments)
mask_ratios = np.linspace(0.1, 0.9, 9)

for mask_ratio in mask_ratios:
    # Create a copy of the base config
    config = base_config.copy()

    # Update experiment name and mask ratio
    config['experiment_name'] = f"cnn_mae_checkerboard_mask_{int(mask_ratio*100)}"
    config['model']['params']['mask_ratio'] = float(mask_ratio)
    config['checkpoint_dir'] = f"./checkpoints/mask_ratio_study/mask_{int(mask_ratio*100)}"

    # Save config file
    config_file = config_dir / f"mask_{int(mask_ratio*100)}.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated config for mask ratio {mask_ratio:.1f} -> {config_file}")

print(f"\nGenerated {len(mask_ratios)} configuration files")
print(f"Mask ratios: {[f'{mr:.1f}' for mr in mask_ratios]}")
