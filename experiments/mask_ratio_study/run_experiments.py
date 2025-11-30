"""Run all mask ratio experiments sequentially."""
import subprocess
import sys
from pathlib import Path
import time

# Get all config files
config_dir = Path("experiments/mask_ratio_study/configs")
config_files = sorted(config_dir.glob("mask_*.yaml"))

print(f"Found {len(config_files)} experiments to run")
print("=" * 60)

failed_experiments = []
successful_experiments = []

for i, config_file in enumerate(config_files, 1):
    mask_ratio = config_file.stem.replace("mask_", "")
    print(f"\n[{i}/{len(config_files)}] Running experiment with mask ratio {mask_ratio}%")
    print("-" * 60)

    start_time = time.time()

    # Run the experiment
    cmd = f"python main.py --config {config_file}"
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                              capture_output=False, text=True)
        elapsed = time.time() - start_time
        print(f"✓ Completed in {elapsed/60:.1f} minutes")
        successful_experiments.append(config_file.name)
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ Failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        failed_experiments.append(config_file.name)
        # Continue with next experiment
        continue

print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)
print(f"Successful: {len(successful_experiments)}/{len(config_files)}")
for exp in successful_experiments:
    print(f"  ✓ {exp}")

if failed_experiments:
    print(f"\nFailed: {len(failed_experiments)}/{len(config_files)}")
    for exp in failed_experiments:
        print(f"  ✗ {exp}")

print("\nAll experiments completed!")
