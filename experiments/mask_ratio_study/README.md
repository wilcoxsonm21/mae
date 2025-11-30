# Mask Ratio Study for CNN MAE

This experiment studies the effect of mask ratio on the performance of CNN Masked Autoencoder (MAE) on the checkerboard dataset.

## Overview

This study investigates how different mask ratios (10% to 90%) affect:
- **Reconstruction performance**: Train and validation loss
- **Downstream task performance**: Composite score across all downstream tasks
- **Individual task correlations**: Rotation, scale, perspective predictions
- **Classification accuracy**: Grid size classification

## Experiment Structure

```
experiments/mask_ratio_study/
├── README.md                           # This file
├── generate_configs.py                 # Generates config files for each mask ratio
├── run_experiments.py                  # Runs all experiments sequentially
├── extract_and_plot_results.py        # Extracts results and creates plots
├── configs/                            # Generated config files
│   ├── mask_10.yaml
│   ├── mask_20.yaml
│   ├── ...
│   └── mask_90.yaml
├── mask_ratio_analysis.png             # Generated plot (after running)
└── mask_ratio_results.csv              # Generated results table (after running)
```

## Usage

### 1. Generate Configuration Files

```bash
cd /home/mwil/mae
python experiments/mask_ratio_study/generate_configs.py
```

This creates 9 config files with mask ratios evenly spaced from 10% to 90%.

### 2. Run All Experiments

```bash
python experiments/mask_ratio_study/run_experiments.py
```

This will:
- Run training for each mask ratio configuration
- Save checkpoints to `checkpoints/mask_ratio_study/mask_XX/`
- Log metrics to wandb
- Run downstream evaluation for each model
- Save downstream results to JSON files

**Note**: This will take considerable time as it trains 9 models sequentially. Each model trains for 20 epochs with downstream evaluation.

### 3. Extract Results and Generate Plots

```bash
python experiments/mask_ratio_study/extract_and_plot_results.py
```

This will:
- Load checkpoints from all experiments
- Extract train/val losses and downstream metrics
- Generate a comprehensive visualization with 6 subplots
- Save results to CSV for further analysis

## Output Files

After running the full pipeline, you'll have:

1. **Checkpoints**: `checkpoints/mask_ratio_study/mask_XX/`
   - `best_model.pt` - Best model checkpoint with losses
   - `final_model.pt` - Final model checkpoint
   - `downstream_eval/results.json` - Downstream evaluation metrics

2. **Visualizations**: `experiments/mask_ratio_study/mask_ratio_analysis.png`
   - 6-panel plot showing all metrics vs mask ratio

3. **Data**: `experiments/mask_ratio_study/mask_ratio_results.csv`
   - CSV file with all metrics for each mask ratio

## Metrics Tracked

### Reconstruction Metrics
- **Train Loss**: MSE reconstruction loss on training set
- **Val Loss**: MSE reconstruction loss on validation set

### Downstream Metrics
- **Composite Score**: Weighted combination of avg R² and grid size accuracy
- **Grid Size Accuracy**: Classification accuracy for grid size (2, 4, 8, 16)
- **Rotation Correlation**: Pearson correlation for rotation angle prediction
- **Scale Correlation**: Pearson correlation for scale factor prediction
- **Perspective X/Y Correlation**: Pearson correlation for perspective distortion prediction

## Expected Results

The study aims to answer:
- **What is the optimal mask ratio?** Which ratio gives best downstream performance?
- **Trade-off analysis**: How does reconstruction loss vs downstream performance change?
- **Task sensitivity**: Are some tasks more sensitive to mask ratio than others?
- **Overfitting detection**: Does higher mask ratio reduce overfitting?

## Customization

To modify the experiment:

1. **Change mask ratios**: Edit `generate_configs.py` line with `np.linspace(0.1, 0.9, 9)`
2. **Modify base config**: The script reads from `configs/examples/cnn_mae_checkerboard.yaml`
3. **Add more metrics**: Update `extract_and_plot_results.py` to include additional metrics

## Notes

- Each experiment uses the same random seed (42) for reproducibility
- All experiments use the same dataset split
- Downstream evaluation uses MLP probes with 100 epochs and early stopping
- Results are automatically saved to JSON for easy extraction
