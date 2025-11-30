"""Create visualizations comparing model performance."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load results
df = pd.read_csv('model_comparison_results.csv')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Impact of Skip Connections on Latent Representation Quality',
             fontsize=16, fontweight='bold', y=0.995)

# Color scheme
colors = {
    'UNet AE (with skip)': '#e74c3c',  # Red
    'UNet MAE (with skip)': '#e67e22',  # Orange
    'CNN AE (no skip)': '#27ae60',  # Green
    'CNN MAE (no skip)': '#16a085'  # Teal
}

# 1. Composite Score Comparison
ax1 = axes[0, 0]
models = df['Model'].tolist()
scores = df['Composite Score'].tolist()
bars = ax1.barh(models, scores, color=[colors[m] for m in models])

# Add value labels
for i, (bar, score) in enumerate(zip(bars, scores)):
    width = bar.get_width()
    ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2,
             f'{score:.4f}', ha='left', va='center', fontweight='bold')

ax1.set_xlabel('Composite Score', fontsize=12, fontweight='bold')
ax1.set_title('Overall Performance (Composite Score)', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 1.0)

# Add improvement annotations
ax1.annotate('', xy=(0.9073, 0), xytext=(0.2206, 0),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax1.text(0.564, -0.3, '+311%', ha='center', fontsize=11, fontweight='bold', color='green')

ax1.annotate('', xy=(0.8908, 2), xytext=(0.4545, 2),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax1.text(0.673, 1.7, '+96%', ha='center', fontsize=11, fontweight='bold', color='green')

# 2. Task-wise Performance Heatmap
ax2 = axes[0, 1]
task_cols = ['grid_size_acc', 'rotation_r2', 'scale_r2', 'perspective_x_r2',
             'perspective_y_r2', 'mean_intensity_r2']
task_names = ['Grid Size\n(acc)', 'Rotation\n(R²)', 'Scale\n(R²)',
              'Perspective X\n(R²)', 'Perspective Y\n(R²)', 'Intensity\n(R²)']

heatmap_data = df[task_cols].values
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
            xticklabels=task_names, yticklabels=df['Model'],
            ax=ax2, vmin=0, vmax=1, cbar_kws={'label': 'Score'})
ax2.set_title('Performance Across All Tasks', fontsize=14, fontweight='bold')
ax2.set_xlabel('')
ax2.set_ylabel('')

# 3. Architecture Comparison (AE vs MAE)
ax3 = axes[1, 0]
architectures = ['UNet AE\n(with skip)', 'CNN AE\n(no skip)', 'UNet MAE\n(with skip)', 'CNN MAE\n(no skip)']
arch_scores = [0.2206, 0.9073, 0.4545, 0.8908]
arch_colors = ['#e74c3c', '#27ae60', '#e67e22', '#16a085']

bars = ax3.bar(range(len(architectures)), arch_scores, color=arch_colors, alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, arch_scores)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, height + 0.02,
             f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Add group labels
ax3.axvspan(-0.5, 1.5, alpha=0.1, color='red', label='With Skip Connections')
ax3.axvspan(1.5, 3.5, alpha=0.1, color='green', label='No Skip Connections')

ax3.set_ylabel('Composite Score', fontsize=12, fontweight='bold')
ax3.set_title('Architecture Impact on Representation Learning', fontsize=14, fontweight='bold')
ax3.set_xticks(range(len(architectures)))
ax3.set_xticklabels(architectures, fontsize=10)
ax3.set_ylim(0, 1.0)
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(axis='y', alpha=0.3)

# 4. Task-specific improvements
ax4 = axes[1, 1]
improvements = []
task_labels = []

for task_col, task_name in zip(task_cols, task_names):
    unet_ae = df.loc[df['Model'] == 'UNet AE (with skip)', task_col].values[0]
    cnn_ae = df.loc[df['Model'] == 'CNN AE (no skip)', task_col].values[0]

    if unet_ae > 0:
        improvement = ((cnn_ae - unet_ae) / unet_ae) * 100
    else:
        improvement = float('inf') if cnn_ae > 0 else 0

    improvements.append(improvement)
    task_labels.append(task_name.replace('\n(R²)', '').replace('\n(acc)', ''))

# Cap improvements for visualization
improvements_capped = [min(imp, 2000) for imp in improvements]
bars = ax4.barh(task_labels, improvements_capped,
               color=['green' if imp > 0 else 'red' for imp in improvements_capped])

# Add value labels
for i, (bar, imp, imp_capped) in enumerate(zip(bars, improvements, improvements_capped)):
    width = bar.get_width()
    if imp == float('inf'):
        label = '∞%'
    elif imp > 2000:
        label = f'>{imp_capped:.0f}%'
    else:
        label = f'+{imp:.1f}%'

    ax4.text(width + 50, bar.get_y() + bar.get_height()/2,
             label, ha='left', va='center', fontweight='bold', fontsize=10)

ax4.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
ax4.set_title('Per-Task Improvement: CNN AE vs UNet AE', fontsize=14, fontweight='bold')
ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax4.set_xlim(-100, max(improvements_capped) + 200)

plt.tight_layout()
plt.savefig('model_comparison_visualization.png', dpi=300, bbox_inches='tight')
print("Visualization saved to model_comparison_visualization.png")

# Create a simpler bar chart for composite scores
fig2, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(models))
bars = ax.bar(x, scores, color=[colors[m] for m in models], alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
            f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

# Add improvement arrows and labels
# UNet AE to CNN AE
ax.annotate('', xy=(2, 0.2206), xytext=(0, 0.2206),
           arrowprops=dict(arrowstyle='->', color='green', lw=3, linestyle='--'))
ax.text(1, 0.24, '+311%', ha='center', fontsize=13, fontweight='bold',
       color='green', bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', linewidth=2))

# UNet MAE to CNN MAE
ax.annotate('', xy=(3, 0.4545), xytext=(1, 0.4545),
           arrowprops=dict(arrowstyle='->', color='green', lw=3, linestyle='--'))
ax.text(2, 0.49, '+96%', ha='center', fontsize=13, fontweight='bold',
       color='green', bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', linewidth=2))

ax.set_ylabel('Composite Score', fontsize=14, fontweight='bold')
ax.set_title('Impact of Removing Skip Connections on Latent Representation Quality',
            fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', edgecolor='black', label='With Skip Connections'),
    Patch(facecolor='#27ae60', edgecolor='black', label='Without Skip Connections')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.9)

plt.tight_layout()
plt.savefig('composite_score_comparison.png', dpi=300, bbox_inches='tight')
print("Simple comparison saved to composite_score_comparison.png")

plt.show()
