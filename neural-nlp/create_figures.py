"""
Create publication-quality figures from brain encoding results.
Based on results from Blank2014, Fedorenko2016, and Pereira2018 benchmarks.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 150

# ============== DATA ==============

# Blank2014-linear benchmark
blank2014 = {
    'models': ['SmolLM2-135M', 'Pythia-70M', 'DistilGPT2', 'Gemma-2B'],
    'best_layer': ['L8', 'L2', 'Last', 'L2'],
    'score': [0.252472, 0.471598, 0.362869, 0.293944],
    'raw': [0.053086, 0.099161, 0.076299, 0.061806],
    'ceiling': 0.210266
}

# Fedorenko2016-linear benchmark
fedorenko2016 = {
    'models': ['SmolLM2-135M', 'Pythia-70M', 'DistilGPT2', 'Gemma-2B'],
    'best_layer': ['L8', 'Last', 'Last', 'Last'],
    'score': [0.194897, 0.554052, 0.554258, 0.982375],
    'raw': [0.043902, 0.124804, 0.124851, 0.221287],
    'ceiling': 0.225257
}

# Pereira2018-encoding benchmark
pereira2018 = {
    'models': ['GPT-2', 'Gemma-2B'],
    'score': [0.81589780, 0.97327393],
    'raw': [0.25991808, 0.34433388],
    'ceiling': [0.31856696, 0.3537892]
}

# Model sizes (approximate parameters in millions)
model_sizes = {
    'SmolLM2-135M': 135,
    'Pythia-70M': 70,
    'DistilGPT2': 82,
    'Gemma-2B': 2000,
    'GPT-2': 124
}

# Colors for models
colors = {
    'SmolLM2-135M': '#E74C3C',  # Red
    'Pythia-70M': '#3498DB',     # Blue
    'DistilGPT2': '#2ECC71',     # Green
    'Gemma-2B': '#9B59B6',       # Purple
    'GPT-2': '#F39C12'           # Orange
}

output_dir = Path('results/figures')
output_dir.mkdir(parents=True, exist_ok=True)


def fig1_benchmark_comparison():
    """
    Figure 1: Bar chart comparing normalized scores across all benchmarks.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Blank2014
    ax1 = axes[0]
    x = np.arange(len(blank2014['models']))
    bars = ax1.bar(x, blank2014['score'], color=[colors[m] for m in blank2014['models']],
                   edgecolor='black', linewidth=0.5)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Ceiling (normalized)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(blank2014['models'], rotation=45, ha='right')
    ax1.set_ylabel('Normalized Score')
    ax1.set_title('Blank2014\n(Language Network)', fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='upper right', fontsize=9)

    # Add value labels
    for bar, score in zip(bars, blank2014['score']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontsize=9)

    # Fedorenko2016
    ax2 = axes[1]
    bars = ax2.bar(x, fedorenko2016['score'], color=[colors[m] for m in fedorenko2016['models']],
                   edgecolor='black', linewidth=0.5)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Ceiling (normalized)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(fedorenko2016['models'], rotation=45, ha='right')
    ax2.set_ylabel('Normalized Score')
    ax2.set_title('Fedorenko2016\n(Language Selectivity)', fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper left', fontsize=9)

    for bar, score in zip(bars, fedorenko2016['score']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontsize=9)

    # Pereira2018
    ax3 = axes[2]
    x3 = np.arange(len(pereira2018['models']))
    bars = ax3.bar(x3, pereira2018['score'], color=[colors[m] for m in pereira2018['models']],
                   edgecolor='black', linewidth=0.5)
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Ceiling (normalized)')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(pereira2018['models'], rotation=45, ha='right')
    ax3.set_ylabel('Normalized Score')
    ax3.set_title('Pereira2018\n(Sentence Encoding)', fontweight='bold')
    ax3.set_ylim(0, 1.1)
    ax3.legend(loc='upper left', fontsize=9)

    for bar, score in zip(bars, pereira2018['score']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Brain Encoding Scores Across Benchmarks', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_benchmark_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_benchmark_comparison.pdf', bbox_inches='tight')
    print(f'[OK] Saved fig1_benchmark_comparison')
    plt.close()


def fig2_raw_vs_ceiling():
    """
    Figure 2: Raw scores vs ceiling - shows how close models get to theoretical maximum.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Fedorenko2016 - all models
    ax1 = axes[0]
    x = np.arange(len(fedorenko2016['models']))
    width = 0.35

    bars1 = ax1.bar(x - width/2, fedorenko2016['raw'], width,
                    color=[colors[m] for m in fedorenko2016['models']],
                    edgecolor='black', linewidth=0.5, label='Model Score')
    bars2 = ax1.bar(x + width/2, [fedorenko2016['ceiling']]*4, width,
                    color='lightgray', edgecolor='black', linewidth=0.5,
                    label='Noise Ceiling', hatch='///')

    ax1.set_xticks(x)
    ax1.set_xticklabels(fedorenko2016['models'], rotation=45, ha='right')
    ax1.set_ylabel('Raw Correlation (Pearson r)')
    ax1.set_title('Fedorenko2016: Raw Scores vs Ceiling', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 0.3)

    # Add percentage labels
    for i, (raw, ceil) in enumerate(zip(fedorenko2016['raw'], [fedorenko2016['ceiling']]*4)):
        pct = raw / ceil * 100
        ax1.text(i, raw + 0.01, f'{pct:.0f}%', ha='center', fontsize=9, fontweight='bold')

    # Pereira2018 - GPT-2 vs Gemma
    ax2 = axes[1]
    x2 = np.arange(len(pereira2018['models']))

    bars1 = ax2.bar(x2 - width/2, pereira2018['raw'], width,
                    color=[colors[m] for m in pereira2018['models']],
                    edgecolor='black', linewidth=0.5, label='Model Score')
    bars2 = ax2.bar(x2 + width/2, pereira2018['ceiling'], width,
                    color='lightgray', edgecolor='black', linewidth=0.5,
                    label='Noise Ceiling', hatch='///')

    ax2.set_xticks(x2)
    ax2.set_xticklabels(pereira2018['models'], rotation=45, ha='right')
    ax2.set_ylabel('Raw Correlation (Pearson r)')
    ax2.set_title('Pereira2018: Raw Scores vs Ceiling', fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, 0.45)

    # Add percentage labels
    for i, (raw, ceil) in enumerate(zip(pereira2018['raw'], pereira2018['ceiling'])):
        pct = raw / ceil * 100
        ax2.text(i, raw + 0.01, f'{pct:.0f}%', ha='center', fontsize=9, fontweight='bold')

    plt.suptitle('Model Performance Relative to Noise Ceiling', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_raw_vs_ceiling.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_raw_vs_ceiling.pdf', bbox_inches='tight')
    print(f'[OK] Saved fig2_raw_vs_ceiling')
    plt.close()


def fig3_model_size_vs_performance():
    """
    Figure 3: Scatter plot of model size vs performance across benchmarks.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Prepare data
    all_models = list(set(blank2014['models'] + fedorenko2016['models'] + pereira2018['models']))

    # Plot for each benchmark
    markers = {'Blank2014': 'o', 'Fedorenko2016': 's', 'Pereira2018': '^'}

    # Blank2014
    for model, score in zip(blank2014['models'], blank2014['score']):
        size = model_sizes[model]
        ax.scatter(size, score, c=colors[model], s=200, marker=markers['Blank2014'],
                  edgecolors='black', linewidths=1.5, zorder=5)

    # Fedorenko2016
    for model, score in zip(fedorenko2016['models'], fedorenko2016['score']):
        size = model_sizes[model]
        ax.scatter(size, score, c=colors[model], s=200, marker=markers['Fedorenko2016'],
                  edgecolors='black', linewidths=1.5, zorder=5)

    # Pereira2018
    for model, score in zip(pereira2018['models'], pereira2018['score']):
        size = model_sizes[model]
        ax.scatter(size, score, c=colors[model], s=200, marker=markers['Pereira2018'],
                  edgecolors='black', linewidths=1.5, zorder=5)

    ax.set_xscale('log')
    ax.set_xlabel('Model Size (Million Parameters)', fontsize=12)
    ax.set_ylabel('Normalized Brain Score', fontsize=12)
    ax.set_title('Model Size vs Brain Alignment', fontsize=14, fontweight='bold')

    # Create legend for benchmarks
    legend_benchmarks = [plt.Line2D([0], [0], marker=m, color='gray', linestyle='',
                                     markersize=10, label=b)
                        for b, m in markers.items()]

    # Create legend for models
    legend_models = [plt.Line2D([0], [0], marker='o', color=colors[m], linestyle='',
                                 markersize=10, label=m, markeredgecolor='black')
                    for m in all_models if m in colors]

    leg1 = ax.legend(handles=legend_benchmarks, loc='upper left', title='Benchmark')
    ax.add_artist(leg1)
    ax.legend(handles=legend_models, loc='lower right', title='Model')

    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Ceiling')
    ax.set_ylim(0, 1.15)
    ax.set_xlim(50, 3000)

    # Add annotations for key findings
    ax.annotate('Gemma-2B achieves\n98% on Fedorenko2016',
                xy=(2000, 0.98), xytext=(800, 0.75),
                arrowprops=dict(arrowstyle='->', color='purple'),
                fontsize=10, color='purple')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_size_vs_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_size_vs_performance.pdf', bbox_inches='tight')
    print(f'[OK] Saved fig3_size_vs_performance')
    plt.close()


def fig4_heatmap_summary():
    """
    Figure 4: Heatmap showing all results in a compact view.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create matrix
    all_models = ['SmolLM2-135M', 'Pythia-70M', 'DistilGPT2', 'GPT-2', 'Gemma-2B']
    benchmarks = ['Blank2014', 'Fedorenko2016', 'Pereira2018']

    # Fill matrix with scores (NaN for missing)
    matrix = np.full((len(all_models), len(benchmarks)), np.nan)

    for i, model in enumerate(all_models):
        if model in blank2014['models']:
            idx = blank2014['models'].index(model)
            matrix[i, 0] = blank2014['score'][idx]
        if model in fedorenko2016['models']:
            idx = fedorenko2016['models'].index(model)
            matrix[i, 1] = fedorenko2016['score'][idx]
        if model in pereira2018['models']:
            idx = pereira2018['models'].index(model)
            matrix[i, 2] = pereira2018['score'][idx]

    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Score', fontsize=11)

    # Set ticks
    ax.set_xticks(np.arange(len(benchmarks)))
    ax.set_yticks(np.arange(len(all_models)))
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.set_yticklabels(all_models, fontsize=11)

    # Add text annotations
    for i in range(len(all_models)):
        for j in range(len(benchmarks)):
            if not np.isnan(matrix[i, j]):
                text_color = 'white' if matrix[i, j] > 0.6 else 'black'
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                       fontsize=12, fontweight='bold', color=text_color)
            else:
                ax.text(j, i, '-', ha='center', va='center', fontsize=12, color='gray')

    ax.set_title('Brain Encoding Performance Summary\n(Normalized Scores)',
                fontsize=14, fontweight='bold')

    # Add model size annotations on the right
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(np.arange(len(all_models)))
    ax2.set_yticklabels([f'{model_sizes[m]}M' for m in all_models], fontsize=10)
    ax2.set_ylabel('Model Size', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_heatmap_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_heatmap_summary.pdf', bbox_inches='tight')
    print(f'[OK] Saved fig4_heatmap_summary')
    plt.close()


def fig5_gemma_highlight():
    """
    Figure 5: Special focus on Gemma-2B performance.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Data for Gemma-2B across benchmarks
    benchmarks = ['Blank2014', 'Fedorenko2016', 'Pereira2018']
    gemma_scores = [0.293944, 0.982375, 0.97327393]
    best_other = [0.471598, 0.554258, 0.81589780]  # Best non-Gemma score
    best_other_names = ['Pythia-70M', 'DistilGPT2', 'GPT-2']

    x = np.arange(len(benchmarks))
    width = 0.35

    bars1 = ax.bar(x - width/2, gemma_scores, width, label='Gemma-2B',
                   color=colors['Gemma-2B'], edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, best_other, width, label='Best Small Model',
                   color='lightblue', edgecolor='black', linewidth=1)

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Ceiling')

    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title('Gemma-2B vs Best Small Model per Benchmark', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1.15)

    # Add value labels and improvement percentages
    for i, (g, o, name) in enumerate(zip(gemma_scores, best_other, best_other_names)):
        ax.text(i - width/2, g + 0.02, f'{g:.2f}', ha='center', fontsize=10, fontweight='bold')
        ax.text(i + width/2, o + 0.02, f'{o:.2f}\n({name})', ha='center', fontsize=9)

        if g > o:
            improvement = (g - o) / o * 100
            ax.annotate(f'+{improvement:.0f}%', xy=(i, max(g, o) + 0.08),
                       fontsize=11, ha='center', color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_gemma_highlight.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_gemma_highlight.pdf', bbox_inches='tight')
    print(f'[OK] Saved fig5_gemma_highlight')
    plt.close()


if __name__ == '__main__':
    print('=' * 50)
    print('Creating Brain Encoding Figures')
    print('=' * 50)

    fig1_benchmark_comparison()
    fig2_raw_vs_ceiling()
    fig3_model_size_vs_performance()
    fig4_heatmap_summary()
    fig5_gemma_highlight()

    print('=' * 50)
    print(f'All figures saved to: {output_dir}')
    print('=' * 50)
