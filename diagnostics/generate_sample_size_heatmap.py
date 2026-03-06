#!/usr/bin/env python3
"""
Generate sample size heatmap showing n for each material-technology pair.

Creates a publication-quality heatmap with materials on Y-axis and
technologies on X-axis, with cell colors/values showing sample size.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.technology_mapping import TECHNOLOGY_CONSOLIDATION


def bin_sample_size(n):
    """Assign sample size to a bin."""
    if n == 1:
        return 'n=1'
    elif n == 2:
        return 'n=2'
    elif n == 3:
        return 'n=3'
    elif 4 <= n <= 5:
        return '4≤n≤5'
    elif 6 <= n <= 10:
        return '6≤n≤10'
    elif 11 <= n <= 20:
        return '11≤n≤20'
    else:
        return 'n>20'


def load_intensity_data():
    """Load and preprocess intensity data with technology consolidation."""
    data_path = ROOT / 'data' / 'intensity_data.csv'
    df = pd.read_csv(data_path)

    # Standardize column names
    df.columns = ['Technology', 'Material', 'g_per_MW']

    # Apply technology consolidation (CDTE→CdTe, ASIGE→a-Si)
    df['Technology'] = df['Technology'].replace(TECHNOLOGY_CONSOLIDATION)

    return df


def create_sample_size_heatmap():
    """Create heatmap of sample sizes by material and technology."""
    print("Creating sample size heatmap...")

    df = load_intensity_data()

    # Count sample sizes per material-tech pair
    pair_counts = df.groupby(['Material', 'Technology']).size().reset_index(name='n')

    # Create pivot table for heatmap
    heatmap_data = pair_counts.pivot(index='Material', columns='Technology', values='n')

    # Sort materials by total observations (descending)
    material_totals = heatmap_data.sum(axis=1).sort_values(ascending=False)
    heatmap_data = heatmap_data.loc[material_totals.index]

    # Sort technologies by total observations (descending)
    tech_totals = heatmap_data.sum(axis=0).sort_values(ascending=False)
    heatmap_data = heatmap_data[tech_totals.index]

    # Define bin order and colors (matching other visualizations)
    bin_order = ['n=1', 'n=2', 'n=3', '4≤n≤5', '6≤n≤10', '11≤n≤20', 'n>20']
    bin_labels = ['1', '2', '3', '4-5', '6-10', '11-20', '>20']  # Simplified labels for colorbar
    colors = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4', '#313695']

    # Create a discrete colormap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(colors)

    # Define boundaries for bins: 0-1, 1-2, 2-3, 3-5, 5-10, 10-20, 20+
    # Use max value + 1 instead of infinity for the upper boundary
    max_val = heatmap_data.max().max()
    boundaries = [0.5, 1.5, 2.5, 3.5, 5.5, 10.5, 20.5, max_val + 0.5]
    norm = BoundaryNorm(boundaries, cmap.N)

    # Create figure with larger size for better spacing
    fig, ax = plt.subplots(figsize=(18, 14))

    # Plot heatmap with discrete colors
    im = ax.imshow(heatmap_data.values, cmap=cmap, norm=norm, aspect='auto')

    # Add text annotations with sample sizes
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            if not np.isnan(value):
                # Use white text for dark backgrounds, black for light
                bin_idx = np.digitize(value, boundaries[1:-1])
                text_color = 'white' if bin_idx >= 4 else 'black'
                ax.text(j, i, f'{int(value)}', ha='center', va='center',
                       color=text_color, fontsize=10, fontweight='bold')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
    ax.set_yticklabels(heatmap_data.index)

    # Add gridlines
    ax.set_xticks(np.arange(len(heatmap_data.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(heatmap_data.index)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    # Styling
    ax.set_xlabel('Technology', fontsize=12, fontweight='bold')
    ax.set_ylabel('Material', fontsize=12, fontweight='bold')
    ax.set_title('Sample Size Distribution by Material and Technology',
                 fontsize=14, fontweight='bold', pad=20)

    # Tight layout
    plt.tight_layout()

    # Save
    outdir = ROOT / 'outputs' / 'fitting_sample_size_diagnostics'
    outfile = outdir / 'fig4_sample_size_heatmap.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {outfile}")

    # Print summary statistics
    print()
    print("Heatmap Statistics:")
    print(f"  Total material-technology combinations: {heatmap_data.size}")
    print(f"  Observed combinations: {heatmap_data.notna().sum().sum()}")
    print(f"  Missing combinations: {heatmap_data.isna().sum().sum()}")
    print(f"  Sample size range: {heatmap_data.min().min():.0f} to {heatmap_data.max().max():.0f}")
    print(f"  Median sample size: {heatmap_data.median().median():.0f}")

    # Print bin distribution
    print()
    print("Bin Distribution:")
    for bin_name in bin_order:
        count = pair_counts['n'].apply(lambda x: bin_sample_size(x) == bin_name).sum()
        print(f"  {bin_name}: {count} pairs")

    return outfile


def main():
    print("=" * 80)
    print("GENERATING SAMPLE SIZE HEATMAP")
    print("=" * 80)
    print()

    # Generate heatmap
    outfile = create_sample_size_heatmap()

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print()
    print(f"Generated: {outfile}")
    print()


if __name__ == '__main__':
    main()
