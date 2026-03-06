#!/usr/bin/env python3
"""
Generate sample size distribution visualizations.

Creates publication-quality stacked bar charts showing how material-technology
pairs are distributed across sample size bins.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set up paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.technology_mapping import TECHNOLOGY_CONSOLIDATION


def load_intensity_data():
    """Load and preprocess intensity data with technology consolidation."""
    data_path = ROOT / 'data' / 'intensity_data.csv'
    df = pd.read_csv(data_path)

    # Standardize column names
    df.columns = ['Technology', 'Material', 'g_per_MW']

    # Apply technology consolidation (CDTE→CdTe, ASIGE→a-Si)
    df['Technology'] = df['Technology'].replace(TECHNOLOGY_CONSOLIDATION)

    return df


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


def plot_technology_distribution():
    """Create stacked bar chart by technology."""
    print("Creating technology distribution plot...")

    df = load_intensity_data()

    # Count sample sizes per pair
    pair_counts = df.groupby(['Technology', 'Material']).size().reset_index(name='n')
    pair_counts['bin'] = pair_counts['n'].apply(bin_sample_size)

    # Bin order and colors
    bin_order = ['n=1', 'n=2', 'n=3', '4≤n≤5', '6≤n≤10', '11≤n≤20', 'n>20']
    colors = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4', '#313695']

    # Create pivot table for stacking
    pivot_data = []

    # Get technologies sorted by total count (ascending)
    tech_counts = pair_counts.groupby('Technology').size().sort_values(ascending=True)
    technologies = tech_counts.index

    for tech in technologies:
        tech_data = pair_counts[pair_counts['Technology'] == tech]
        bin_counts = tech_data['bin'].value_counts()

        row = {'Technology': tech, 'Total': len(tech_data)}
        for bin_name in bin_order:
            row[bin_name] = bin_counts.get(bin_name, 0)

        pivot_data.append(row)

    pivot_df = pd.DataFrame(pivot_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot stacked bars
    bottom = np.zeros(len(pivot_df))

    for idx, bin_name in enumerate(bin_order):
        values = pivot_df[bin_name].values
        ax.barh(pivot_df['Technology'], values, left=bottom,
                color=colors[idx], label=bin_name, edgecolor='white', linewidth=0.5)
        bottom += values

    # Styling
    ax.set_xlabel('Number of Material-Technology Pairs', fontsize=12, fontweight='bold')
    ax.set_ylabel('Technology', fontsize=12, fontweight='bold')
    ax.set_title('Sample Size Distribution by Technology', fontsize=14, fontweight='bold', pad=20)

    # Set x-axis ticks to even numbers from 0 to 20
    ax.set_xticks(np.arange(0, 21, 2))
    ax.set_xlim(0, 20)

    # Legend
    ax.legend(title='Sample Size Bin', loc='lower right', frameon=True, fontsize=10)

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Tight layout
    plt.tight_layout()

    # Save
    outdir = ROOT / 'outputs' / 'fitting_sample_size_diagnostics'
    outfile = outdir / 'fig1_sample_size_distribution_by_technology.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {outfile}")
    return outfile


def plot_material_distribution():
    """Create stacked bar chart by material (top 20 by total observations)."""
    print("Creating material distribution plot...")

    df = load_intensity_data()

    # Count sample sizes per pair
    pair_counts = df.groupby(['Technology', 'Material']).size().reset_index(name='n')
    pair_counts['bin'] = pair_counts['n'].apply(bin_sample_size)

    # Bin order and colors
    bin_order = ['n=1', 'n=2', 'n=3', '4≤n≤5', '6≤n≤10', '11≤n≤20', 'n>20']
    colors = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4', '#313695']

    # Create pivot table for stacking
    pivot_data = []

    # Get materials sorted by total count
    mat_counts = pair_counts.groupby('Material').size().sort_values(ascending=False)

    # Keep top 20 materials, then reverse to show smallest-to-largest
    top_20_materials = mat_counts.head(20).sort_values(ascending=True).index

    for material in top_20_materials:
        mat_data = pair_counts[pair_counts['Material'] == material]
        bin_counts = mat_data['bin'].value_counts()

        row = {'Material': material, 'Total': len(mat_data)}
        for bin_name in bin_order:
            row[bin_name] = bin_counts.get(bin_name, 0)

        pivot_data.append(row)

    pivot_df = pd.DataFrame(pivot_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot stacked bars
    bottom = np.zeros(len(pivot_df))

    for idx, bin_name in enumerate(bin_order):
        values = pivot_df[bin_name].values
        ax.barh(pivot_df['Material'], values, left=bottom,
                color=colors[idx], label=bin_name, edgecolor='white', linewidth=0.5)
        bottom += values

    # Styling
    ax.set_xlabel('Number of Material-Technology Pairs', fontsize=12, fontweight='bold')
    ax.set_ylabel('Material', fontsize=12, fontweight='bold')
    ax.set_title('Sample Size Distribution by Material (Top 20)', fontsize=14, fontweight='bold', pad=20)

    # Set x-axis ticks to even numbers from 0 to 20
    ax.set_xticks(np.arange(0, 21, 2))
    ax.set_xlim(0, 20)

    # Legend
    ax.legend(title='Sample Size Bin', loc='lower right', frameon=True, fontsize=10)

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Tight layout
    plt.tight_layout()

    # Save
    outdir = ROOT / 'outputs' / 'fitting_sample_size_diagnostics'
    outfile = outdir / 'fig2_sample_size_distribution_by_material.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {outfile}")
    return outfile


def plot_overall_distribution():
    """Create overall sample size distribution histogram."""
    print("Creating overall distribution plot...")

    df = load_intensity_data()

    # Count sample sizes per pair
    pair_counts = df.groupby(['Technology', 'Material']).size().reset_index(name='n')
    pair_counts['bin'] = pair_counts['n'].apply(bin_sample_size)

    # Bin order and colors
    bin_order = ['n=1', 'n=2', 'n=3', '4≤n≤5', '6≤n≤10', '11≤n≤20', 'n>20']
    colors = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4', '#313695']

    # Count pairs in each bin
    bin_counts = pair_counts['bin'].value_counts().reindex(bin_order, fill_value=0)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar plot
    bars = ax.bar(range(len(bin_order)), bin_counts.values, color=colors,
                   edgecolor='black', linewidth=1.5)

    # Add count labels on bars
    for idx, (bar, count) in enumerate(zip(bars, bin_counts.values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Styling
    ax.set_xticks(range(len(bin_order)))
    ax.set_xticklabels(bin_order, fontsize=11)
    ax.set_xlabel('Sample Size Bin', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Material-Technology Pairs', fontsize=12, fontweight='bold')
    ax.set_title('Overall Sample Size Distribution Across All Material-Technology Pairs',
                 fontsize=14, fontweight='bold', pad=20)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add summary stats as text box
    total_pairs = len(pair_counts)
    low_n = (pair_counts['n'] < 5).sum()
    pct_low = 100 * low_n / total_pairs

    textstr = f'Total pairs: {total_pairs}\n'
    textstr += f'n<5: {low_n} ({pct_low:.1f}%)\n'
    textstr += f'n≥10: {(pair_counts["n"] >= 10).sum()} ({100*(pair_counts["n"] >= 10).sum()/total_pairs:.1f}%)'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # Tight layout
    plt.tight_layout()

    # Save
    outdir = ROOT / 'outputs' / 'fitting_sample_size_diagnostics'
    outfile = outdir / 'fig3_overall_sample_size_distribution.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {outfile}")
    return outfile


def main():
    print("=" * 80)
    print("GENERATING SAMPLE SIZE DISTRIBUTION VISUALIZATIONS")
    print("=" * 80)
    print()

    # Generate plots
    tech_file = plot_technology_distribution()
    print()

    mat_file = plot_material_distribution()
    print()

    overall_file = plot_overall_distribution()
    print()

    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print()
    print("Generated:")
    print(f"  - {tech_file}")
    print(f"  - {mat_file}")
    print(f"  - {overall_file}")
    print()


if __name__ == '__main__':
    main()
