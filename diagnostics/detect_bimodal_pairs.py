#!/usr/bin/env python3
"""
Detect Bimodal Material Intensity Distributions
================================================

Scans all technology-material pairs to identify bimodal distributions that mix
distinct material component types (e.g., cell vs BOS materials in photovoltaics).

Uses gap-based detection: looks for large ratio jumps between consecutive sorted values.

Outputs:
- CSV report of all bimodal pairs with split thresholds and cluster sizes
- Detailed visualizations for each bimodal pair showing the two clusters

Usage:
    python diagnostics/detect_bimodal_pairs.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.technology_mapping import TECHNOLOGY_CONSOLIDATION

# Output directory
OUTPUT_DIR = ROOT / 'outputs' / 'bimodal_detection'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def detect_bimodal(values, min_gap_ratio=5.0, position_range=(0.2, 0.8)):
    """
    Detect if data has a bimodal distribution with large gap separating two clusters.

    Parameters
    ----------
    values : array-like
        Data values to test for bimodality
    min_gap_ratio : float
        Minimum ratio between consecutive values to be considered a gap (default: 5.0)
    position_range : tuple
        (min, max) percentile positions where gap must occur (default: (0.2, 0.8))
        Excludes gaps at extreme ends of distribution

    Returns
    -------
    is_bimodal : bool
        True if data appears to be bimodal
    split_threshold : float or None
        Value to split low/high clusters (midpoint of gap)
    gap_info : dict
        Additional information about the detected gap
    """
    if len(values) < 3:
        return False, None, {}

    sorted_vals = np.sort(values)
    n = len(sorted_vals)

    # Find all gaps (consecutive value ratios)
    gaps = []
    for i in range(n - 1):
        if sorted_vals[i] > 0:  # Avoid division by zero
            ratio = sorted_vals[i + 1] / sorted_vals[i]
            percentile_pos = (i + 1) / n
            gaps.append({
                'index': i,
                'low_val': sorted_vals[i],
                'high_val': sorted_vals[i + 1],
                'ratio': ratio,
                'percentile_pos': percentile_pos
            })

    if not gaps:
        return False, None, {}

    # Find largest gap that meets criteria
    valid_gaps = [
        g for g in gaps
        if g['ratio'] >= min_gap_ratio and
        position_range[0] <= g['percentile_pos'] <= position_range[1]
    ]

    if not valid_gaps:
        return False, None, {}

    # Select largest gap
    largest_gap = max(valid_gaps, key=lambda g: g['ratio'])

    # Split threshold: geometric mean of gap endpoints (appropriate for log-scale data)
    split_threshold = np.sqrt(largest_gap['low_val'] * largest_gap['high_val'])

    # Count cluster sizes
    n_low = np.sum(values <= split_threshold)
    n_high = np.sum(values > split_threshold)

    gap_info = {
        'gap_ratio': largest_gap['ratio'],
        'gap_low': largest_gap['low_val'],
        'gap_high': largest_gap['high_val'],
        'gap_percentile': largest_gap['percentile_pos'],
        'n_low': n_low,
        'n_high': n_high
    }

    return True, split_threshold, gap_info


def visualize_bimodal_pair(technology, material, values, split_threshold, gap_info, output_path):
    """
    Create visualization showing the two clusters in a bimodal distribution.

    Parameters
    ----------
    technology : str
        Technology name
    material : str
        Material name
    values : array
        All intensity values
    split_threshold : float
        Threshold separating low and high clusters
    gap_info : dict
        Information about the gap
    output_path : Path
        Where to save the visualization
    """
    n = len(values)
    low_cluster = values[values <= split_threshold]
    high_cluster = values[values > split_threshold]

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'{technology} - {material} (n={n}) — Bimodal Distribution Detected',
        fontsize=16, fontweight='bold'
    )

    # Panel 1: Histogram with clusters highlighted
    ax1 = axes[0]

    # Plot low cluster
    ax1.hist(low_cluster, bins=min(10, len(low_cluster)), alpha=0.6,
             color='#3498db', edgecolor='black', label=f'Low Cluster (n={len(low_cluster)})')

    # Plot high cluster
    ax1.hist(high_cluster, bins=min(10, len(high_cluster)), alpha=0.6,
             color='#e74c3c', edgecolor='black', label=f'High Cluster (n={len(high_cluster)})')

    # Mark split threshold
    ax1.axvline(split_threshold, color='black', linestyle='--', linewidth=2,
                label=f'Split Threshold = {split_threshold:.1f} g/MW')

    ax1.set_xlabel('Intensity (g/MW)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Histogram with Clusters', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Sorted values showing gap
    ax2 = axes[1]
    sorted_vals = np.sort(values)

    # Plot sorted values
    indices = np.arange(len(sorted_vals))
    colors = ['#3498db' if v <= split_threshold else '#e74c3c' for v in sorted_vals]
    ax2.scatter(indices, sorted_vals, c=colors, s=80, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.plot(indices, sorted_vals, 'k-', alpha=0.3, linewidth=1)

    # Highlight gap
    gap_idx = np.where(sorted_vals == gap_info['gap_low'])[0][0]
    ax2.plot([gap_idx, gap_idx + 1],
             [gap_info['gap_low'], gap_info['gap_high']],
             'r-', linewidth=3, alpha=0.7, label=f'Gap: {gap_info["gap_ratio"]:.1f}x ratio')

    # Mark threshold
    ax2.axhline(split_threshold, color='black', linestyle='--', linewidth=2, alpha=0.5)

    ax2.set_xlabel('Observation Index (sorted)', fontsize=12)
    ax2.set_ylabel('Intensity (g/MW)', fontsize=12)
    ax2.set_title('Sorted Values Showing Gap', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, frameon=True, fancybox=True)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Cluster statistics table
    ax3 = axes[2]
    ax3.axis('off')

    # Prepare statistics
    stats_data = [
        ['Total Observations', f'{n}'],
        ['', ''],
        ['Low Cluster', ''],
        ['  Count', f'{len(low_cluster)}'],
        ['  Min', f'{np.min(low_cluster):.2f} g/MW'],
        ['  Median', f'{np.median(low_cluster):.2f} g/MW'],
        ['  Max', f'{np.max(low_cluster):.2f} g/MW'],
        ['  Mean ± SD', f'{np.mean(low_cluster):.1f} ± {np.std(low_cluster):.1f} g/MW'],
        ['', ''],
        ['High Cluster', ''],
        ['  Count', f'{len(high_cluster)}'],
        ['  Min', f'{np.min(high_cluster):.2f} g/MW'],
        ['  Median', f'{np.median(high_cluster):.2f} g/MW'],
        ['  Max', f'{np.max(high_cluster):.2f} g/MW'],
        ['  Mean ± SD', f'{np.mean(high_cluster):.1f} ± {np.std(high_cluster):.1f} g/MW'],
        ['', ''],
        ['Gap Information', ''],
        ['  Split Threshold', f'{split_threshold:.2f} g/MW'],
        ['  Gap Ratio', f'{gap_info["gap_ratio"]:.2f}x'],
        ['  Low → High', f'{gap_info["gap_low"]:.1f} → {gap_info["gap_high"]:.1f} g/MW'],
        ['  Gap Percentile', f'{gap_info["gap_percentile"]:.1%}'],
    ]

    # Create table
    table = ax3.table(cellText=stats_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    # Style table
    for i in range(len(stats_data)):
        cell = table[(i, 0)]
        # Headers
        if stats_data[i][0] in ['Low Cluster', 'High Cluster', 'Gap Information']:
            cell.set_facecolor('#34495e')
            cell.set_text_props(weight='bold', color='white')
            table[(i, 1)].set_facecolor('#34495e')
        # Empty rows
        elif stats_data[i][0] == '':
            cell.set_facecolor('#ecf0f1')
            table[(i, 1)].set_facecolor('#ecf0f1')

    ax3.set_title('Cluster Statistics', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 80)
    print("DETECTING BIMODAL MATERIAL INTENSITY DISTRIBUTIONS")
    print("=" * 80)
    print()

    # Load intensity data
    print("Loading intensity data...")
    data_path = ROOT / 'data' / 'intensity_data.csv'
    df = pd.read_csv(data_path)
    df.columns = ['Technology', 'Material', 'g_per_MW']

    # Apply technology consolidation
    df['Technology'] = df['Technology'].replace(TECHNOLOGY_CONSOLIDATION)

    print(f"✓ Loaded {len(df)} intensity observations")
    print(f"✓ {len(df['Technology'].unique())} technologies")
    print(f"✓ {len(df['Material'].unique())} materials")
    print()

    # Test each tech-material pair
    print("=" * 80)
    print("SCANNING FOR BIMODAL DISTRIBUTIONS")
    print("=" * 80)
    print()

    results = []
    bimodal_pairs = []

    for (tech, mat), group in df.groupby(['Technology', 'Material']):
        values = group['g_per_MW'].values
        n = len(values)

        # Skip pairs with too few observations
        if n < 3:
            continue

        # Detect bimodality
        is_bimodal, split_threshold, gap_info = detect_bimodal(values)

        if is_bimodal:
            bimodal_pairs.append((tech, mat))

            result = {
                'technology': tech,
                'material': mat,
                'n_total': n,
                'is_bimodal': True,
                'split_threshold': split_threshold,
                'gap_ratio': gap_info['gap_ratio'],
                'gap_low': gap_info['gap_low'],
                'gap_high': gap_info['gap_high'],
                'n_low': gap_info['n_low'],
                'n_high': gap_info['n_high'],
                'low_median': np.median(values[values <= split_threshold]),
                'high_median': np.median(values[values > split_threshold]),
            }
            results.append(result)

            print(f"✓ BIMODAL: {tech} - {mat} (n={n})")
            print(f"  Gap: {gap_info['gap_low']:.1f} → {gap_info['gap_high']:.1f} ({gap_info['gap_ratio']:.1f}x ratio)")
            print(f"  Clusters: Low n={gap_info['n_low']} (median={result['low_median']:.1f}), "
                  f"High n={gap_info['n_high']} (median={result['high_median']:.1f})")

    print()
    print("=" * 80)
    print(f"FOUND {len(bimodal_pairs)} BIMODAL PAIRS")
    print("=" * 80)
    print()

    if not results:
        print("No bimodal distributions detected.")
        return

    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values('gap_ratio', ascending=False)

    # Display summary
    print("Top bimodal pairs by gap ratio:")
    print()
    display_cols = ['technology', 'material', 'n_total', 'gap_ratio', 'n_low', 'n_high']
    print(summary_df[display_cols].head(15).to_string(index=False))
    print()

    # Save summary CSV
    output_csv = OUTPUT_DIR / 'bimodal_pairs_summary.csv'
    summary_df.to_csv(output_csv, index=False)
    print(f"✓ Saved summary: {output_csv}")
    print()

    # Generate visualizations
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    print()

    for idx, row in summary_df.iterrows():
        tech = row['technology']
        mat = row['material']
        split_threshold = row['split_threshold']

        # Get data
        pair_data = df[(df['Technology'] == tech) & (df['Material'] == mat)]['g_per_MW'].values

        # Reconstruct gap_info
        gap_info = {
            'gap_ratio': row['gap_ratio'],
            'gap_low': row['gap_low'],
            'gap_high': row['gap_high'],
            'gap_percentile': 0.5,  # Approximate
            'n_low': row['n_low'],
            'n_high': row['n_high']
        }

        # Create safe filename
        safe_tech = tech.replace(' ', '_').replace('/', '-')
        safe_mat = mat.replace(' ', '_').replace('/', '-')
        output_file = OUTPUT_DIR / f"bimodal_{safe_tech}_{safe_mat}.png"

        print(f"Generating: {tech} - {mat}...")
        visualize_bimodal_pair(tech, mat, pair_data, split_threshold, gap_info, output_file)
        print(f"  ✓ Saved: {output_file.name}")

    print()
    print("=" * 80)
    print("BIMODAL DETECTION STATISTICS")
    print("=" * 80)
    print()

    print(f"Total tech-material pairs scanned: {len(df.groupby(['Technology', 'Material']))}")
    print(f"Bimodal pairs detected: {len(bimodal_pairs)} ({100*len(bimodal_pairs)/len(df.groupby(['Technology', 'Material'])):.1f}%)")
    print()

    # Breakdown by material
    print("Most common materials in bimodal pairs:")
    material_counts = summary_df['material'].value_counts()
    for mat, count in material_counts.head(10).items():
        print(f"  {mat:20s}: {count} pairs")
    print()

    # Breakdown by technology
    print("Most common technologies in bimodal pairs:")
    tech_counts = summary_df['technology'].value_counts()
    for tech, count in tech_counts.head(10).items():
        print(f"  {tech:30s}: {count} pairs")
    print()

    # Gap ratio statistics
    print("Gap ratio statistics:")
    print(f"  Min:    {summary_df['gap_ratio'].min():.2f}x")
    print(f"  Median: {summary_df['gap_ratio'].median():.2f}x")
    print(f"  Mean:   {summary_df['gap_ratio'].mean():.2f}x")
    print(f"  Max:    {summary_df['gap_ratio'].max():.2f}x")
    print()

    # Cluster size balance
    summary_df['cluster_balance'] = summary_df[['n_low', 'n_high']].min(axis=1) / summary_df[['n_low', 'n_high']].max(axis=1)
    print("Cluster size balance (min/max ratio):")
    print(f"  Very unbalanced (<0.2): {len(summary_df[summary_df['cluster_balance'] < 0.2])} pairs")
    print(f"  Unbalanced (0.2-0.5):   {len(summary_df[(summary_df['cluster_balance'] >= 0.2) & (summary_df['cluster_balance'] < 0.5)])} pairs")
    print(f"  Balanced (≥0.5):        {len(summary_df[summary_df['cluster_balance'] >= 0.5])} pairs")
    print()

    print("=" * 80)
    print(f"COMPLETE - {len(bimodal_pairs)} bimodal pairs identified")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)
    print()

    print("NEXT STEPS:")
    print("1. Review visualizations to confirm bimodal pairs are legitimate")
    print("2. Implement component-based fitting for bimodal pairs")
    print("3. Update Monte Carlo sampling to sum component distributions")


if __name__ == '__main__':
    main()
