#!/usr/bin/env python3
"""
Regenerate Table 1 with non-overlapping sample size bins.

Changes from cumulative (overlapping) to exclusive (non-overlapping) bins:
- OLD: n≥20, n≥10, n≥5, n<5 (overlapping)
- NEW: n≥20, 10≤n<20, 5≤n<10, n<5 (non-overlapping)
"""

import sys
from pathlib import Path
import pandas as pd

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


def assign_tier(tech_data):
    """Assign tier based on best material's sample size."""
    best_n = tech_data['n'].max()

    if best_n >= 30:
        return 1
    elif best_n >= 10:
        return 2
    elif best_n >= 5:
        return 3
    else:
        return 4


def generate_table1_non_overlapping():
    """Generate Table 1 with non-overlapping sample size bins."""

    print("Loading intensity data...")
    df = load_intensity_data()

    # Count sample sizes per material-tech pair
    pair_counts = df.groupby(['Technology', 'Material']).size().reset_index(name='n')

    # Compute statistics per technology
    tech_stats = []

    for tech, group in pair_counts.groupby('Technology'):
        # Total materials
        n_materials = len(group)

        # Best material (highest n)
        best_idx = group['n'].idxmax()
        best_material = group.loc[best_idx, 'Material']
        best_n = group.loc[best_idx, 'n']

        # Compute best CV (need to fit distribution)
        best_values = df[(df['Technology'] == tech) & (df['Material'] == best_material)]['g_per_MW'].values
        from scipy.stats import lognorm
        import numpy as np

        if len(best_values) >= 2 and (best_values > 0).all():
            try:
                shape, loc, scale = lognorm.fit(best_values, floc=0)
                sigma = shape
                best_cv = np.sqrt(np.exp(sigma**2) - 1)
            except:
                best_cv = 0.0
        else:
            best_cv = 0.0

        # Count materials in NON-OVERLAPPING bins
        n_ge_20 = (group['n'] >= 20).sum()
        n_10_to_20 = ((group['n'] >= 10) & (group['n'] < 20)).sum()
        n_5_to_10 = ((group['n'] >= 5) & (group['n'] < 10)).sum()
        n_lt_5 = (group['n'] < 5).sum()

        # Median CV for n≥10 materials
        eligible_materials = group[group['n'] >= 10]['Material'].tolist()

        if len(eligible_materials) > 0:
            cvs = []
            for mat in eligible_materials:
                mat_values = df[(df['Technology'] == tech) & (df['Material'] == mat)]['g_per_MW'].values
                if len(mat_values) >= 2 and (mat_values > 0).all():
                    try:
                        shape, loc, scale = lognorm.fit(mat_values, floc=0)
                        sigma = shape
                        cv = np.sqrt(np.exp(sigma**2) - 1)
                        cvs.append(cv)
                    except:
                        pass  # Skip materials that fail to fit

            median_cv = np.median(cvs) if cvs else None
        else:
            median_cv = None

        # Assign tier
        tier = assign_tier(group)

        tech_stats.append({
            'Technology': tech,
            'Tier': tier,
            'Materials': n_materials,
            'Best material': best_material,
            'Best n': best_n,
            'Best CV': round(best_cv, 3),
            'n≥20': n_ge_20,
            '10≤n<20': n_10_to_20,
            '5≤n<10': n_5_to_10,
            'n<5': n_lt_5,
            'Median CV (n≥10)': round(median_cv, 3) if median_cv is not None else '—'
        })

    # Create DataFrame and sort by Tier, then Best n descending
    table1 = pd.DataFrame(tech_stats)
    table1 = table1.sort_values(['Tier', 'Best n'], ascending=[True, False])

    return table1


def main():
    print("=" * 80)
    print("REGENERATING TABLE 1 WITH NON-OVERLAPPING BINS")
    print("=" * 80)
    print()

    # Generate table
    table1 = generate_table1_non_overlapping()

    # Display
    print("Table 1: Technology Reference Landscape (Non-Overlapping Bins)")
    print()
    print(table1.to_string(index=False))
    print()

    # Save as markdown
    outdir = ROOT / 'outputs' / 'fitting_sample_size_diagnostics'
    outfile = outdir / 'table1_technology_landscape.md'

    with open(outfile, 'w') as f:
        f.write("# Table 1: Technology Reference Landscape\n\n")
        f.write(table1.to_markdown(index=False))
        f.write("\n")

    print(f"✓ Saved: {outfile}")
    print()
    print("=" * 80)
    print("Key Change:")
    print("  OLD (cumulative):  n≥20, n≥10, n≥5, n<5")
    print("  NEW (exclusive):   n≥20, 10≤n<20, 5≤n<10, n<5")
    print("=" * 80)


if __name__ == '__main__':
    main()
