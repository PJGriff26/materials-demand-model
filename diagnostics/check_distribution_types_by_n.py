#!/usr/bin/env python3
"""
Check distribution types by sample size bin
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_ingestion import load_all_data
from src.distribution_fitting import DistributionFitter


def main():
    print("=" * 80)
    print("DISTRIBUTION TYPES BY SAMPLE SIZE")
    print("=" * 80)
    print()

    # Load data
    data_path = ROOT / 'data' / 'intensity_data.csv'
    data = load_all_data(
        intensity_path=data_path,
        national_capacity_path=ROOT / 'data' / 'StdScen24_annual_national.csv'
    )

    # Fit distributions
    fitter = DistributionFitter()
    fitted_dists = fitter.fit_all(data['intensity'])

    # Collect distribution info
    results = []
    for (tech, mat), dist_info in fitted_dists.items():
        if dist_info.best_fit is not None:
            results.append({
                'Technology': tech,
                'Material': mat,
                'n': dist_info.n_samples,
                'Distribution': dist_info.best_fit.distribution_name,
                'KS_pvalue': dist_info.best_fit.ks_pvalue,
                'AIC': dist_info.best_fit.aic
            })

    df = pd.DataFrame(results)

    # Define sample size bins
    bins = [
        ('n=1', df['n'] == 1),
        ('n=2', df['n'] == 2),
        ('n=3', df['n'] == 3),
        ('n=4', df['n'] == 4),
        ('n=5-9', (df['n'] >= 5) & (df['n'] < 10)),
        ('n=10-19', (df['n'] >= 10) & (df['n'] < 20)),
        ('n≥20', df['n'] >= 20),
        ('ALL', df['n'] >= 0),
    ]

    print("\nDistribution type breakdown by sample size bin:")
    print("=" * 80)

    for bin_name, condition in bins:
        subset = df[condition]
        if len(subset) == 0:
            continue

        print(f"\n{bin_name} ({len(subset)} pairs):")
        dist_counts = subset['Distribution'].value_counts()

        for dist_type in ['lognormal', 'gamma', 'truncated_normal', 'uniform']:
            count = dist_counts.get(dist_type, 0)
            pct = 100 * count / len(subset)
            print(f"  {dist_type:20s}: {count:3d} ({pct:5.1f}%)")

    # Focus on well-characterized (n ≥ 5)
    print("\n" + "=" * 80)
    print("WELL-CHARACTERIZED PAIRS (n ≥ 5)")
    print("=" * 80)

    well_char = df[df['n'] >= 5]
    print(f"\nTotal: {len(well_char)} pairs")

    dist_counts = well_char['Distribution'].value_counts()
    for dist_type, count in dist_counts.items():
        pct = 100 * count / len(well_char)
        print(f"  {dist_type:20s}: {count:3d} ({pct:5.1f}%)")

    # Even stricter: n ≥ 10
    print("\n" + "=" * 80)
    print("VERY WELL-CHARACTERIZED PAIRS (n ≥ 10)")
    print("=" * 80)

    very_well = df[df['n'] >= 10]
    print(f"\nTotal: {len(very_well)} pairs")

    dist_counts = very_well['Distribution'].value_counts()
    for dist_type, count in dist_counts.items():
        pct = 100 * count / len(very_well)
        print(f"  {dist_type:20s}: {count:3d} ({pct:5.1f}%)")

    # Show some examples of lognormal fits
    print("\n" + "=" * 80)
    print("LOGNORMAL FITS (all sample sizes)")
    print("=" * 80)

    lognormal_fits = df[df['Distribution'] == 'lognormal'].sort_values('n', ascending=False)
    print(f"\nTotal lognormal fits: {len(lognormal_fits)}")
    print("\nAll lognormal fits:")
    print(lognormal_fits[['Technology', 'Material', 'n', 'KS_pvalue', 'AIC']].to_string(index=False))

    # Compare with gamma for same high-n pairs
    print("\n" + "=" * 80)
    print("HIGH-N GAMMA FITS (n ≥ 10)")
    print("=" * 80)

    gamma_high = df[(df['Distribution'] == 'gamma') & (df['n'] >= 10)].sort_values('n', ascending=False)
    print(f"\nTotal gamma fits with n ≥ 10: {len(gamma_high)}")
    print("\nFirst 10 gamma fits (n ≥ 10):")
    print(gamma_high.head(10)[['Technology', 'Material', 'n', 'KS_pvalue']].to_string(index=False))


if __name__ == '__main__':
    main()
