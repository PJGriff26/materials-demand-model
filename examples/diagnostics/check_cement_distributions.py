"""
Check Cement distributions across all technologies
===================================================

Identify which Cement-technology combinations are using parametric distributions
and whether they're generating extreme outliers.
"""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_dir)

import pandas as pd
import numpy as np
from data_ingestion import load_all_data
from distribution_fitting import DistributionFitter

# Load data
print("Loading data...")
data = load_all_data(
    '../data/intensity_data.csv',
    '../data/StdScen24_annual_national.csv'
)
intensity_df = data['intensity']

# Fit distributions
print("Fitting distributions...")
fitter = DistributionFitter()
fitted_dists = fitter.fit_all(intensity_df)

print("\n" + "=" * 100)
print("CEMENT DISTRIBUTION ANALYSIS ACROSS ALL TECHNOLOGIES")
print("=" * 100)

# Find all Cement entries
cement_entries = [(tech, mat, dist_obj) for (tech, mat), dist_obj in fitted_dists.items() if 'Cement' in mat]

print(f"\nFound {len(cement_entries)} technology-Cement combinations\n")

problematic = []

for tech, mat, dist_obj in sorted(cement_entries, key=lambda x: x[2].n_samples, reverse=True):
    # Sample 100k to check for extreme outliers
    sample = dist_obj.sample(100000, random_state=42)

    max_val = np.max(sample)
    median_val = np.median(sample)
    mean_val = np.mean(sample)
    p99 = np.percentile(sample, 99)
    p99_9 = np.percentile(sample, 99.9)

    ratio_max_median = max_val / median_val if median_val > 0 else float('inf')
    ratio_p99_median = p99 / median_val if median_val > 0 else float('inf')
    cv = (np.std(sample) / mean_val) * 100 if mean_val > 0 else float('inf')

    # Classify
    if ratio_max_median > 1000:
        status = "❌ EXTREME"
        problematic.append((tech, mat, dist_obj, ratio_max_median))
    elif ratio_max_median > 100:
        status = "⚠️  LARGE"
    elif ratio_max_median > 10:
        status = "⚠️  MODERATE"
    else:
        status = "✓ HEALTHY"

    dist_type = "PARAMETRIC" if dist_obj.use_parametric else "EMPIRICAL"
    best_fit_name = dist_obj.best_fit.distribution_name if dist_obj.best_fit else "none"

    print(f"{tech:20s}  n={dist_obj.n_samples:3d}  {dist_type:11s}  {best_fit_name:15s}")
    print(f"  Raw data: {dist_obj.raw_data}")
    print(f"  Median: {median_val:10.2f}  Mean: {mean_val:10.2f}  CV: {cv:6.1f}%")
    print(f"  p99: {p99:12.2e}  (ratio: {ratio_p99_median:8.2f}×)")
    print(f"  Max: {max_val:12.2e}  (ratio: {ratio_max_median:8.2e}×)  {status}")
    print()

if problematic:
    print("\n" + "=" * 100)
    print(f"FOUND {len(problematic)} PROBLEMATIC CEMENT DISTRIBUTIONS")
    print("=" * 100)

    for tech, mat, dist_obj, ratio in sorted(problematic, key=lambda x: x[3], reverse=True):
        print(f"\n{tech} + {mat}:")
        print(f"  n = {dist_obj.n_samples}")
        print(f"  Raw data: {dist_obj.raw_data}")
        print(f"  Use parametric: {dist_obj.use_parametric}")
        if dist_obj.best_fit:
            print(f"  Best fit: {dist_obj.best_fit.distribution_name}")
            print(f"  Parameters: {dist_obj.best_fit.parameters}")
        print(f"  Max/Median ratio: {ratio:.2e}×  ← WILL CORRUPT MEAN!")

else:
    print("\n" + "=" * 100)
    print("✓ NO EXTREMELY PROBLEMATIC DISTRIBUTIONS FOUND")
    print("  All Cement distributions have reasonable tail behavior")
    print("=" * 100)
