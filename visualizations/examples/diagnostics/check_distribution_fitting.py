"""
Distribution Fitting Diagnostic and Visualization Tool
=======================================================

This script checks the distribution fitting algorithm and creates detailed
visualizations comparing:
1. Raw intensity data points
2. Fitted distributions (PDF and CDF)
3. Monte Carlo samples drawn from the fitted distribution
4. Goodness-of-fit statistics

Focus: Aluminum (well-behaved) and Cement (problematic)
"""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data_ingestion import load_all_data
from distribution_fitting import DistributionFitter
from scipy import stats

# Configuration
TEST_CASES = [
    ('ASIGE', 'Aluminum', 'Well-behaved material'),
    ('Gas', 'Cement', 'Problematic material - fat tails'),
]

N_MONTE_CARLO_SAMPLES = 10000  # Match the simulation

print("=" * 80)
print("DISTRIBUTION FITTING DIAGNOSTIC")
print("=" * 80)

# Load data
print("\n1. Loading data...")
data = load_all_data(
    '../data/intensity_data.csv',
    '../data/StdScen24_annual_national.csv'
)
intensity_df = data['intensity']

# Fit distributions
print("2. Fitting distributions...")
fitter = DistributionFitter()
fitted_dists = fitter.fit_all(intensity_df)

# Create figure with subplots for each test case
fig = plt.figure(figsize=(16, 6 * len(TEST_CASES)))
gs = gridspec.GridSpec(len(TEST_CASES), 3, figure=fig, hspace=0.35, wspace=0.3)

for idx, (technology, material, description) in enumerate(TEST_CASES):
    print(f"\n{'=' * 80}")
    print(f"TEST CASE {idx + 1}: {technology} + {material}")
    print(f"Description: {description}")
    print("=" * 80)

    # Get raw data
    raw_data = intensity_df[
        (intensity_df['technology'] == technology) &
        (intensity_df['material'] == material)
    ]['intensity_t_per_mw'].values

    print(f"\nRaw data: {raw_data}")
    print(f"  n = {len(raw_data)}")
    print(f"  mean = {np.mean(raw_data):.3f} t/MW")
    print(f"  std = {np.std(raw_data):.3f} t/MW")
    print(f"  median = {np.median(raw_data):.3f} t/MW")
    print(f"  min = {np.min(raw_data):.3f} t/MW")
    print(f"  max = {np.max(raw_data):.3f} t/MW")
    print(f"  CV = {(np.std(raw_data) / np.mean(raw_data)) * 100:.1f}%")

    # Get fitted distribution
    if (technology, material) not in fitted_dists:
        print(f"  ✗ No distribution fitted!")
        continue

    dist_obj = fitted_dists[(technology, material)]

    print(f"\nDistribution fitting results:")
    print(f"  Number of distributions fitted: {len(dist_obj.fitted_distributions)}")
    print(f"  Use parametric: {dist_obj.use_parametric}")
    print(f"  Recommendation: {dist_obj.recommendation}")

    if dist_obj.fitted_distributions:
        print(f"\n  Fitted distributions (sorted by AIC):")
        for i, fit in enumerate(dist_obj.fitted_distributions[:4], 1):
            ks_status = "PASS" if fit.passes_ks_test() else "FAIL"
            print(f"    {i}. {fit.distribution_name:20s} "
                  f"AIC={fit.aic:8.2f}  KS={fit.ks_statistic:.4f} (p={fit.ks_pvalue:.4f}) [{ks_status}]")

    # Generate Monte Carlo samples
    print(f"\n3. Generating {N_MONTE_CARLO_SAMPLES:,} Monte Carlo samples...")
    mc_samples = dist_obj.sample(N_MONTE_CARLO_SAMPLES, random_state=42)

    print(f"\nMonte Carlo sample statistics:")
    print(f"  mean = {np.mean(mc_samples):.3f} t/MW")
    print(f"  std = {np.std(mc_samples):.3f} t/MW")
    print(f"  median = {np.median(mc_samples):.3f} t/MW")
    print(f"  min = {np.min(mc_samples):.3f} t/MW")
    print(f"  max = {np.max(mc_samples):.3f} t/MW")
    print(f"  CV = {(np.std(mc_samples) / np.mean(mc_samples)) * 100:.1f}%")

    # Check for extreme outliers
    print(f"\nPercentile analysis:")
    percentiles = [2, 5, 25, 50, 75, 95, 97, 99, 99.9]
    for p in percentiles:
        val = np.percentile(mc_samples, p)
        ratio = val / np.median(mc_samples) if np.median(mc_samples) > 0 else float('inf')
        status = "✓" if ratio < 10 else ("⚠️" if ratio < 100 else "❌")
        print(f"  p{p:5.1f}: {val:15.3f} t/MW  (ratio to median: {ratio:8.2f}x) {status}")

    # === SUBPLOT 1: Histogram with fitted PDF ===
    ax1 = fig.add_subplot(gs[idx, 0])

    # Histogram of raw data
    ax1.hist(raw_data, bins=max(3, len(raw_data)), alpha=0.7, color='steelblue',
             edgecolor='black', density=True, label='Raw data')

    # If using parametric, plot the fitted PDF
    if dist_obj.use_parametric and dist_obj.best_fit:
        best_fit = dist_obj.best_fit
        x_range = np.linspace(max(0, np.min(raw_data) * 0.5),
                              np.max(raw_data) * 1.5, 200)

        # Get scipy distribution
        if best_fit.distribution_name == 'lognormal':
            fitted_dist = stats.lognorm(
                s=best_fit.parameters['s'],
                scale=best_fit.parameters['scale']
            )
        elif best_fit.distribution_name == 'gamma':
            fitted_dist = stats.gamma(
                a=best_fit.parameters['a'],
                scale=best_fit.parameters['scale']
            )
        elif best_fit.distribution_name == 'uniform':
            fitted_dist = stats.uniform(
                loc=best_fit.parameters['loc'],
                scale=best_fit.parameters['scale']
            )
        elif best_fit.distribution_name == 'truncated_normal':
            fitted_dist = stats.truncnorm(
                a=best_fit.parameters['a'],
                b=best_fit.parameters['b'],
                loc=best_fit.parameters['loc'],
                scale=best_fit.parameters['scale']
            )

        pdf = fitted_dist.pdf(x_range)
        ax1.plot(x_range, pdf, 'r-', linewidth=2,
                label=f'Fitted {best_fit.distribution_name}')

    ax1.axvline(np.mean(raw_data), color='blue', linestyle='--', linewidth=1.5,
               label=f'Raw mean ({np.mean(raw_data):.2f})')
    ax1.axvline(np.median(raw_data), color='green', linestyle='--', linewidth=1.5,
               label=f'Raw median ({np.median(raw_data):.2f})')

    ax1.set_xlabel('Material Intensity (t/MW)', fontsize=11)
    ax1.set_ylabel('Probability Density', fontsize=11)
    ax1.set_title(f'{technology} + {material}\nHistogram & Fitted PDF',
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # === SUBPLOT 2: CDF Comparison ===
    ax2 = fig.add_subplot(gs[idx, 1])

    # Empirical CDF of raw data
    sorted_raw = np.sort(raw_data)
    ecdf_raw = np.arange(1, len(sorted_raw) + 1) / len(sorted_raw)
    ax2.step(sorted_raw, ecdf_raw, where='post', linewidth=2,
            color='steelblue', label='Raw data ECDF')

    # Fitted CDF
    if dist_obj.use_parametric and dist_obj.best_fit:
        x_range = np.linspace(max(0, np.min(raw_data) * 0.5),
                              np.max(raw_data) * 1.5, 200)
        cdf = fitted_dist.cdf(x_range)
        ax2.plot(x_range, cdf, 'r-', linewidth=2,
                label=f'Fitted {best_fit.distribution_name} CDF')

        # Show KS statistic visually
        ks_stat = best_fit.ks_statistic
        ax2.text(0.05, 0.95, f'KS statistic: {ks_stat:.4f}\nKS p-value: {best_fit.ks_pvalue:.4f}',
                transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_xlabel('Material Intensity (t/MW)', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_title(f'Cumulative Distribution Function\n(Goodness of Fit Check)',
                 fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    # === SUBPLOT 3: Monte Carlo Samples vs Raw Data ===
    ax3 = fig.add_subplot(gs[idx, 2])

    # Histogram of Monte Carlo samples
    bins = np.linspace(0, np.percentile(mc_samples, 99), 50)
    ax3.hist(mc_samples, bins=bins, alpha=0.6, color='orange',
            edgecolor='black', density=True, label=f'MC samples (n={N_MONTE_CARLO_SAMPLES:,})')

    # Overlay raw data points as scatter
    for i, val in enumerate(raw_data):
        ax3.axvline(val, color='red', linestyle='-', linewidth=3, alpha=0.7,
                   ymax=0.1 if i == 0 else 0.1)
    ax3.scatter(raw_data, [0] * len(raw_data), color='red', s=200, zorder=5,
               marker='D', edgecolor='black', linewidth=1.5, label='Raw data points')

    # Add percentile markers
    for p in [50, 95, 99]:
        val = np.percentile(mc_samples, p)
        ax3.axvline(val, color='purple', linestyle=':', linewidth=1.5, alpha=0.7)
        ax3.text(val, ax3.get_ylim()[1] * 0.9, f'p{p}', fontsize=8,
                ha='center', color='purple', fontweight='bold')

    ax3.set_xlabel('Material Intensity (t/MW)', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title(f'Monte Carlo Samples vs Raw Data\n(99th percentile shown)',
                 fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # Add statistics box
    stats_text = (f"MC Mean: {np.mean(mc_samples):.2f}\n"
                 f"MC Median: {np.median(mc_samples):.2f}\n"
                 f"MC Std: {np.std(mc_samples):.2f}\n"
                 f"MC CV: {(np.std(mc_samples)/np.mean(mc_samples))*100:.1f}%")
    ax3.text(0.98, 0.97, stats_text, transform=ax3.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Overall title
fig.suptitle('Distribution Fitting Diagnostic: Comparing Raw Data, Fitted Distributions, and Monte Carlo Samples',
            fontsize=14, fontweight='bold', y=0.995)

# Save figure
output_path = '../outputs/distribution_fitting_diagnostic.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n{'=' * 80}")
print(f"✓ Visualization saved to: {output_path}")
print("=" * 80)

# === Additional Check: Extreme Tail Behavior ===
print(f"\n{'=' * 80}")
print("EXTREME TAIL BEHAVIOR CHECK")
print("=" * 80)

for technology, material, description in TEST_CASES:
    if (technology, material) not in fitted_dists:
        continue

    dist_obj = fitted_dists[(technology, material)]

    print(f"\n{technology} + {material}:")

    # Generate large sample to check for extreme outliers
    large_sample = dist_obj.sample(100000, random_state=123)

    # Check ratio of max to median
    max_val = np.max(large_sample)
    median_val = np.median(large_sample)
    max_to_median_ratio = max_val / median_val

    print(f"  100k sample max: {max_val:.2e} t/MW")
    print(f"  100k sample median: {median_val:.2e} t/MW")
    print(f"  Ratio (max/median): {max_to_median_ratio:.2e}x")

    if max_to_median_ratio > 1000:
        print(f"  ❌ WARNING: Extreme outliers detected! Max is {max_to_median_ratio:.0f}× the median")
        print(f"     This will corrupt the arithmetic mean in Monte Carlo aggregation")
    elif max_to_median_ratio > 100:
        print(f"  ⚠️  CAUTION: Large outliers present. Max is {max_to_median_ratio:.0f}× the median")
    else:
        print(f"  ✓ HEALTHY: Max is only {max_to_median_ratio:.1f}× the median - well-behaved distribution")

print(f"\n{'=' * 80}")
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
