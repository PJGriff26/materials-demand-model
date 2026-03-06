#!/usr/bin/env python3
"""
Batch Generate Fit Visualizations for All Material-Technology Pairs
====================================================================

Creates comprehensive 4-panel distribution fit plots for all 167 observed
material-technology pairs in the intensity dataset.

Panels:
1. Top-left: Histogram + Fitted Lognormal PDF
2. Top-right: CDF Comparison (empirical vs fitted)
3. Bottom-left: Q-Q Plot (log-scale)
4. Bottom-right: Monte Carlo Samples Analysis

Output Organization:
- By sample size bin: outputs/fitting_sample_size_diagnostics/fit_visualizations/n_[bin]/
- Bins: n1, n2, n3, n4-5, n6-10, n11-20, n20plus
- Index file: fit_visualizations_index.md

Usage:
    python diagnostics/generate_all_fit_visualizations.py
    python diagnostics/generate_all_fit_visualizations.py --bins n1 n2  # Generate only specific bins
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import lognorm, kstest
from scipy import stats as sp_stats

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


def assign_bin(n):
    """Assign sample size to directory bin."""
    if n == 1:
        return 'n1'
    elif n == 2:
        return 'n2'
    elif n == 3:
        return 'n3'
    elif 4 <= n <= 5:
        return 'n4-5'
    elif 6 <= n <= 10:
        return 'n6-10'
    elif 11 <= n <= 20:
        return 'n11-20'
    else:
        return 'n20plus'


def fit_lognormal(values):
    """Fit lognormal distribution and compute diagnostics."""
    n = len(values)

    if n < 2 or (values <= 0).any():
        return None

    try:
        # Fit lognormal (floc=0 forces location to 0)
        shape, loc, scale = lognorm.fit(values, floc=0)

        # Compute CV from lognormal parameters
        sigma = shape
        cv = np.sqrt(np.exp(sigma**2) - 1)

        # Goodness-of-fit: Kolmogorov-Smirnov test
        ks_stat, ks_pval = kstest(values, lambda x: lognorm.cdf(x, shape, loc, scale))

        # RMSE between empirical and fitted CDF
        sorted_vals = np.sort(values)
        empirical_cdf = np.arange(1, n + 1) / n
        fitted_cdf = lognorm.cdf(sorted_vals, shape, loc, scale)
        rmse = np.sqrt(np.mean((empirical_cdf - fitted_cdf)**2))

        return {
            'shape': shape,
            'loc': loc,
            'scale': scale,
            'mean': lognorm.mean(shape, loc, scale),
            'median': lognorm.median(shape, loc, scale),
            'std': lognorm.std(shape, loc, scale),
            'cv': cv,
            'ks_stat': ks_stat,
            'ks_pval': ks_pval,
            'rmse': rmse
        }
    except:
        return None


def create_4panel_plot(technology, material, values, fit_params, n_samples=10000):
    """
    Create comprehensive 4-panel plot for a material-technology pair.

    Returns:
        fig: matplotlib figure object
    """
    n = len(values)

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
                          top=0.92, bottom=0.08, left=0.10, right=0.95)

    if fit_params is None:
        # Can't fit - show raw data only
        ax = fig.add_subplot(gs[:, :])
        ax.hist(values, bins=min(n, 10), alpha=0.7, color='steelblue', edgecolor='black')
        ax.scatter(values, [0]*n, color='red', s=100, zorder=5, marker='D')
        ax.set_xlabel('Intensity (g/MW)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{technology} - {material} (n={n}) - FITTING FAILED',
                    fontsize=13, fontweight='bold', color='red')
        ax.grid(alpha=0.3)
        return fig

    # Unpack fit parameters
    shape, loc, scale = fit_params['shape'], fit_params['loc'], fit_params['scale']
    cv = fit_params['cv']
    ks_pval = fit_params['ks_pval']
    rmse = fit_params['rmse']

    # Generate MC samples from fitted distribution
    np.random.seed(42)
    mc_samples = lognorm.rvs(shape, loc, scale, size=n_samples)

    # ========================================================================
    # PANEL 1: Histogram + Fitted PDF
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    # Histogram
    ax1.hist(values, bins=max(3, min(n, 15)), alpha=0.6, color='steelblue',
             edgecolor='black', density=True, label='Observed data', linewidth=1.2)

    # Raw data points on x-axis
    ax1.scatter(values, [0]*n, color='red', s=80, zorder=5,
               marker='D', edgecolor='darkred', linewidth=1, label='Raw values')

    # Fitted PDF
    x_range = np.linspace(max(0.001, values.min()*0.5), values.max()*1.5, 500)
    fitted_pdf = lognorm.pdf(x_range, shape, loc, scale)
    ax1.plot(x_range, fitted_pdf, 'orange', linewidth=2.5, label='Fitted lognormal PDF')

    # Median and mean lines
    median_val = fit_params['median']
    mean_val = fit_params['mean']
    ax1.axvline(median_val, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Median: {median_val:.1f}')
    ax1.axvline(mean_val, color='purple', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_val:.1f}')

    ax1.set_xlabel('Intensity (g/MW)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=10, fontweight='bold')
    ax1.set_title('Histogram + Fitted PDF', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(alpha=0.3)

    # ========================================================================
    # PANEL 2: CDF Comparison
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    # Empirical CDF
    sorted_vals = np.sort(values)
    empirical_cdf = np.arange(1, n + 1) / n
    ax2.step(sorted_vals, empirical_cdf, where='post', color='steelblue',
             linewidth=2, label='Empirical CDF', alpha=0.8)
    ax2.scatter(sorted_vals, empirical_cdf, color='steelblue', s=60, zorder=5, edgecolor='black')

    # Fitted CDF
    x_cdf = np.linspace(max(0.001, values.min()*0.5), values.max()*1.2, 500)
    fitted_cdf = lognorm.cdf(x_cdf, shape, loc, scale)
    ax2.plot(x_cdf, fitted_cdf, 'orange', linewidth=2.5, label='Fitted lognormal CDF')

    ax2.set_xlabel('Intensity (g/MW)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=10, fontweight='bold')
    ax2.set_title('CDF Comparison', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1])

    # ========================================================================
    # PANEL 3: Q-Q Plot (Log Scale)
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    # Theoretical quantiles from fitted lognormal
    theoretical_quantiles = lognorm.ppf(empirical_cdf, shape, loc, scale)

    # Q-Q plot
    ax3.scatter(theoretical_quantiles, sorted_vals, s=80, alpha=0.7,
               color='steelblue', edgecolor='black', linewidth=1)

    # 1:1 reference line
    min_val = min(theoretical_quantiles.min(), sorted_vals.min())
    max_val = max(theoretical_quantiles.max(), sorted_vals.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Perfect fit (1:1 line)', alpha=0.7)

    ax3.set_xlabel('Theoretical Quantiles (Fitted)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Observed Quantiles', fontsize=10, fontweight='bold')
    ax3.set_title('Q-Q Plot (Lognormal)', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')

    # ========================================================================
    # PANEL 4: Monte Carlo Samples Analysis
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    # Histogram of MC samples
    ax4.hist(mc_samples, bins=50, alpha=0.5, color='lightcoral',
             edgecolor='black', density=True, label=f'MC samples (n={n_samples:,})')

    # Overlay observed data for comparison
    ax4.hist(values, bins=max(3, min(n, 10)), alpha=0.6, color='steelblue',
             edgecolor='black', density=True, label='Observed data', linewidth=1.5)

    # Fitted PDF
    x_mc = np.linspace(max(0.001, mc_samples.min()*0.5), np.percentile(mc_samples, 99), 500)
    fitted_pdf_mc = lognorm.pdf(x_mc, shape, loc, scale)
    ax4.plot(x_mc, fitted_pdf_mc, 'orange', linewidth=2.5, label='Fitted PDF', alpha=0.8)

    # Percentiles
    p2_5 = np.percentile(mc_samples, 2.5)
    p50 = np.percentile(mc_samples, 50)
    p97_5 = np.percentile(mc_samples, 97.5)

    ax4.axvline(p50, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'MC p50: {p50:.1f}')
    ax4.axvline(p2_5, color='purple', linestyle=':', linewidth=1.5, alpha=0.5,
               label=f'MC 95% CI: [{p2_5:.1f}, {p97_5:.1f}]')
    ax4.axvline(p97_5, color='purple', linestyle=':', linewidth=1.5, alpha=0.5)

    ax4.set_xlabel('Intensity (g/MW)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Probability Density', fontsize=10, fontweight='bold')
    ax4.set_title('Monte Carlo Sampling Validation', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=7, loc='upper right')
    ax4.grid(alpha=0.3)
    ax4.set_xlim(left=0)

    # ========================================================================
    # Overall Title with Key Statistics
    # ========================================================================
    quality_color = 'green' if ks_pval > 0.05 else ('orange' if ks_pval > 0.01 else 'red')
    quality_text = 'GOOD FIT' if ks_pval > 0.05 else ('MARGINAL FIT' if ks_pval > 0.01 else 'POOR FIT')

    title = f'{technology} - {material}'
    subtitle = f'n={n} | CV={cv:.2f} | KS p-value={ks_pval:.4f} ({quality_text}) | RMSE={rmse:.4f}'

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    fig.text(0.5, 0.95, subtitle, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor=quality_color, alpha=0.3))

    return fig


def generate_all_visualizations(filter_bins=None):
    """
    Generate 4-panel plots for all material-technology pairs.

    Args:
        filter_bins: List of bin names to generate (e.g., ['n1', 'n2']). If None, generate all.
    """
    print("=" * 80)
    print("BATCH GENERATION: FIT VISUALIZATIONS FOR ALL PAIRS")
    print("=" * 80)
    print()

    # Load data
    print("Loading intensity data...")
    df = load_intensity_data()

    # Count pairs
    pair_counts = df.groupby(['Technology', 'Material']).size().reset_index(name='n')
    print(f"Found {len(pair_counts)} material-technology pairs\n")

    # Create output directories
    base_outdir = ROOT / 'outputs' / 'fitting_sample_size_diagnostics' / 'fit_visualizations'
    base_outdir.mkdir(parents=True, exist_ok=True)

    bins = ['n1', 'n2', 'n3', 'n4-5', 'n6-10', 'n11-20', 'n20plus']
    for bin_name in bins:
        (base_outdir / bin_name).mkdir(exist_ok=True)

    # Generate plots
    index_data = []
    successful = 0
    failed = 0

    for idx, row in pair_counts.iterrows():
        tech = row['Technology']
        mat = row['Material']
        n = row['n']
        bin_name = assign_bin(n)

        # Skip if filtering by bins
        if filter_bins and bin_name not in filter_bins:
            continue

        print(f"[{idx+1}/{len(pair_counts)}] {tech} - {mat} (n={n}, bin={bin_name})...", end=' ')

        # Get values
        values = df[(df['Technology'] == tech) & (df['Material'] == mat)]['g_per_MW'].values

        # Fit distribution
        fit_params = fit_lognormal(values)

        # Create plot
        fig = create_4panel_plot(tech, mat, values, fit_params)

        # Save
        safe_tech = tech.replace(' ', '_').replace('/', '-')
        safe_mat = mat.replace(' ', '_').replace('/', '-')
        filename = f'{safe_tech}_{safe_mat}_n{n}.png'
        outfile = base_outdir / bin_name / filename

        fig.savefig(outfile, dpi=200, bbox_inches='tight')
        plt.close(fig)

        # Track for index
        if fit_params:
            index_data.append({
                'Technology': tech,
                'Material': mat,
                'n': n,
                'Bin': bin_name,
                'CV': fit_params['cv'],
                'KS p-value': fit_params['ks_pval'],
                'RMSE': fit_params['rmse'],
                'Fit Quality': 'Good' if fit_params['ks_pval'] > 0.05 else ('Marginal' if fit_params['ks_pval'] > 0.01 else 'Poor'),
                'File': f'{bin_name}/{filename}'
            })
            successful += 1
            print("✓")
        else:
            index_data.append({
                'Technology': tech,
                'Material': mat,
                'n': n,
                'Bin': bin_name,
                'CV': np.nan,
                'KS p-value': np.nan,
                'RMSE': np.nan,
                'Fit Quality': 'Failed',
                'File': f'{bin_name}/{filename}'
            })
            failed += 1
            print("✗ (fit failed)")

    # Create index file
    print()
    print("Creating index file...")
    index_df = pd.DataFrame(index_data)
    index_df = index_df.sort_values(['Bin', 'KS p-value'], ascending=[True, False])

    index_file = base_outdir / 'fit_visualizations_index.md'
    with open(index_file, 'w') as f:
        f.write("# Fit Visualizations Index\n\n")
        f.write(f"**Total pairs:** {len(pair_counts)}\n")
        f.write(f"**Successfully fitted:** {successful}\n")
        f.write(f"**Failed to fit:** {failed}\n\n")

        f.write("## Summary by Sample Size Bin\n\n")
        bin_summary = index_df.groupby('Bin').agg({
            'n': ['count', 'mean'],
            'CV': 'median',
            'KS p-value': 'median',
            'RMSE': 'median'
        }).round(3)
        f.write(bin_summary.to_markdown())
        f.write("\n\n")

        f.write("## All Pairs (sorted by bin, then fit quality)\n\n")
        display_df = index_df[['Technology', 'Material', 'n', 'Bin', 'CV', 'KS p-value', 'RMSE', 'Fit Quality', 'File']]
        f.write(display_df.to_markdown(index=False))
        f.write("\n")

    print(f"✓ Saved index: {index_file}")

    # Print summary
    print()
    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Total pairs: {len(pair_counts)}")
    print(f"Successfully fitted: {successful}")
    print(f"Failed to fit: {failed}")
    print()
    print("Output directory:")
    print(f"  {base_outdir}")
    print()
    print("Subdirectories by sample size:")
    for bin_name in bins:
        count = len(list((base_outdir / bin_name).glob('*.png')))
        print(f"  {bin_name:10s}: {count:3d} plots")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Generate fit visualizations for all material-technology pairs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diagnostics/generate_all_fit_visualizations.py
  python diagnostics/generate_all_fit_visualizations.py --bins n1 n2 n3
  python diagnostics/generate_all_fit_visualizations.py --bins n20plus
        """
    )
    parser.add_argument('--bins', nargs='+', choices=['n1', 'n2', 'n3', 'n4-5', 'n6-10', 'n11-20', 'n20plus'],
                       help='Generate only specified bins (default: all)')

    args = parser.parse_args()

    generate_all_visualizations(filter_bins=args.bins)


if __name__ == '__main__':
    main()
