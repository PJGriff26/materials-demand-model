#!/usr/bin/env python3
"""
Compare Distribution Fits for High-N Material-Technology Pairs
===============================================================

Generates 4-panel comparison visualizations showing:
1. Histogram + fitted PDFs (lognormal, gamma, truncated_normal, uniform)
2. Empirical vs fitted CDFs
3. Q-Q plots for each distribution type
4. Fit quality metrics table (AIC, KS p-value, RMSE)

Helps assess whether AIC-selected distributions are visually better fits
than alternatives (especially lognormal vs gamma).

Usage:
    python diagnostics/compare_distribution_fits.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import lognorm, gamma as gamma_dist, truncnorm, uniform, kstest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_ingestion import load_all_data
from src.technology_mapping import TECHNOLOGY_CONSOLIDATION

# Output directory
OUTPUT_DIR = ROOT / 'outputs' / 'fitting_sample_size_diagnostics' / 'fit_comparisons'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_all_distributions(values):
    """
    Fit all 4 distribution types and compute fit quality metrics.

    Returns dict with keys: lognormal, gamma, truncated_normal, uniform
    Each containing: {params, aic, ks_pval, rmse, fitted_dist}
    """
    n = len(values)
    results = {}

    # Sort values for Q-Q and CDF plots
    sorted_vals = np.sort(values)
    empirical_cdf = np.arange(1, n + 1) / n

    # 1. Lognormal
    try:
        shape, loc, scale = lognorm.fit(values, floc=0)
        dist = lognorm(s=shape, loc=loc, scale=scale)
        log_likelihood = np.sum(dist.logpdf(values))
        aic = 2 * 3 - 2 * log_likelihood  # 3 parameters
        ks_stat, ks_pval = kstest(values, lambda x: dist.cdf(x))
        fitted_cdf = dist.cdf(sorted_vals)
        rmse = np.sqrt(np.mean((empirical_cdf - fitted_cdf)**2))

        results['lognormal'] = {
            'params': {'shape': shape, 'loc': loc, 'scale': scale},
            'aic': aic,
            'ks_pval': ks_pval,
            'rmse': rmse,
            'dist': dist,
            'color': '#e74c3c',  # Red
            'label': 'Lognormal'
        }
    except:
        results['lognormal'] = None

    # 2. Gamma
    try:
        a, loc, scale = gamma_dist.fit(values)
        dist = gamma_dist(a=a, loc=loc, scale=scale)
        log_likelihood = np.sum(dist.logpdf(values))
        aic = 2 * 3 - 2 * log_likelihood
        ks_stat, ks_pval = kstest(values, lambda x: dist.cdf(x))
        fitted_cdf = dist.cdf(sorted_vals)
        rmse = np.sqrt(np.mean((empirical_cdf - fitted_cdf)**2))

        results['gamma'] = {
            'params': {'a': a, 'loc': loc, 'scale': scale},
            'aic': aic,
            'ks_pval': ks_pval,
            'rmse': rmse,
            'dist': dist,
            'color': '#3498db',  # Blue
            'label': 'Gamma'
        }
    except:
        results['gamma'] = None

    # 3. Truncated Normal
    try:
        from scipy.optimize import minimize

        data_mean = np.mean(values)
        data_std = np.std(values, ddof=1)

        def truncnorm_objective(params):
            loc, scale = params
            if scale <= 0:
                return 1e10
            a = (0 - loc) / scale
            try:
                dist = truncnorm(a=a, b=np.inf, loc=loc, scale=scale)
                dist_mean = dist.mean()
                dist_std = dist.std()
                return (dist_mean - data_mean)**2 + (dist_std - data_std)**2
            except:
                return 1e10

        result = minimize(truncnorm_objective, [data_mean, data_std], method='Nelder-Mead')
        loc_fit, scale_fit = result.x
        a_fit = (0 - loc_fit) / scale_fit

        dist = truncnorm(a=a_fit, b=np.inf, loc=loc_fit, scale=scale_fit)
        log_likelihood = np.sum(dist.logpdf(values))
        aic = 2 * 2 - 2 * log_likelihood  # 2 parameters (loc, scale; a is derived)
        ks_stat, ks_pval = kstest(values, lambda x: dist.cdf(x))
        fitted_cdf = dist.cdf(sorted_vals)
        rmse = np.sqrt(np.mean((empirical_cdf - fitted_cdf)**2))

        results['truncated_normal'] = {
            'params': {'a': a_fit, 'loc': loc_fit, 'scale': scale_fit},
            'aic': aic,
            'ks_pval': ks_pval,
            'rmse': rmse,
            'dist': dist,
            'color': '#2ecc71',  # Green
            'label': 'Truncated Normal'
        }
    except:
        results['truncated_normal'] = None

    # 4. Uniform
    try:
        min_val, max_val = np.min(values), np.max(values)
        loc = min_val
        scale = max_val - min_val

        dist = uniform(loc=loc, scale=scale)
        log_likelihood = np.sum(dist.logpdf(values))
        aic = 2 * 2 - 2 * log_likelihood  # 2 parameters
        ks_stat, ks_pval = kstest(values, lambda x: dist.cdf(x))
        fitted_cdf = dist.cdf(sorted_vals)
        rmse = np.sqrt(np.mean((empirical_cdf - fitted_cdf)**2))

        results['uniform'] = {
            'params': {'loc': loc, 'scale': scale},
            'aic': aic,
            'ks_pval': ks_pval,
            'rmse': rmse,
            'dist': dist,
            'color': '#95a5a6',  # Gray
            'label': 'Uniform'
        }
    except:
        results['uniform'] = None

    return results


def create_comparison_plot(technology, material, values, n, output_path):
    """Create 4-panel comparison plot."""

    # Fit all distributions
    fits = fit_all_distributions(values)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{technology} - {material} (n={n})', fontsize=16, fontweight='bold')

    # Panel 1: Histogram + PDFs
    ax1 = axes[0, 0]
    ax1.hist(values, bins=min(30, n//2), density=True, alpha=0.3, color='black', edgecolor='black')

    x_range = np.linspace(min(values) * 0.9, max(values) * 1.1, 500)

    for dist_name, fit_info in fits.items():
        if fit_info is not None:
            pdf_vals = fit_info['dist'].pdf(x_range)
            ax1.plot(x_range, pdf_vals, color=fit_info['color'],
                    linewidth=2, label=f"{fit_info['label']} (AIC={fit_info['aic']:.1f})")

    ax1.set_xlabel('Intensity (t/MW)', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Histogram + Fitted PDFs', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, frameon=True, fancybox=True)
    ax1.grid(True, alpha=0.3)

    # Panel 2: CDF Comparison
    ax2 = axes[0, 1]
    sorted_vals = np.sort(values)
    empirical_cdf = np.arange(1, n + 1) / n
    ax2.plot(sorted_vals, empirical_cdf, 'ko', markersize=4, label='Empirical', alpha=0.6)

    for dist_name, fit_info in fits.items():
        if fit_info is not None:
            fitted_cdf = fit_info['dist'].cdf(sorted_vals)
            ax2.plot(sorted_vals, fitted_cdf, color=fit_info['color'],
                    linewidth=2, label=f"{fit_info['label']} (KS p={fit_info['ks_pval']:.3f})")

    ax2.set_xlabel('Intensity (t/MW)', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title('Empirical vs Fitted CDFs', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, frameon=True, fancybox=True)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Q-Q Plots (4 small subplots)
    ax3 = axes[1, 0]
    ax3.axis('off')

    # Create mini grid for Q-Q plots
    gs = axes[1, 0].get_gridspec()
    ax3.remove()
    subfig = fig.add_subfigure(gs[1, 0])
    subaxes = subfig.subplots(2, 2)
    subfig.suptitle('Q-Q Plots', fontsize=13, fontweight='bold')

    for idx, (dist_name, fit_info) in enumerate(fits.items()):
        row, col = idx // 2, idx % 2
        ax_qq = subaxes[row, col]

        if fit_info is not None:
            # Generate theoretical quantiles
            theoretical_quantiles = fit_info['dist'].ppf(empirical_cdf)
            ax_qq.scatter(theoretical_quantiles, sorted_vals, color=fit_info['color'], alpha=0.6, s=20)

            # Add diagonal line
            min_val = min(theoretical_quantiles.min(), sorted_vals.min())
            max_val = max(theoretical_quantiles.max(), sorted_vals.max())
            ax_qq.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.5)

            ax_qq.set_title(fit_info['label'], fontsize=10, fontweight='bold')
            ax_qq.set_xlabel('Theoretical', fontsize=9)
            ax_qq.set_ylabel('Observed', fontsize=9)
            ax_qq.grid(True, alpha=0.3)
        else:
            ax_qq.text(0.5, 0.5, 'Fit Failed', ha='center', va='center', fontsize=10)
            ax_qq.set_title(dist_name.replace('_', ' ').title(), fontsize=10)

    # Panel 4: Fit Quality Metrics Table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Prepare table data
    table_data = []
    headers = ['Distribution', 'AIC', 'KS p-value', 'RMSE', 'Winner']

    # Find best AIC
    valid_fits = {k: v for k, v in fits.items() if v is not None}
    if valid_fits:
        best_aic_dist = min(valid_fits.keys(), key=lambda k: valid_fits[k]['aic'])

        for dist_name in ['lognormal', 'gamma', 'truncated_normal', 'uniform']:
            fit_info = fits.get(dist_name)
            if fit_info is not None:
                winner = '✓' if dist_name == best_aic_dist else ''
                table_data.append([
                    fit_info['label'],
                    f"{fit_info['aic']:.2f}",
                    f"{fit_info['ks_pval']:.4f}",
                    f"{fit_info['rmse']:.4f}",
                    winner
                ])
            else:
                table_data.append([dist_name.replace('_', ' ').title(), '—', '—', '—', ''])

        # Create table
        table = ax4.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colWidths=[0.3, 0.15, 0.2, 0.15, 0.1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Highlight winner row
        winner_row = None
        for idx, row in enumerate(table_data):
            if row[4] == '✓':
                winner_row = idx + 1
                break

        if winner_row:
            for i in range(len(headers)):
                table[(winner_row, i)].set_facecolor('#d5f4e6')

        ax4.set_title('Fit Quality Metrics', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Return summary info
    if valid_fits:
        return {
            'technology': technology,
            'material': material,
            'n': n,
            'best_dist': best_aic_dist,
            'best_aic': valid_fits[best_aic_dist]['aic'],
            'best_ks_pval': valid_fits[best_aic_dist]['ks_pval']
        }
    return None


def main():
    print("=" * 80)
    print("COMPARING DISTRIBUTION FITS FOR HIGH-N PAIRS")
    print("=" * 80)
    print()

    # Load intensity data
    print("Loading intensity data...")
    data_path = ROOT / 'data' / 'intensity_data.csv'
    df = pd.read_csv(data_path)

    # Standardize columns
    df.columns = ['Technology', 'Material', 'g_per_MW']

    # Apply technology consolidation
    df['Technology'] = df['Technology'].replace(TECHNOLOGY_CONSOLIDATION)

    # Convert to t/MW
    df['t_per_MW'] = df['g_per_MW'] / 1000

    print(f"✓ Loaded {len(df)} intensity observations")

    # Compute sample sizes
    sample_sizes = df.groupby(['Technology', 'Material']).size().reset_index(name='n')

    # Get top 12 highest-n pairs
    top_pairs = sample_sizes.nlargest(12, 'n')

    print(f"\nTop 12 highest-n pairs:")
    print(top_pairs.to_string(index=False))
    print()

    # Generate comparison plots
    print("=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)
    print()

    summary_results = []

    for idx, row in top_pairs.iterrows():
        tech, mat, n = row['Technology'], row['Material'], row['n']

        # Get data for this pair
        pair_data = df[(df['Technology'] == tech) & (df['Material'] == mat)]['t_per_MW'].values

        # Create safe filename
        safe_tech = tech.replace(' ', '_').replace('/', '-')
        safe_mat = mat.replace(' ', '_').replace('/', '-')
        output_file = OUTPUT_DIR / f"compare_fits_{safe_tech}_{safe_mat}.png"

        print(f"Generating: {tech} - {mat} (n={n})...")
        result = create_comparison_plot(tech, mat, pair_data, n, output_file)

        if result:
            summary_results.append(result)
            print(f"  ✓ Best fit: {result['best_dist']} (AIC={result['best_aic']:.2f}, KS p={result['best_ks_pval']:.4f})")

        print(f"  Saved: {output_file.name}")
        print()

    # Create summary table
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()

    summary_df = pd.DataFrame(summary_results)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_path = OUTPUT_DIR / 'fit_comparison_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Summary saved: {summary_path}")

    # Distribution winner counts
    print("\n" + "=" * 80)
    print("DISTRIBUTION WINNER BREAKDOWN")
    print("=" * 80)

    winner_counts = summary_df['best_dist'].value_counts()
    print("\nBest distribution by AIC (top 12 highest-n pairs):")
    for dist_name, count in winner_counts.items():
        pct = 100 * count / len(summary_df)
        print(f"  {dist_name:20s}: {count:2d} ({pct:5.1f}%)")

    print("\n" + "=" * 80)
    print(f"COMPLETE - {len(summary_results)} comparison plots generated")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
