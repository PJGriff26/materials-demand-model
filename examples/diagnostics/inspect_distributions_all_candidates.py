"""
Distribution Inspector - All Candidates View
=============================================

Shows ALL candidate distributions that were fitted, highlighting which one was selected.

Usage:
    python inspect_distributions_all_candidates.py --material Cement --technology Gas
    python inspect_distributions_all_candidates.py --material Indium --technology CdTe
"""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_dir)

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch
from data_ingestion import load_all_data
from distribution_fitting import DistributionFitter
from scipy import stats

def get_scipy_dist(fit):
    """Convert DistributionFit object to scipy distribution"""
    params = fit.parameters

    # Check if parameters are empty (rejected distribution)
    if not params:
        return None

    try:
        if fit.distribution_name == 'lognormal':
            return stats.lognorm(
                s=params['s'],
                loc=params.get('loc', 0),
                scale=params['scale']
            )
        elif fit.distribution_name == 'gamma':
            return stats.gamma(
                a=params['a'],
                loc=params.get('loc', 0),
                scale=params['scale']
            )
        elif fit.distribution_name == 'uniform':
            return stats.uniform(
                loc=params['loc'],
                scale=params['scale']
            )
        elif fit.distribution_name == 'truncated_normal':
            return stats.truncnorm(
                a=params['a'],
                b=params['b'],
                loc=params['loc'],
                scale=params['scale']
            )
        else:
            return None
    except (KeyError, TypeError):
        # Missing required parameters - distribution was rejected
        return None

def create_all_candidates_plot(technology, material, dist_obj, output_path=None):
    """
    Create comprehensive plot showing ALL candidate distributions.

    Layout:
    - Top panels: One subplot per candidate distribution (PDF comparison)
    - Middle panel: All CDFs on same plot
    - Bottom left: Goodness-of-fit comparison table
    - Bottom right: Monte Carlo samples from SELECTED distribution
    """
    raw_data = dist_obj.raw_data
    n = len(raw_data)

    # Get all fitted distributions
    all_fits = dist_obj.fitted_distributions if hasattr(dist_obj, 'fitted_distributions') else []
    best_fit = dist_obj.best_fit

    if not all_fits:
        print(f"  No fitted distributions found for {technology} + {material}")
        return

    n_candidates = len(all_fits)

    # Create figure with dynamic layout based on number of candidates
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(4, max(4, n_candidates), figure=fig, hspace=0.35, wspace=0.3,
                          height_ratios=[1.2, 1.2, 0.8, 1])

    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Data range for plotting
    x_min = max(0, np.min(raw_data) * 0.9)
    x_max = np.max(raw_data) * 1.1
    x_range = np.linspace(x_min, x_max, 500)

    # ========================================================================
    # TOP PANELS: Individual PDFs for each candidate
    # ========================================================================
    for i, fit in enumerate(all_fits):
        if i >= 4:  # Limit to 4 panels to avoid overcrowding
            break

        ax = fig.add_subplot(gs[0, i])

        # Is this the selected distribution?
        is_selected = (fit.distribution_name == best_fit.distribution_name and
                      fit.aic == best_fit.aic)

        # Histogram of raw data
        ax.hist(raw_data, bins=max(3, min(n, 10)), alpha=0.4, color='gray',
               edgecolor='black', density=True, label='Raw data', linewidth=1)

        # Raw data points
        ax.scatter(raw_data, [0]*n, color='red', s=100, zorder=5,
                  marker='D', edgecolor='darkred', linewidth=1, alpha=0.6)

        # Fitted PDF
        fitted_dist = get_scipy_dist(fit)
        if fitted_dist:
            pdf = fitted_dist.pdf(x_range)
            color = 'green' if is_selected else colors[i % len(colors)]
            linewidth = 4 if is_selected else 2.5
            linestyle = '-' if is_selected else '--'

            ax.plot(x_range, pdf, color=color, linewidth=linewidth,
                   linestyle=linestyle, label=f'{fit.distribution_name}', alpha=0.9)

        # Title with selection indicator
        title_prefix = "✓ SELECTED: " if is_selected else ""
        title_color = 'green' if is_selected else 'black'
        ax.set_title(f'{title_prefix}{fit.distribution_name.upper()}\nAIC={fit.aic:.1f}, KS p={fit.ks_pvalue:.3f}',
                    fontsize=11, fontweight='bold', color=title_color)

        ax.set_xlabel('Intensity (t/MW)', fontsize=10)
        ax.set_ylabel('PDF', fontsize=10)
        ax.grid(alpha=0.3, linestyle=':')
        ax.legend(fontsize=9, loc='upper right')

        # Highlight selected with green border
        if is_selected:
            for spine in ax.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)

    # ========================================================================
    # MIDDLE PANELS: All CDFs on same plot (spanning 2 columns)
    # ========================================================================
    ax_cdf = fig.add_subplot(gs[1, :2])

    # Empirical CDF
    sorted_raw = np.sort(raw_data)
    ecdf = np.arange(1, n + 1) / n
    ax_cdf.step(sorted_raw, ecdf, where='post', linewidth=3,
               color='black', label='Empirical CDF', alpha=0.7, linestyle='-.')

    # All fitted CDFs
    for i, fit in enumerate(all_fits):
        fitted_dist = get_scipy_dist(fit)
        if fitted_dist:
            is_selected = (fit.distribution_name == best_fit.distribution_name and
                          fit.aic == best_fit.aic)

            cdf = fitted_dist.cdf(x_range)
            color = 'green' if is_selected else colors[i % len(colors)]
            linewidth = 3.5 if is_selected else 2
            linestyle = '-' if is_selected else '--'
            alpha = 0.9 if is_selected else 0.6

            label = f'{fit.distribution_name}'
            if is_selected:
                label = f'✓ {label} (SELECTED)'

            ax_cdf.plot(x_range, cdf, color=color, linewidth=linewidth,
                       linestyle=linestyle, alpha=alpha, label=label)

    ax_cdf.set_xlabel('Intensity (t/MW)', fontsize=11, fontweight='bold')
    ax_cdf.set_ylabel('CDF', fontsize=11, fontweight='bold')
    ax_cdf.set_title(f'CDF Comparison: All Candidates\n{technology} + {material} (n={n})',
                    fontsize=12, fontweight='bold')
    ax_cdf.legend(fontsize=10, loc='lower right', framealpha=0.9)
    ax_cdf.grid(alpha=0.3, linestyle='--')
    ax_cdf.set_ylim([-0.05, 1.05])

    # ========================================================================
    # MIDDLE-RIGHT: Q-Q Plot for SELECTED distribution
    # ========================================================================
    ax_qq = fig.add_subplot(gs[1, 2:])

    fitted_dist = get_scipy_dist(best_fit)
    if fitted_dist:
        # Theoretical quantiles
        theoretical_quantiles = fitted_dist.ppf(np.linspace(0.01, 0.99, 100))
        # Empirical quantiles
        empirical_quantiles = np.percentile(raw_data, np.linspace(1, 99, 100))

        ax_qq.scatter(theoretical_quantiles, empirical_quantiles, s=100,
                     color='green', alpha=0.6, edgecolor='darkgreen', linewidth=1.5)

        # Perfect fit line
        min_val = min(np.min(theoretical_quantiles), np.min(empirical_quantiles))
        max_val = max(np.max(theoretical_quantiles), np.max(empirical_quantiles))
        ax_qq.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                  label='Perfect fit', alpha=0.7)

        ax_qq.set_xlabel(f'Theoretical Quantiles ({best_fit.distribution_name})',
                        fontsize=11, fontweight='bold')
        ax_qq.set_ylabel('Empirical Quantiles', fontsize=11, fontweight='bold')
        ax_qq.set_title(f'Q-Q Plot: {best_fit.distribution_name.upper()} (Selected)',
                       fontsize=12, fontweight='bold', color='green')
        ax_qq.legend(fontsize=10)
        ax_qq.grid(alpha=0.3, linestyle='--')

    # ========================================================================
    # BOTTOM-LEFT: Comparison Table
    # ========================================================================
    ax_table = fig.add_subplot(gs[2:, :2])
    ax_table.axis('off')

    # Create comparison data
    table_data = []
    headers = ['Distribution', 'AIC ↓', 'BIC ↓', 'KS stat ↓', 'KS p-val ↑', 'Selected']

    for fit in sorted(all_fits, key=lambda x: x.aic):
        is_selected = (fit.distribution_name == best_fit.distribution_name and
                      fit.aic == best_fit.aic)

        row = [
            fit.distribution_name,
            f'{fit.aic:.2f}',
            f'{fit.bic:.2f}',
            f'{fit.ks_statistic:.4f}',
            f'{fit.ks_pvalue:.4f}',
            '✓ YES' if is_selected else ''
        ]
        table_data.append(row)

    # Create table
    table = ax_table.table(cellText=table_data, colLabels=headers,
                          cellLoc='center', loc='center',
                          colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=11)

    # Style rows
    for i in range(len(table_data)):
        is_selected_row = '✓ YES' in table_data[i]
        bg_color = '#D6EAD6' if is_selected_row else 'white'

        for j in range(len(headers)):
            cell = table[(i + 1, j)]
            cell.set_facecolor(bg_color)

            if is_selected_row:
                cell.set_text_props(weight='bold')

            # Highlight lowest AIC/BIC
            if j == 1 and table_data[i][j] == min(row[1] for row in table_data):
                cell.set_facecolor('#90EE90')
            if j == 2 and table_data[i][j] == min(row[2] for row in table_data):
                cell.set_facecolor('#90EE90')

    ax_table.set_title('Goodness-of-Fit Comparison\n(Lower AIC/BIC/KS is better, Higher p-value is better)',
                      fontsize=12, fontweight='bold', pad=20)

    # ========================================================================
    # BOTTOM-RIGHT: Monte Carlo Samples (from SELECTED distribution)
    # ========================================================================
    ax_mc = fig.add_subplot(gs[2:, 2:])

    # Sample 10,000 from selected distribution
    mc_samples = dist_obj.sample(10000, random_state=42)

    # Histogram
    ax_mc.hist(mc_samples, bins=50, alpha=0.6, color='green', edgecolor='black',
              density=False, label='MC samples (10k)')

    # Statistics
    mc_mean = np.mean(mc_samples)
    mc_median = np.median(mc_samples)
    mc_std = np.std(mc_samples)
    mc_max = np.max(mc_samples)
    mc_p95 = np.percentile(mc_samples, 95)

    # Mark statistics
    ax_mc.axvline(mc_mean, color='darkgreen', linestyle='--', linewidth=2.5,
                 label=f'Mean: {mc_mean:.3f}', alpha=0.8)
    ax_mc.axvline(mc_median, color='green', linestyle=':', linewidth=2.5,
                 label=f'Median: {mc_median:.3f}', alpha=0.8)

    # Overlay raw data points for comparison
    for val in raw_data:
        ax_mc.axvline(val, color='red', alpha=0.3, linewidth=1.5, ymax=0.15)
    ax_mc.scatter(raw_data, [0]*n, color='red', s=100, zorder=5,
                 marker='D', edgecolor='darkred', linewidth=1, label='Raw data', alpha=0.7)

    ax_mc.set_xlabel('Intensity (t/MW)', fontsize=11, fontweight='bold')
    ax_mc.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax_mc.set_title(f'Monte Carlo Samples: {best_fit.distribution_name.upper()}\n'
                   f'Mean={mc_mean:.3f}, Median={mc_median:.3f}, Std={mc_std:.3f}',
                   fontsize=12, fontweight='bold', color='green')
    ax_mc.legend(fontsize=10, loc='upper right')
    ax_mc.grid(alpha=0.3, linestyle='--')

    # Add tail behavior warning if needed
    ratio = mc_max / mc_median if mc_median > 0 else 0
    if ratio > 50:
        warning_text = f'⚠️  WARNING: Max/Median = {ratio:.1f}× (large tail)'
        ax_mc.text(0.5, 0.95, warning_text, transform=ax_mc.transAxes,
                  fontsize=11, color='red', weight='bold',
                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                  ha='center', va='top')
    elif ratio > 10:
        info_text = f'ℹ️  Max/Median = {ratio:.1f}× (moderate tail)'
        ax_mc.text(0.5, 0.95, info_text, transform=ax_mc.transAxes,
                  fontsize=10, color='orange', weight='bold',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                  ha='center', va='top')
    else:
        ok_text = f'✓ Max/Median = {ratio:.1f}× (healthy)'
        ax_mc.text(0.5, 0.95, ok_text, transform=ax_mc.transAxes,
                  fontsize=10, color='green', weight='bold',
                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                  ha='center', va='top')

    # Overall title
    fig.suptitle(f'Distribution Fitting Analysis: {technology} + {material}\n'
                f'Sample size: n={n}, Selected: {best_fit.distribution_name} (AIC={best_fit.aic:.2f})',
                fontsize=16, fontweight='bold', y=0.98)

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Inspect all candidate distributions that were fitted'
    )
    parser.add_argument('--material', type=str, required=True,
                       help='Material name (e.g., Cement, Aluminum)')
    parser.add_argument('--technology', type=str, required=True,
                       help='Technology name (e.g., Gas, ASIGE, onshore wind)')
    parser.add_argument('--outdir', type=str, default='../outputs/distribution_inspect',
                       help='Output directory for plots')

    args = parser.parse_args()

    print("=" * 80)
    print("DISTRIBUTION INSPECTION TOOL - ALL CANDIDATES VIEW")
    print("=" * 80)
    print()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    print("1. Loading data...")
    data = load_all_data(
        '../data/intensity_data.csv',
        '../data/StdScen24_annual_national.csv'
    )
    intensity_df = data['intensity']

    # Fit distributions
    print("2. Fitting distributions...")
    fitter = DistributionFitter()
    fitted_dists = fitter.fit_all(intensity_df)

    print()
    print("3. Distribution fitting complete:")
    n_total = len(fitted_dists)
    n_parametric = sum(1 for d in fitted_dists.values() if d.use_parametric)
    print(f"   Total combinations: {n_total}")
    print(f"   Using parametric: {n_parametric} ({100*n_parametric/n_total:.1f}%)")
    print(f"   Using empirical: {n_total - n_parametric} ({100*(n_total-n_parametric)/n_total:.1f}%)")
    print()

    # Find requested combination
    key = (args.technology, args.material)
    if key not in fitted_dists:
        print(f"❌ ERROR: Combination not found: {args.technology} + {args.material}")
        print()
        print("Available combinations:")
        for (tech, mat) in sorted(fitted_dists.keys()):
            if mat == args.material or tech == args.technology:
                print(f"  - {tech} + {mat}")
        return

    # Create visualization
    print(f"4. Creating visualization for {args.technology} + {args.material}...")
    dist_obj = fitted_dists[key]

    # Create filename
    safe_tech = args.technology.replace(' ', '_').replace('/', '-')
    safe_mat = args.material.replace(' ', '_')
    output_path = os.path.join(args.outdir, f'{safe_tech}_{safe_mat}_all_candidates.png')

    create_all_candidates_plot(args.technology, args.material, dist_obj, output_path)

    print()
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
