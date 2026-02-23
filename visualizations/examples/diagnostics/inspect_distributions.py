"""
Interactive Distribution Inspector
===================================

Comprehensive visualization tool to inspect:
1. Fitted parametric distributions
2. Monte Carlo sampling results
3. Goodness-of-fit statistics
4. Tail behavior analysis

Usage:
    python inspect_distributions.py --material Cement --technology Gas
    python inspect_distributions.py --material Aluminum --technology ASIGE
    python inspect_distributions.py --show_all  # Show all problematic distributions
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
from matplotlib.patches import Rectangle
from data_ingestion import load_all_data
from distribution_fitting import DistributionFitter
from scipy import stats

def create_distribution_plot(technology, material, dist_obj, output_path=None):
    """
    Create comprehensive 4-panel plot for a single distribution.

    Panels:
    1. Top-left: Histogram + Fitted PDF
    2. Top-right: CDF Comparison (empirical vs fitted)
    3. Bottom-left: Q-Q Plot
    4. Bottom-right: Monte Carlo samples analysis
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Get raw data
    raw_data = dist_obj.raw_data
    n = len(raw_data)

    # Generate Monte Carlo samples
    mc_samples = dist_obj.sample(10000, random_state=42)

    # Get fitted distribution
    best_fit = dist_obj.best_fit
    use_parametric = dist_obj.use_parametric

    # Get scipy distribution object
    if best_fit.distribution_name == 'lognormal':
        fitted_dist = stats.lognorm(
            s=best_fit.parameters['s'],
            loc=best_fit.parameters.get('loc', 0),
            scale=best_fit.parameters['scale']
        )
    elif best_fit.parameters['a']:
        fitted_dist = stats.gamma(
            a=best_fit.parameters['a'],
            loc=best_fit.parameters.get('loc', 0),
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
    else:
        fitted_dist = None

    # ========================================================================
    # PANEL 1: Histogram + Fitted PDF
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    # Histogram of raw data
    ax1.hist(raw_data, bins=max(3, min(n, 15)), alpha=0.6, color='steelblue',
             edgecolor='black', density=True, label='Raw data', linewidth=1.5)

    # Overlay raw data points
    for val in raw_data:
        ax1.axvline(val, color='red', alpha=0.3, linewidth=2, ymax=0.1)
    ax1.scatter(raw_data, [0]*n, color='red', s=150, zorder=5,
               marker='D', edgecolor='darkred', linewidth=1.5, label='Raw data points')

    # Fitted PDF
    if fitted_dist:
        x_range = np.linspace(max(0, np.min(raw_data) * 0.8),
                              np.max(raw_data) * 1.2, 300)
        pdf = fitted_dist.pdf(x_range)
        ax1.plot(x_range, pdf, 'g-', linewidth=3, label=f'Fitted {best_fit.distribution_name}')

        # Mark mean and median of fitted distribution
        if best_fit.distribution_name != 'uniform':
            try:
                fitted_mean = fitted_dist.mean()
                fitted_median = fitted_dist.median()
                ax1.axvline(fitted_mean, color='green', linestyle='--', linewidth=2,
                           alpha=0.7, label=f'Fitted mean ({fitted_mean:.2f})')
                ax1.axvline(fitted_median, color='darkgreen', linestyle=':', linewidth=2,
                           alpha=0.7, label=f'Fitted median ({fitted_median:.2f})')
            except:
                pass

    ax1.set_xlabel('Material Intensity (t/MW)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax1.set_title(f'Panel 1: Histogram & Fitted PDF\n{technology} + {material} (n={n})',
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(alpha=0.3, linestyle='--')

    # ========================================================================
    # PANEL 2: CDF Comparison
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    # Empirical CDF
    sorted_raw = np.sort(raw_data)
    ecdf = np.arange(1, n + 1) / n
    ax2.step(sorted_raw, ecdf, where='post', linewidth=3,
            color='steelblue', label='Empirical CDF', alpha=0.8)

    # Fitted CDF
    if fitted_dist:
        x_range = np.linspace(max(0, np.min(raw_data) * 0.8),
                              np.max(raw_data) * 1.2, 300)
        cdf = fitted_dist.cdf(x_range)
        ax2.plot(x_range, cdf, 'g-', linewidth=3, label=f'Fitted {best_fit.distribution_name} CDF',
                alpha=0.8)

        # Show KS statistic visually
        ks_stat = best_fit.ks_statistic
        ks_pval = best_fit.ks_pvalue
        ks_status = "PASS" if ks_pval > 0.05 else "FAIL"
        ks_color = 'green' if ks_pval > 0.05 else 'red'

        stats_text = (f'KS statistic: {ks_stat:.4f}\n'
                     f'KS p-value: {ks_pval:.4f} [{ks_status}]\n'
                     f'AIC: {best_fit.aic:.2f}\n'
                     f'BIC: {best_fit.bic:.2f}')
        ax2.text(0.05, 0.95, stats_text,
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor=ks_color, linewidth=2))

    ax2.set_xlabel('Material Intensity (t/MW)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax2.set_title(f'Panel 2: CDF Comparison\n(Goodness of Fit)',
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 1)

    # ========================================================================
    # PANEL 3: Q-Q Plot
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    if fitted_dist:
        # Theoretical quantiles
        theoretical_quantiles = fitted_dist.ppf(np.linspace(0.01, 0.99, n))

        # Q-Q plot
        ax3.scatter(theoretical_quantiles, sorted_raw, s=100, alpha=0.7,
                   color='steelblue', edgecolor='black', linewidth=1.5)

        # 45-degree line
        qq_min = min(np.min(theoretical_quantiles), np.min(sorted_raw))
        qq_max = max(np.max(theoretical_quantiles), np.max(sorted_raw))
        ax3.plot([qq_min, qq_max], [qq_min, qq_max], 'r--', linewidth=2,
                label='Perfect fit line')

        # Calculate R²
        r_squared = np.corrcoef(theoretical_quantiles, sorted_raw)[0, 1] ** 2
        ax3.text(0.05, 0.95, f'R² = {r_squared:.4f}',
                transform=ax3.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax3.set_xlabel(f'Theoretical Quantiles ({best_fit.distribution_name})', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Sample Quantiles', fontsize=12, fontweight='bold')
    ax3.set_title(f'Panel 3: Q-Q Plot\n(Fit Quality Check)',
                 fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3, linestyle='--')

    # ========================================================================
    # PANEL 4: Monte Carlo Samples Analysis
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    # Histogram of MC samples (clip at 99th percentile for visualization)
    mc_99 = np.percentile(mc_samples, 99)
    mc_clipped = mc_samples[mc_samples <= mc_99]

    bins = np.linspace(0, mc_99, 50)
    ax4.hist(mc_clipped, bins=bins, alpha=0.6, color='orange',
            edgecolor='black', density=True, label=f'MC samples (n=10,000)\n99th percentile shown')

    # Overlay raw data points
    for val in raw_data:
        if val <= mc_99:
            ax4.axvline(val, color='red', alpha=0.3, linewidth=2, ymax=0.1)
    raw_in_range = raw_data[raw_data <= mc_99]
    ax4.scatter(raw_in_range, [0]*len(raw_in_range), color='red', s=150, zorder=5,
               marker='D', edgecolor='darkred', linewidth=1.5, label='Raw data points')

    # Percentile markers
    for p, style, alpha_val in [(50, '-', 0.9), (95, '--', 0.7), (99, ':', 0.7)]:
        val = np.percentile(mc_samples, p)
        if val <= mc_99:
            ax4.axvline(val, color='purple', linestyle=style, linewidth=2, alpha=alpha_val)
            ax4.text(val, ax4.get_ylim()[1] * 0.95, f'p{p}', fontsize=9,
                    ha='center', color='purple', fontweight='bold')

    ax4.set_xlabel('Material Intensity (t/MW)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax4.set_title(f'Panel 4: Monte Carlo Samples\n(10,000 draws)',
                 fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9, loc='upper right')
    ax4.grid(alpha=0.3, linestyle='--')

    # ========================================================================
    # PANEL 5: Tail Behavior Analysis
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, :])

    # Percentile analysis
    percentiles = [1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5, 99.9]
    pct_values = np.percentile(mc_samples, percentiles)

    # Create bars
    colors = ['green' if v / pct_values[5] < 10 else 'orange' if v / pct_values[5] < 100 else 'red'
              for v in pct_values]

    bars = ax5.bar(range(len(percentiles)), pct_values, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.5)

    # Add ratio labels
    median_val = pct_values[5]  # p50
    for i, (bar, val) in enumerate(zip(bars, pct_values)):
        ratio = val / median_val if median_val > 0 else 0
        label = f'{val:.2f}\n({ratio:.1f}×)'
        ax5.text(bar.get_x() + bar.get_width() / 2, val, label,
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax5.set_xticks(range(len(percentiles)))
    ax5.set_xticklabels([f'p{p}' for p in percentiles], fontsize=10)
    ax5.set_xlabel('Percentile', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Material Intensity (t/MW)', fontsize=12, fontweight='bold')
    ax5.set_title(f'Panel 5: Tail Behavior Analysis\n(Ratio to median shown in parentheses)',
                 fontsize=13, fontweight='bold')
    ax5.grid(alpha=0.3, linestyle='--', axis='y')

    # Add status indicator
    max_ratio = np.max(pct_values) / median_val if median_val > 0 else 0
    cv = (np.std(mc_samples) / np.mean(mc_samples)) * 100 if np.mean(mc_samples) > 0 else 0

    if max_ratio < 10 and cv < 100:
        status = "✓ HEALTHY"
        status_color = 'green'
    elif max_ratio < 100 and cv < 200:
        status = "⚠️ MODERATE"
        status_color = 'orange'
    else:
        status = "❌ EXTREME"
        status_color = 'red'

    status_text = (f'Distribution Status: {status}\n'
                  f'Max/Median: {max_ratio:.1f}×\n'
                  f'CV: {cv:.1f}%\n'
                  f'Mean: {np.mean(mc_samples):.2f}\n'
                  f'Median: {median_val:.2f}\n'
                  f'Std: {np.std(mc_samples):.2f}')

    ax5.text(0.98, 0.97, status_text,
            transform=ax5.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                     edgecolor=status_color, linewidth=3))

    # Overall title
    fig.suptitle(f'Distribution Analysis: {technology} + {material}\n'
                 f'Distribution: {best_fit.distribution_name} | n={n} | {dist_obj.recommendation}',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    return fig

def main():
    parser = argparse.ArgumentParser(description='Inspect fitted distributions and Monte Carlo results')
    parser.add_argument('--material', type=str, help='Material name')
    parser.add_argument('--technology', type=str, help='Technology name')
    parser.add_argument('--show_all', action='store_true', help='Show all problematic distributions')
    parser.add_argument('--outdir', type=str, default='../outputs/distribution_inspection',
                       help='Output directory for plots')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    print("="*80)
    print("DISTRIBUTION INSPECTION TOOL")
    print("="*80)

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

    print(f"\n3. Distribution fitting complete:")
    n_total = len(fitted_dists)
    n_parametric = sum(1 for d in fitted_dists.values() if d.use_parametric)
    print(f"   Total combinations: {n_total}")
    print(f"   Using parametric: {n_parametric} ({100*n_parametric/n_total:.1f}%)")
    print(f"   Using empirical: {n_total - n_parametric} ({100*(n_total-n_parametric)/n_total:.1f}%)")

    # Specific material/technology
    if args.material and args.technology:
        print(f"\n4. Creating visualization for {args.technology} + {args.material}...")

        key = (args.technology, args.material)
        if key in fitted_dists:
            dist_obj = fitted_dists[key]
            output_path = os.path.join(args.outdir,
                                      f'{args.technology}_{args.material}_inspection.png')
            create_distribution_plot(args.technology, args.material, dist_obj, output_path)
            plt.show()
        else:
            print(f"   ✗ {args.technology} + {args.material} not found!")
            print(f"   Available combinations:")
            for tech, mat in sorted(fitted_dists.keys())[:20]:
                print(f"     - {tech} + {mat}")

    # Show all problematic distributions
    elif args.show_all:
        print(f"\n4. Finding problematic distributions...")

        problematic = []
        for (tech, mat), dist_obj in fitted_dists.items():
            # Sample to check tail behavior
            sample = dist_obj.sample(10000, random_state=42)
            median_val = np.median(sample)
            max_val = np.max(sample)
            ratio = max_val / median_val if median_val > 0 else 0

            if ratio > 10 or not dist_obj.use_parametric:
                problematic.append((tech, mat, dist_obj, ratio))

        print(f"   Found {len(problematic)} problematic distributions")

        # Create plots for top 10 most problematic
        for i, (tech, mat, dist_obj, ratio) in enumerate(sorted(problematic, key=lambda x: x[3], reverse=True)[:10], 1):
            print(f"\n   [{i}/10] Creating plot for {tech} + {mat} (max/median: {ratio:.1f}×)...")
            output_path = os.path.join(args.outdir, f'{i:02d}_{tech}_{mat}_inspection.png')
            fig = create_distribution_plot(tech, mat, dist_obj, output_path)
            plt.close(fig)

        print(f"\n✓ All plots saved to: {args.outdir}")

    else:
        print("\nUsage:")
        print("  Inspect specific distribution:")
        print("    python inspect_distributions.py --material Cement --technology Gas")
        print("\n  Show all problematic distributions:")
        print("    python inspect_distributions.py --show_all")

    print("\n" + "="*80)
    print("INSPECTION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
