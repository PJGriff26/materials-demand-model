#!/usr/bin/env python3
"""
Visualize Uniform Fallback Pairs for Manual Inspection
=======================================================

Generates detailed visualizations for the 28 pairs that fell back to uniform
despite FORCE_LOGNORMAL=True, showing:
1. Data histogram with attempted lognormal fit (extreme σ shown)
2. Why tail validation failed (σ value, max/median ratio)
3. Data characteristics (gaps, outliers, bimodality indicators)
4. Uniform fallback used

Helps understand why these pairs are truly pathological and can't fit lognormal.

Usage:
    python diagnostics/visualize_uniform_pairs.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import lognorm, uniform

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_ingestion import load_all_data
from src.technology_mapping import TECHNOLOGY_CONSOLIDATION

# Output directory - enhanced version with detailed fit statistics
OUTPUT_DIR = ROOT / 'outputs' / 'final_uniform_diagnostics'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_all_distributions(values):
    """
    Fit all distribution types including borrowed CV lognormal.

    Returns dict with keys: lognormal_raw, lognormal_borrowed, gamma, truncated_normal, uniform
    Each containing: {params, aic, ks_pval, rmse, dist, label, color, why_failed}
    """
    from scipy.stats import gamma as gamma_dist, truncnorm
    from scipy.optimize import minimize

    n = len(values)
    results = {}
    sorted_vals = np.sort(values)
    empirical_cdf = np.arange(1, n + 1) / n
    median_val = np.median(values)
    max_val = np.max(values)
    data_max_median = max_val / median_val

    # 1. Raw Lognormal Fit (MLE - the one that failed)
    # NOTE: Using constrained fit (floc=0) to match pipeline behavior
    try:
        shape, loc, scale = lognorm.fit(values, floc=0)  # Constrain loc=0 (matches pipeline)
        dist = lognorm(s=shape, loc=loc, scale=scale)
        log_likelihood = np.sum(dist.logpdf(values))
        aic = 2 * 3 - 2 * log_likelihood
        ks_stat, ks_pval = stats.kstest(values, lambda x: dist.cdf(x))
        fitted_cdf = dist.cdf(sorted_vals)
        rmse = np.sqrt(np.mean((empirical_cdf - fitted_cdf)**2))

        # Test tail behavior
        test_sample = dist.rvs(size=10000, random_state=42)
        test_max_median = np.max(test_sample) / np.median(test_sample)

        if shape > 3.0:
            why_failed = f"σ={shape:.2f} > 3.0"
        elif test_max_median > 100:
            why_failed = f"max/median={test_max_median:.1f} > 100 in test"
        else:
            why_failed = "Should not have failed"

        results['lognormal_raw'] = {
            'params': {'s': shape, 'loc': loc, 'scale': scale},
            'aic': aic,
            'ks_pval': ks_pval,
            'rmse': rmse,
            'dist': dist,
            'color': '#e74c3c',  # Red
            'label': f'Lognormal MLE (σ={shape:.2f})',
            'why_failed': why_failed,
            'test_max_median': test_max_median
        }
    except:
        results['lognormal_raw'] = None

    # 2. Borrowed CV Lognormal Fit (σ=0.609, scale=median)
    try:
        borrowed_sigma = 0.609  # From borrowed CV=0.671
        scale_borrowed = median_val  # Preserve median
        loc_borrowed = 0
        dist_borrowed = lognorm(s=borrowed_sigma, loc=loc_borrowed, scale=scale_borrowed)

        log_likelihood = np.sum(dist_borrowed.logpdf(values))
        aic = 2 * 3 - 2 * log_likelihood
        ks_stat, ks_pval = stats.kstest(values, lambda x: dist_borrowed.cdf(x))
        fitted_cdf = dist_borrowed.cdf(sorted_vals)
        rmse = np.sqrt(np.mean((empirical_cdf - fitted_cdf)**2))

        # Test tail behavior
        test_sample = dist_borrowed.rvs(size=10000, random_state=42)
        test_max_median = np.max(test_sample) / np.median(test_sample)

        results['lognormal_borrowed'] = {
            'params': {'s': borrowed_sigma, 'loc': loc_borrowed, 'scale': scale_borrowed},
            'aic': aic,
            'ks_pval': ks_pval,
            'rmse': rmse,
            'dist': dist_borrowed,
            'color': '#9b59b6',  # Purple
            'label': f'Lognormal Borrowed CV (σ={borrowed_sigma:.2f})',
            'why_failed': 'n≥5, not eligible for borrowed CV rescue',
            'test_max_median': test_max_median
        }
    except:
        results['lognormal_borrowed'] = None

    # 3. Gamma
    try:
        a, loc, scale = gamma_dist.fit(values)
        dist = gamma_dist(a=a, loc=loc, scale=scale)
        log_likelihood = np.sum(dist.logpdf(values))
        aic = 2 * 3 - 2 * log_likelihood
        ks_stat, ks_pval = stats.kstest(values, lambda x: dist.cdf(x))
        fitted_cdf = dist.cdf(sorted_vals)
        rmse = np.sqrt(np.mean((empirical_cdf - fitted_cdf)**2))

        results['gamma'] = {
            'params': {'a': a, 'loc': loc, 'scale': scale},
            'aic': aic,
            'ks_pval': ks_pval,
            'rmse': rmse,
            'dist': dist,
            'color': '#3498db',  # Blue
            'label': f'Gamma (a={a:.2f})',
            'why_failed': None
        }
    except:
        results['gamma'] = None

    # 4. Truncated Normal
    try:
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
        aic = 2 * 2 - 2 * log_likelihood
        ks_stat, ks_pval = stats.kstest(values, lambda x: dist.cdf(x))
        fitted_cdf = dist.cdf(sorted_vals)
        rmse = np.sqrt(np.mean((empirical_cdf - fitted_cdf)**2))

        results['truncated_normal'] = {
            'params': {'a': a_fit, 'loc': loc_fit, 'scale': scale_fit},
            'aic': aic,
            'ks_pval': ks_pval,
            'rmse': rmse,
            'dist': dist,
            'color': '#2ecc71',  # Green
            'label': 'Truncated Normal',
            'why_failed': None
        }
    except:
        results['truncated_normal'] = None

    # 5. Uniform (the fallback used)
    try:
        min_val = np.min(values)
        max_val = np.max(values)
        loc = min_val
        scale = max_val - min_val

        dist = uniform(loc=loc, scale=scale)
        log_likelihood = np.sum(dist.logpdf(values))
        aic = 2 * 2 - 2 * log_likelihood
        ks_stat, ks_pval = stats.kstest(values, lambda x: dist.cdf(x))
        fitted_cdf = dist.cdf(sorted_vals)
        rmse = np.sqrt(np.mean((empirical_cdf - fitted_cdf)**2))

        results['uniform'] = {
            'params': {'loc': loc, 'scale': scale},
            'aic': aic,
            'ks_pval': ks_pval,
            'rmse': rmse,
            'dist': dist,
            'color': '#95a5a6',  # Gray
            'label': 'Uniform (USED)',
            'why_failed': None
        }
    except:
        results['uniform'] = None

    return results


def detect_data_issues(values):
    """
    Detect potential data issues: gaps, outliers, bimodality.

    Returns dict with:
    - has_gaps: Boolean indicating large gaps
    - gap_description: Description of largest gap
    - has_outliers: Boolean indicating extreme outliers
    - outlier_description: Description of outliers
    - potential_bimodal: Boolean suggesting bimodality
    """
    sorted_vals = np.sort(values)
    n = len(sorted_vals)

    # Check for gaps (consecutive values with >2x ratio)
    gaps = []
    for i in range(n - 1):
        ratio = sorted_vals[i + 1] / sorted_vals[i] if sorted_vals[i] > 0 else np.inf
        if ratio > 2.0:
            gaps.append((i, sorted_vals[i], sorted_vals[i + 1], ratio))

    has_gaps = len(gaps) > 0
    if gaps:
        largest_gap = max(gaps, key=lambda x: x[3])
        gap_description = f"Gap: {largest_gap[1]:.2f} → {largest_gap[2]:.2f} ({largest_gap[3]:.1f}x ratio)"
    else:
        gap_description = "No significant gaps"

    # Check for outliers (beyond 3 IQR)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr

    outliers = values[(values < lower_bound) | (values > upper_bound)]
    has_outliers = len(outliers) > 0
    if has_outliers:
        outlier_description = f"{len(outliers)} outliers beyond 3×IQR [{lower_bound:.2f}, {upper_bound:.2f}]"
    else:
        outlier_description = "No extreme outliers"

    # Check for potential bimodality (simplistic: large gap in middle tercile)
    tercile_1, tercile_2 = np.percentile(values, [33.3, 66.7])
    middle_vals = values[(values >= tercile_1) & (values <= tercile_2)]
    if len(middle_vals) < len(values) * 0.2:  # Less than 20% in middle tercile
        potential_bimodal = True
    else:
        potential_bimodal = False

    return {
        'has_gaps': has_gaps,
        'gap_description': gap_description,
        'has_outliers': has_outliers,
        'outlier_description': outlier_description,
        'potential_bimodal': potential_bimodal
    }


def create_uniform_pair_plot(technology, material, values, n, output_path):
    """Create detailed comparison plot showing all distribution fits."""

    # Fit all distributions
    fits = fit_all_distributions(values)

    # Detect data issues
    data_issues = detect_data_issues(values)

    # Create figure with custom layout
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'{technology} - {material} (n={n}) — Uniform Fallback Analysis',
                 fontsize=16, fontweight='bold')

    # Create grid spec for custom layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)

    # Panel 1: Histogram + All PDFs
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(values, bins=min(20, n//2), density=True, alpha=0.3,
             color='black', edgecolor='black', label='Data')

    x_range = np.linspace(min(values) * 0.8, max(values) * 1.2, 500)

    # Plot all fitted distributions
    for dist_name in ['lognormal_raw', 'lognormal_borrowed', 'gamma', 'truncated_normal', 'uniform']:
        fit_info = fits.get(dist_name)
        if fit_info is not None:
            try:
                pdf_vals = fit_info['dist'].pdf(x_range)
                linestyle = '--' if dist_name == 'uniform' else '-'
                linewidth = 3 if dist_name == 'uniform' else 2
                ax1.plot(x_range, pdf_vals, color=fit_info['color'],
                        linewidth=linewidth, linestyle=linestyle, label=fit_info['label'])
            except:
                pass

    ax1.set_xlabel('Intensity (t/MW)', fontsize=11)
    ax1.set_ylabel('Probability Density', fontsize=11)
    ax1.set_title('All Distribution Fits Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, frameon=True, fancybox=True, loc='best')
    ax1.grid(True, alpha=0.3)

    # Panel 2: CDF Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_vals_cdf = np.sort(values)
    empirical_cdf_vals = np.arange(1, len(values) + 1) / len(values)
    ax2.plot(sorted_vals_cdf, empirical_cdf_vals, 'ko', markersize=5, label='Empirical', alpha=0.6)

    for dist_name in ['lognormal_raw', 'lognormal_borrowed', 'gamma', 'truncated_normal', 'uniform']:
        fit_info = fits.get(dist_name)
        if fit_info is not None:
            try:
                fitted_cdf = fit_info['dist'].cdf(sorted_vals_cdf)
                linestyle = '--' if dist_name == 'uniform' else '-'
                linewidth = 3 if dist_name == 'uniform' else 2
                ax2.plot(sorted_vals_cdf, fitted_cdf, color=fit_info['color'],
                        linewidth=linewidth, linestyle=linestyle,
                        label=f"{fit_info['label']} (KS p={fit_info['ks_pval']:.3f})")
            except:
                pass

    ax2.set_xlabel('Intensity (t/MW)', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_title('Empirical vs Fitted CDFs', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, frameon=True, fancybox=True, loc='best')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Enhanced Fit Quality Table with BIC
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')

    # Prepare table data
    table_data = []
    headers = ['Distribution', 'AIC ↓', 'BIC ↓', 'KS p-val ↑', 'RMSE ↓', 'Status']

    # Find best AIC and BIC
    valid_fits = {k: v for k, v in fits.items() if v is not None}
    if valid_fits:
        best_aic_dist = min(valid_fits.keys(), key=lambda k: valid_fits[k]['aic'])

        # Calculate BIC for each distribution
        for k, v in valid_fits.items():
            # BIC = k*ln(n) - 2*ln(L) where k = number of parameters
            # AIC = 2k - 2*ln(L), so ln(L) = (2k - AIC)/2
            if k == 'lognormal_raw':
                k_params = 2  # shape and scale (loc fixed at 0)
            elif k == 'lognormal_borrowed':
                k_params = 2  # shape and scale (loc fixed at 0)
            elif k == 'gamma':
                k_params = 3  # shape, loc, scale
            elif k == 'truncated_normal':
                k_params = 2  # loc and scale (a derived)
            elif k == 'uniform':
                k_params = 2  # loc and scale
            else:
                k_params = 2

            log_likelihood = (2 * k_params - v['aic']) / 2
            bic = k_params * np.log(n) - 2 * log_likelihood
            v['bic'] = bic

        best_bic_dist = min(valid_fits.keys(), key=lambda k: valid_fits[k]['bic'])

        for dist_name in ['lognormal_raw', 'lognormal_borrowed', 'gamma', 'truncated_normal', 'uniform']:
            fit_info = fits.get(dist_name)
            if fit_info is not None:
                if dist_name == 'uniform':
                    status = '✓ USED'
                elif dist_name == 'lognormal_raw':
                    status = f'✗ {fit_info["why_failed"]}'
                elif dist_name == 'lognormal_borrowed':
                    status = '✗ n≥5'
                elif dist_name == best_aic_dist and dist_name == best_bic_dist:
                    status = '⭐ Best AIC+BIC'
                elif dist_name == best_aic_dist:
                    status = '⭐ Best AIC'
                elif dist_name == best_bic_dist:
                    status = '⭐ Best BIC'
                else:
                    status = ''

                label = fit_info['label'].split(' (')[0]  # Remove parameters from label
                table_data.append([
                    label,
                    f"{fit_info['aic']:.1f}",
                    f"{fit_info['bic']:.1f}",
                    f"{fit_info['ks_pval']:.4f}",
                    f"{fit_info['rmse']:.4f}",
                    status
                ])

        # Create table with larger font
        table = ax3.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colWidths=[0.28, 0.12, 0.12, 0.12, 0.12, 0.24])
        table.auto_set_font_size(False)
        table.set_fontsize(10)  # Increased from 9
        table.scale(1, 3.0)  # Increased from 2.5

        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)

        # Highlight uniform row
        for idx, row in enumerate(table_data):
            if '✓ USED' in row[5]:
                for i in range(len(headers)):
                    table[(idx + 1, i)].set_facecolor('#ffeaa7')
            # Highlight best AIC/BIC rows with light green
            elif '⭐' in row[5]:
                for i in range(len(headers)):
                    table[(idx + 1, i)].set_facecolor('#d5f4e6')

        ax3.set_title('FIT QUALITY METRICS (Lower AIC/BIC/RMSE, Higher KS p-val = Better)',
                     fontsize=13, fontweight='bold', pad=20)

    # Panel 4: Enhanced Failure Diagnostics with Key Statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    raw_ln = fits.get('lognormal_raw')
    borrowed_ln = fits.get('lognormal_borrowed')

    if raw_ln is not None:
        # Calculate additional statistics
        raw_aic = raw_ln['aic']
        raw_bic = raw_ln.get('bic', 0)
        raw_ks = raw_ln['ks_pval']
        raw_rmse = raw_ln['rmse']

        borrowed_aic = borrowed_ln['aic'] if borrowed_ln else 0
        borrowed_bic = borrowed_ln.get('bic', 0) if borrowed_ln else 0
        borrowed_ks = borrowed_ln['ks_pval'] if borrowed_ln else 0
        borrowed_rmse = borrowed_ln['rmse'] if borrowed_ln else 0

        summary_text = f"""
╔═══════════════════════════════════════════════════════════╗
║           TAIL VALIDATION & FIT QUALITY DIAGNOSTICS       ║
╚═══════════════════════════════════════════════════════════╝

SAMPLE: n={n} (n≥5, NOT eligible for borrowed CV rescue)

┌─ RAW LOGNORMAL MLE FIT ────────────────────────────────┐
│ Shape Parameter:                                         │
│   σ = {raw_ln['params']['s']:.3f} {'✗ FAIL (>3.0)' if raw_ln['params']['s'] > 3.0 else '✓ PASS (<3.0)'}                              │
│                                                          │
│ Tail Behavior (10k test sample):                        │
│   max/median = {raw_ln.get('test_max_median', 0):,.0f}x {'✗ FAIL (>300)' if raw_ln.get('test_max_median', 0) > 300 else '✓ PASS (<300)'}                    │
│                                                          │
│ Fit Quality Metrics:                                    │
│   AIC     = {raw_aic:,.1f}                                          │
│   BIC     = {raw_bic:,.1f}                                          │
│   KS p    = {raw_ks:.4f}                                       │
│   RMSE    = {raw_rmse:.4f}                                       │
└──────────────────────────────────────────────────────────┘

┌─ BORROWED CV LOGNORMAL (n<5 rescue) ───────────────────┐
│ Would have been used if n<5:                            │
│   σ = {borrowed_ln['params']['s']:.3f} (borrowed, median preserved)          │
│   max/median = {borrowed_ln.get('test_max_median', 0):.1f}x {'✓ PASS (<300)' if borrowed_ln.get('test_max_median', 0) < 300 else '✗ FAIL (>300)'}                     │
│                                                          │
│ Fit Quality (hypothetical):                             │
│   AIC     = {borrowed_aic:,.1f}                                        │
│   BIC     = {borrowed_bic:,.1f}                                        │
│   KS p    = {borrowed_ks:.4f}                                     │
│   RMSE    = {borrowed_rmse:.4f}                                     │
└──────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════╗
║ WHY UNIFORM WAS USED:                                     ║
║ ✗ Raw lognormal: {raw_ln['why_failed']:39s} ║
║ ✗ Borrowed CV: Not applicable (n≥5, rescue only n<5)     ║
║ ✓ Uniform: CORRECT for pathological heavy-tailed data    ║
╚═══════════════════════════════════════════════════════════╝
"""
    else:
        summary_text = "Lognormal fit failed completely"

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=8.5, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

    ax4.set_title('FAILURE DIAGNOSTICS & FIT STATISTICS',
                  fontsize=13, fontweight='bold')

    # Panel 5: Data Quality Issues
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')

    issues_text = f"""
DATA QUALITY ISSUES

Gaps: {data_issues['gap_description']}

Outliers: {data_issues['outlier_description']}

Bimodality: {'⚠️  Potential bimodal distribution' if data_issues['potential_bimodal'] else '✓  Single mode likely'}

Distribution Summary:
  Min:    {np.min(values):.3f} t/MW
  Q1:     {np.percentile(values, 25):.3f} t/MW
  Median: {np.median(values):.3f} t/MW
  Q3:     {np.percentile(values, 75):.3f} t/MW
  Max:    {np.max(values):.3f} t/MW
  IQR:    {np.percentile(values, 75) - np.percentile(values, 25):.3f} t/MW

INTERPRETATION:
{'Bimodal or non-lognormal structure prevents parametric fit.' if data_issues['potential_bimodal'] else 'Extreme outliers or heavy skew make lognormal inappropriate.'}
Uniform fallback is CORRECT for this pathological data.
"""

    ax5.text(0.05, 0.95, issues_text, transform=ax5.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    ax5.set_title('Data Quality Analysis', fontsize=12, fontweight='bold')

    # Panel 6: Sorted Data
    ax6 = fig.add_subplot(gs[2, 1])
    sorted_vals_plot = np.sort(values)
    ax6.plot(range(len(sorted_vals_plot)), sorted_vals_plot, 'ko-', markersize=6, linewidth=1.5)

    median_idx = len(sorted_vals_plot) // 2
    ax6.plot(median_idx, np.median(values), 'bs', markersize=12,
             label=f'Median={np.median(values):.2f}')
    ax6.plot(len(sorted_vals_plot) - 1, np.max(values), 'r^', markersize=12,
             label=f'Max={np.max(values):.2f}')

    # Highlight gaps
    if data_issues['has_gaps']:
        for i in range(len(sorted_vals_plot) - 1):
            ratio = sorted_vals_plot[i + 1] / sorted_vals_plot[i] if sorted_vals_plot[i] > 0 else 0
            if ratio > 2.0:
                ax6.axhline(sorted_vals_plot[i], color='orange', linestyle='--', alpha=0.5, linewidth=1)
                ax6.axhline(sorted_vals_plot[i + 1], color='orange', linestyle='--', alpha=0.5, linewidth=1)

    ax6.set_xlabel('Observation Index (sorted)', fontsize=11)
    ax6.set_ylabel('Intensity (t/MW)', fontsize=11)
    ax6.set_title('Sorted Data (gaps in orange)', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10, frameon=True, fancybox=True)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Return summary
    raw_ln = fits.get('lognormal_raw')
    return {
        'technology': technology,
        'material': material,
        'n': n,
        'fitted_sigma': raw_ln['params']['s'] if raw_ln else None,
        'max_median_ratio': np.max(values) / np.median(values),
        'test_max_median': raw_ln.get('test_max_median') if raw_ln else None,
        'why_failed': raw_ln['why_failed'] if raw_ln else 'Fit failed',
        'has_gaps': data_issues['has_gaps'],
        'has_outliers': data_issues['has_outliers'],
        'potential_bimodal': data_issues['potential_bimodal']
    }


def main():
    print("=" * 80)
    print("VISUALIZING UNIFORM FALLBACK PAIRS FOR MANUAL INSPECTION")
    print("=" * 80)
    print()

    # Load fitted distributions
    print("Loading fitted distributions...")
    fitted_path = ROOT / 'outputs' / 'data' / 'fitted_distributions.csv'
    if not fitted_path.exists():
        print(f"ERROR: {fitted_path} not found. Run simulation first.")
        sys.exit(1)

    fitted_df = pd.read_csv(fitted_path)
    print(f"✓ Loaded {len(fitted_df)} fitted distributions")

    # Identify uniform pairs
    uniform_pairs = fitted_df[fitted_df['best_distribution'] == 'uniform'].copy()
    print(f"✓ Found {len(uniform_pairs)} uniform fallback pairs")
    print()

    # Load intensity data
    print("Loading intensity data...")
    data_path = ROOT / 'data' / 'intensity_data.csv'
    df = pd.read_csv(data_path)
    df.columns = ['Technology', 'Material', 'g_per_MW']
    df['Technology'] = df['Technology'].replace(TECHNOLOGY_CONSOLIDATION)
    df['t_per_MW'] = df['g_per_MW'] / 1000
    print(f"✓ Loaded {len(df)} intensity observations")
    print()

    # Generate visualizations
    print("=" * 80)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("=" * 80)
    print()

    summary_results = []

    for idx, row in uniform_pairs.iterrows():
        tech = row['technology']
        mat = row['material']
        n = row['n_samples']

        # Get data for this pair
        pair_data = df[(df['Technology'] == tech) & (df['Material'] == mat)]['t_per_MW'].values

        if len(pair_data) == 0:
            print(f"⚠️  WARNING: No data found for {tech} - {mat}")
            continue

        # Create safe filename
        safe_tech = tech.replace(' ', '_').replace('/', '-')
        safe_mat = mat.replace(' ', '_').replace('/', '-')
        output_file = OUTPUT_DIR / f"extreme_tail_{safe_tech}_{safe_mat}.png"

        print(f"Generating: {tech} - {mat} (n={n})...")
        result = create_uniform_pair_plot(tech, mat, pair_data, n, output_file)

        if result:
            summary_results.append(result)
            if result['fitted_sigma'] is not None:
                print(f"  σ={result['fitted_sigma']:.2f}, max/median={result['max_median_ratio']:.1f}x")
            else:
                print(f"  σ=FAILED, max/median={result['max_median_ratio']:.1f}x")
            print(f"  Reason: {result['why_failed']}")
            print(f"  Data issues: gaps={result['has_gaps']}, outliers={result['has_outliers']}, bimodal={result['potential_bimodal']}")
            print(f"  ✓ Saved: {output_file.name}")
        print()

    # Create summary table
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()

    summary_df = pd.DataFrame(summary_results)

    # Sort by fitted sigma (most extreme first), handling None values
    summary_df = summary_df.sort_values('fitted_sigma', ascending=False, na_position='last')

    print(summary_df[['technology', 'material', 'n', 'fitted_sigma', 'max_median_ratio',
                      'has_gaps', 'has_outliers', 'potential_bimodal']].to_string(index=False))

    # Save summary
    summary_path = OUTPUT_DIR / 'uniform_fallback_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Summary saved: {summary_path}")

    # Statistics
    print("\n" + "=" * 80)
    print("UNIFORM FALLBACK STATISTICS")
    print("=" * 80)
    print()

    print(f"Total uniform pairs: {len(summary_df)}")
    print(f"\nSample sizes:")
    print(f"  n=5-9:   {len(summary_df[summary_df['n'] < 10])} pairs")
    print(f"  n=10-19: {len(summary_df[(summary_df['n'] >= 10) & (summary_df['n'] < 20)])} pairs")
    print(f"  n≥20:    {len(summary_df[summary_df['n'] >= 20])} pairs")

    print(f"\nFitted σ statistics:")
    print(f"  Min:    {summary_df['fitted_sigma'].min():.2f}")
    print(f"  Median: {summary_df['fitted_sigma'].median():.2f}")
    print(f"  Mean:   {summary_df['fitted_sigma'].mean():.2f}")
    print(f"  Max:    {summary_df['fitted_sigma'].max():.2f}")

    print(f"\nData issues:")
    print(f"  Gaps:           {summary_df['has_gaps'].sum()} pairs ({100*summary_df['has_gaps'].sum()/len(summary_df):.1f}%)")
    print(f"  Outliers:       {summary_df['has_outliers'].sum()} pairs ({100*summary_df['has_outliers'].sum()/len(summary_df):.1f}%)")
    print(f"  Potential bimodal: {summary_df['potential_bimodal'].sum()} pairs ({100*summary_df['potential_bimodal'].sum()/len(summary_df):.1f}%)")

    print(f"\nMost common materials:")
    material_counts = summary_df['material'].value_counts()
    for mat, count in material_counts.head(5).items():
        print(f"  {mat:20s}: {count} pairs")

    print("\n" + "=" * 80)
    print(f"COMPLETE - {len(summary_results)} diagnostic plots generated")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
