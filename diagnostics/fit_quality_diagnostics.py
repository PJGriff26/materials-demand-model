#!/usr/bin/env python3
"""
Fit Quality Diagnostics for Material Intensity Distributions

Generates comprehensive diagnostics tables showing:
- Goodness-of-fit statistics by sample size
- Parameter stability metrics
- Observed vs fitted comparisons
- Recommendations for sample size thresholds

This helps inform decisions about:
- When to trust fitted distributions vs borrow/inflate
- Appropriate sample size cutoffs for different methods
- Quality control for distribution fitting
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import lognorm, kstest, anderson

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

# ============================================================================
# FIT QUALITY METRICS
# ============================================================================

def fit_lognormal_with_diagnostics(values):
    """
    Fit lognormal distribution and compute comprehensive diagnostics.

    Returns dict with:
    - mu, sigma: Lognormal parameters
    - cv_fitted: Coefficient of variation from fitted distribution
    - ks_stat, ks_pval: Kolmogorov-Smirnov test statistics
    - ad_stat, ad_crit: Anderson-Darling test statistic and critical values
    - rmse: Root mean squared error between empirical and fitted CDF
    - fit_quality: Categorical assessment (Excellent/Good/Fair/Poor)
    """
    if len(values) < 2:
        return None

    try:
        # Fit lognormal
        shape, loc, scale = lognorm.fit(values, floc=0)
        mu = np.log(scale)
        sigma = shape
        cv_fitted = np.sqrt(np.exp(sigma**2) - 1)

        # Kolmogorov-Smirnov test
        ks_result = kstest(values, lambda x: lognorm.cdf(x, shape, loc, scale))
        ks_stat = ks_result.statistic
        ks_pval = ks_result.pvalue

        # Anderson-Darling test (for lognormal)
        # Transform to normal and test
        log_values = np.log(values)
        ad_result = anderson(log_values, dist='norm')
        ad_stat = ad_result.statistic
        ad_crit = ad_result.critical_values[2]  # 5% significance level

        # RMSE between empirical and fitted CDF
        sorted_vals = np.sort(values)
        empirical_cdf = np.arange(1, len(values) + 1) / len(values)
        fitted_cdf = lognorm.cdf(sorted_vals, shape, loc, scale)
        rmse = np.sqrt(np.mean((empirical_cdf - fitted_cdf)**2))

        # Categorical fit quality
        # Good fit: KS p>0.05, AD stat < critical value, RMSE < 0.15
        if ks_pval > 0.10 and ad_stat < ad_crit and rmse < 0.10:
            fit_quality = "Excellent"
        elif ks_pval > 0.05 and ad_stat < ad_crit * 1.5 and rmse < 0.15:
            fit_quality = "Good"
        elif ks_pval > 0.01 and rmse < 0.25:
            fit_quality = "Fair"
        else:
            fit_quality = "Poor"

        return {
            'mu': mu,
            'sigma': sigma,
            'cv_fitted': cv_fitted,
            'ks_stat': ks_stat,
            'ks_pval': ks_pval,
            'ad_stat': ad_stat,
            'ad_crit': ad_crit,
            'rmse': rmse,
            'fit_quality': fit_quality
        }
    except:
        return None


def compute_observed_statistics(values):
    """Compute observed (empirical) statistics from raw data."""
    if len(values) < 2:
        return {}

    mean_obs = np.mean(values)
    std_obs = np.std(values, ddof=1)
    cv_obs = std_obs / mean_obs if mean_obs > 0 else 0

    return {
        'mean_obs': mean_obs,
        'std_obs': std_obs,
        'cv_obs': cv_obs,
        'median_obs': np.median(values),
        'q25_obs': np.percentile(values, 25),
        'q75_obs': np.percentile(values, 75),
        'iqr_obs': np.percentile(values, 75) - np.percentile(values, 25)
    }


# ============================================================================
# DIAGNOSTIC TABLE GENERATION
# ============================================================================

def generate_fit_diagnostics_by_sample_size(df):
    """
    Generate comprehensive fit diagnostics aggregated by sample size bins.

    Returns DataFrame with:
    - Sample size statistics
    - Fit quality distributions
    - Parameter stability metrics
    - Observed vs fitted comparisons
    """

    # Compute fit diagnostics for each pair
    diagnostics = []

    for (tech, material), group in df.groupby(['Technology', 'Material']):
        values = group['g_per_MW'].values
        n = len(values)

        # Get fit diagnostics
        fit_diag = fit_lognormal_with_diagnostics(values)
        obs_stats = compute_observed_statistics(values)

        if fit_diag is not None:
            diagnostics.append({
                'Technology': tech,
                'Material': material,
                'n': n,
                **fit_diag,
                **obs_stats
            })

    diag_df = pd.DataFrame(diagnostics)

    # Define sample size bins
    def bin_sample_size(n):
        if n == 1:
            return '1'
        elif n == 2:
            return '2'
        elif n == 3:
            return '3'
        elif 4 <= n <= 5:
            return '4-5'
        elif 6 <= n <= 10:
            return '6-10'
        elif 11 <= n <= 20:
            return '11-20'
        elif 21 <= n <= 50:
            return '21-50'
        else:
            return '>50'

    diag_df['n_bin'] = diag_df['n'].apply(bin_sample_size)

    # Aggregate by sample size bin
    bin_order = ['1', '2', '3', '4-5', '6-10', '11-20', '21-50', '>50']

    summary_rows = []

    for bin_name in bin_order:
        bin_data = diag_df[diag_df['n_bin'] == bin_name]

        if len(bin_data) == 0:
            continue

        # Count fit quality categories
        quality_counts = bin_data['fit_quality'].value_counts()
        pct_excellent = 100 * quality_counts.get('Excellent', 0) / len(bin_data)
        pct_good = 100 * quality_counts.get('Good', 0) / len(bin_data)
        pct_fair = 100 * quality_counts.get('Fair', 0) / len(bin_data)
        pct_poor = 100 * quality_counts.get('Poor', 0) / len(bin_data)

        # KS test pass rate (p > 0.05)
        ks_pass_rate = 100 * (bin_data['ks_pval'] > 0.05).sum() / len(bin_data)

        # AD test pass rate (stat < critical)
        ad_pass_rate = 100 * (bin_data['ad_stat'] < bin_data['ad_crit']).sum() / len(bin_data)

        # Parameter stability (CV of sigma)
        sigma_cv = bin_data['sigma'].std() / bin_data['sigma'].mean() if bin_data['sigma'].mean() > 0 else np.nan

        # Observed vs fitted CV comparison
        cv_bias = (bin_data['cv_fitted'] - bin_data['cv_obs']).median()
        cv_ratio = (bin_data['cv_fitted'] / bin_data['cv_obs'].replace(0, np.nan)).median()

        summary_rows.append({
            'Sample Size': bin_name,
            'Pairs': len(bin_data),
            'Mean n': bin_data['n'].mean(),
            'KS Pass %': ks_pass_rate,
            'AD Pass %': ad_pass_rate,
            'Excellent %': pct_excellent,
            'Good %': pct_good,
            'Fair %': pct_fair,
            'Poor %': pct_poor,
            'Median RMSE': bin_data['rmse'].median(),
            'σ Stability (CV)': sigma_cv,
            'CV Bias (fit-obs)': cv_bias,
            'CV Ratio (fit/obs)': cv_ratio,
            'Median σ': bin_data['sigma'].median(),
            'Median CV (obs)': bin_data['cv_obs'].median(),
            'Median CV (fit)': bin_data['cv_fitted'].median()
        })

    summary_df = pd.DataFrame(summary_rows)

    return summary_df, diag_df


def generate_material_specific_diagnostics(diag_df):
    """
    Generate diagnostics for specific high-importance materials.

    Shows how fit quality varies across sample sizes for key materials.
    """
    # Key structural materials + critical materials
    key_materials = ['Copper', 'Steel', 'Aluminum', 'Cement', 'Nickel',
                     'Neodymium', 'Indium', 'Silicon', 'Silver', 'Tellurium']

    material_diag = diag_df[diag_df['Material'].isin(key_materials)].copy()
    material_diag = material_diag.sort_values(['Material', 'n'])

    # Select key columns
    cols = ['Material', 'Technology', 'n', 'fit_quality', 'ks_pval',
            'rmse', 'cv_obs', 'cv_fitted', 'sigma']

    return material_diag[cols]


def generate_threshold_recommendations(summary_df):
    """
    Generate recommendations for sample size thresholds based on fit quality.
    """
    recommendations = []

    # Threshold 1: Minimum for reliable fitting (>50% good/excellent fits)
    good_plus_excellent = summary_df['Excellent %'] + summary_df['Good %']
    reliable_threshold = summary_df[good_plus_excellent > 50]

    if len(reliable_threshold) > 0:
        min_reliable = reliable_threshold.iloc[0]['Sample Size']
        recommendations.append({
            'Threshold': 'Reliable Fitting',
            'Sample Size': min_reliable,
            'Criterion': '>50% Good/Excellent fits',
            'Value': f"{reliable_threshold.iloc[0]['Excellent %'] + reliable_threshold.iloc[0]['Good %']:.1f}%",
            'Recommendation': 'Use fitted distributions without adjustment'
        })

    # Threshold 2: Minimum for KS test (>80% pass rate)
    ks_reliable = summary_df[summary_df['KS Pass %'] > 80]
    if len(ks_reliable) > 0:
        min_ks = ks_reliable.iloc[0]['Sample Size']
        recommendations.append({
            'Threshold': 'KS Test Passing',
            'Sample Size': min_ks,
            'Criterion': '>80% KS test pass (p>0.05)',
            'Value': f"{ks_reliable.iloc[0]['KS Pass %']:.1f}%",
            'Recommendation': 'Lognormal assumption well-supported'
        })

    # Threshold 3: Low RMSE (<0.15)
    low_rmse = summary_df[summary_df['Median RMSE'] < 0.15]
    if len(low_rmse) > 0:
        min_rmse = low_rmse.iloc[0]['Sample Size']
        recommendations.append({
            'Threshold': 'Low Fit Error',
            'Sample Size': min_rmse,
            'Criterion': 'Median RMSE < 0.15',
            'Value': f"{low_rmse.iloc[0]['Median RMSE']:.3f}",
            'Recommendation': 'Fitted CDF closely matches empirical CDF'
        })

    # Threshold 4: Where poor fits dominate (>50% poor)
    poor_dominated = summary_df[summary_df['Poor %'] > 50]
    if len(poor_dominated) > 0:
        max_poor = poor_dominated.iloc[-1]['Sample Size']
        recommendations.append({
            'Threshold': 'Poor Fit Zone',
            'Sample Size': f'≤{max_poor}',
            'Criterion': '>50% Poor fits',
            'Value': f"{poor_dominated.iloc[-1]['Poor %']:.1f}%",
            'Recommendation': 'Use borrowing or inflation methods'
        })

    return pd.DataFrame(recommendations)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("FIT QUALITY DIAGNOSTICS FOR MATERIAL INTENSITY DISTRIBUTIONS")
    print("=" * 80)
    print()

    # Load data
    print("[1/4] Loading intensity data...")
    df = load_intensity_data()
    print(f"      Loaded {len(df)} intensity values")
    print(f"      Covering {df.groupby(['Technology', 'Material']).ngroups} material-tech pairs")
    print()

    # Generate diagnostics by sample size
    print("[2/4] Computing fit diagnostics by sample size...")
    summary_df, diag_df = generate_fit_diagnostics_by_sample_size(df)
    print(f"      Generated diagnostics for {len(diag_df)} pairs")
    print()

    # Generate material-specific diagnostics
    print("[3/4] Computing material-specific diagnostics...")
    material_diag = generate_material_specific_diagnostics(diag_df)
    print(f"      Generated diagnostics for {material_diag['Material'].nunique()} key materials")
    print()

    # Generate threshold recommendations
    print("[4/4] Computing threshold recommendations...")
    recommendations = generate_threshold_recommendations(summary_df)
    print(f"      Generated {len(recommendations)} threshold recommendations")
    print()

    # ========================================================================
    # TABLE 1: FIT QUALITY BY SAMPLE SIZE
    # ========================================================================

    print("=" * 80)
    print("TABLE 1: FIT QUALITY DIAGNOSTICS BY SAMPLE SIZE")
    print("=" * 80)
    print()

    # Format for display
    display_df = summary_df.copy()
    display_df['Mean n'] = display_df['Mean n'].round(1)
    display_df['KS Pass %'] = display_df['KS Pass %'].round(1)
    display_df['AD Pass %'] = display_df['AD Pass %'].round(1)
    display_df['Excellent %'] = display_df['Excellent %'].round(1)
    display_df['Good %'] = display_df['Good %'].round(1)
    display_df['Fair %'] = display_df['Fair %'].round(1)
    display_df['Poor %'] = display_df['Poor %'].round(1)
    display_df['Median RMSE'] = display_df['Median RMSE'].round(3)
    display_df['σ Stability (CV)'] = display_df['σ Stability (CV)'].round(2)
    display_df['CV Bias (fit-obs)'] = display_df['CV Bias (fit-obs)'].round(3)
    display_df['CV Ratio (fit/obs)'] = display_df['CV Ratio (fit/obs)'].round(2)
    display_df['Median σ'] = display_df['Median σ'].round(3)
    display_df['Median CV (obs)'] = display_df['Median CV (obs)'].round(3)
    display_df['Median CV (fit)'] = display_df['Median CV (fit)'].round(3)

    print(display_df.to_string(index=False))
    print()

    # ========================================================================
    # TABLE 2: THRESHOLD RECOMMENDATIONS
    # ========================================================================

    print("=" * 80)
    print("TABLE 2: SAMPLE SIZE THRESHOLD RECOMMENDATIONS")
    print("=" * 80)
    print()
    print(recommendations.to_string(index=False))
    print()

    # ========================================================================
    # TABLE 3: MATERIAL-SPECIFIC DIAGNOSTICS (TOP 20 BY N)
    # ========================================================================

    print("=" * 80)
    print("TABLE 3: FIT QUALITY FOR KEY MATERIALS (Top 20 by sample size)")
    print("=" * 80)
    print()

    material_display = material_diag.nlargest(20, 'n').copy()
    material_display['ks_pval'] = material_display['ks_pval'].round(4)
    material_display['rmse'] = material_display['rmse'].round(3)
    material_display['cv_obs'] = material_display['cv_obs'].round(3)
    material_display['cv_fitted'] = material_display['cv_fitted'].round(3)
    material_display['sigma'] = material_display['sigma'].round(3)

    print(material_display.to_string(index=False))
    print()

    # ========================================================================
    # SAVE OUTPUTS
    # ========================================================================

    outdir = ROOT / 'outputs' / 'fitting_sample_size_diagnostics'
    outdir.mkdir(parents=True, exist_ok=True)

    # Save Table 1 (markdown)
    table1_md = outdir / 'table6_fit_diagnostics_by_sample_size.md'
    with open(table1_md, 'w') as f:
        f.write("# Table 6: Fit Quality Diagnostics by Sample Size\n\n")
        f.write(display_df.to_markdown(index=False))
        f.write("\n")
    print(f"✓ Saved Table 1: {table1_md}")

    # Save Table 2 (markdown)
    table2_md = outdir / 'table7_threshold_recommendations.md'
    with open(table2_md, 'w') as f:
        f.write("# Table 7: Sample Size Threshold Recommendations\n\n")
        f.write(recommendations.to_markdown(index=False))
        f.write("\n")
    print(f"✓ Saved Table 2: {table2_md}")

    # Save Table 3 (markdown)
    table3_md = outdir / 'table8_material_specific_diagnostics.md'
    with open(table3_md, 'w') as f:
        f.write("# Table 8: Fit Quality for Key Materials\n\n")
        f.write(material_display.to_markdown(index=False))
        f.write("\n")
    print(f"✓ Saved Table 3: {table3_md}")

    # Save full diagnostics CSV
    full_csv = outdir / 'fit_diagnostics_full.csv'
    diag_df.to_csv(full_csv, index=False)
    print(f"✓ Saved full diagnostics: {full_csv}")

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
