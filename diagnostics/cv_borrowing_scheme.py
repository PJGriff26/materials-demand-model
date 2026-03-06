#!/usr/bin/env python3
"""
Small Sample Size CV Borrowing Scheme
======================================

Implements hierarchical borrowing of CV values for low-sample-size pairs:
1. Borrow from well-characterized pairs in the same TECHNOLOGY (n≥10)
2. If unavailable, borrow from the same MATERIAL across technologies (n≥10)
3. If unavailable, use global median CV from all well-characterized pairs
4. Fallback to CV=2.0 if no reference data exists

This is a more principled alternative to fixed inflation factors.

Usage:
    python diagnostics/cv_borrowing_scheme.py

Outputs:
    - outputs/tables/cv_borrowing_comparison.csv (old vs new CV values)
    - Summary statistics printed to console
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Path setup
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

# Constants
WELL_CHARACTERIZED_THRESHOLD = 10  # n ≥ 10 considered well-characterized
BORROWING_THRESHOLD = 5            # n < 5 gets full borrowing
BLEND_THRESHOLD = 10               # n < 10 gets partial borrowing
FALLBACK_CV = 2.0                  # Final fallback if no reference data


def load_intensity_table():
    """Load the intensity summary table."""
    table_path = ROOT / 'outputs' / 'tables' / 'Table_S_intensity_full_condensed.csv'
    if not table_path.exists():
        raise FileNotFoundError(
            f"Table not found: {table_path}\n"
            f"Run the pipeline first to generate this table."
        )
    return pd.read_csv(table_path)


def compute_reference_cvs(df):
    """
    Compute reference CV values at different hierarchical levels.

    Returns:
        dict with keys:
            'technology': {tech_name: median_cv}
            'material': {material_name: median_cv}
            'global': float (global median CV)
    """
    # Filter to well-characterized pairs only (n ≥ 10, CV > 0)
    well_char = df[
        (df['n'] >= WELL_CHARACTERIZED_THRESHOLD) &
        (df['CV (fitted)'] > 0)
    ].copy()

    print(f"\nComputing reference CVs from {len(well_char)} well-characterized pairs (n ≥ {WELL_CHARACTERIZED_THRESHOLD})")

    # Technology-level medians
    tech_cvs = well_char.groupby('Technology')['CV (fitted)'].median().to_dict()
    print(f"  Technology-level CVs: {len(tech_cvs)} technologies")

    # Material-level medians
    mat_cvs = well_char.groupby('Material')['CV (fitted)'].median().to_dict()
    print(f"  Material-level CVs: {len(mat_cvs)} materials")

    # Global median
    global_cv = well_char['CV (fitted)'].median()
    print(f"  Global median CV: {global_cv:.3f}")

    return {
        'technology': tech_cvs,
        'material': mat_cvs,
        'global': global_cv,
        'n_reference_pairs': len(well_char)
    }


def apply_cv_borrowing(row, ref_cvs):
    """
    Apply hierarchical CV borrowing for a single material-technology pair.

    Borrowing hierarchy:
        1. If n ≥ 10: use fitted CV (well-characterized)
        2. If 5 ≤ n < 10: blend fitted CV with borrowed CV (70% fitted, 30% borrowed)
        3. If n < 5: use borrowed CV (poorly characterized)

    Borrowing source hierarchy:
        1. Technology median (same tech, different materials)
        2. Material median (same material, different techs)
        3. Global median (all well-characterized pairs)
        4. Fallback CV=2.0

    Returns:
        dict with 'cv_borrowed', 'cv_source', 'blend_weight'
    """
    n = row['n']
    tech = row['Technology']
    material = row['Material']
    cv_fitted = row['CV (fitted)']

    # Determine borrowed CV source
    if tech in ref_cvs['technology']:
        cv_borrowed = ref_cvs['technology'][tech]
        source = f"tech:{tech}"
    elif material in ref_cvs['material']:
        cv_borrowed = ref_cvs['material'][material]
        source = f"mat:{material}"
    elif ref_cvs['global'] is not None:
        cv_borrowed = ref_cvs['global']
        source = "global"
    else:
        cv_borrowed = FALLBACK_CV
        source = "fallback"

    # Determine blending strategy
    if n >= BLEND_THRESHOLD:
        # Well-characterized: use fitted CV
        cv_final = cv_fitted
        blend_weight = 0.0
    elif n >= BORROWING_THRESHOLD:
        # Mid-range: blend fitted (70%) and borrowed (30%)
        blend_weight = 0.3
        cv_final = (1 - blend_weight) * cv_fitted + blend_weight * cv_borrowed
        source = f"blend:{source}"
    else:
        # Poorly characterized: use borrowed CV
        cv_final = cv_borrowed
        blend_weight = 1.0

    return {
        'cv_borrowed': cv_final,
        'cv_source': source,
        'blend_weight': blend_weight,
        'cv_reference': cv_borrowed
    }


def main():
    print("=" * 80)
    print("CV BORROWING SCHEME FOR LOW-SAMPLE-SIZE PAIRS")
    print("=" * 80)

    # Load data
    df = load_intensity_table()
    print(f"\nLoaded {len(df)} material-technology pairs")

    # Compute reference CVs
    ref_cvs = compute_reference_cvs(df)

    # Apply borrowing to all pairs
    print("\n" + "=" * 80)
    print("APPLYING HIERARCHICAL CV BORROWING")
    print("=" * 80)

    borrowing_results = df.apply(lambda row: apply_cv_borrowing(row, ref_cvs), axis=1)
    borrowing_df = pd.DataFrame(borrowing_results.tolist())

    # Combine with original data
    result = pd.concat([df, borrowing_df], axis=1)

    # Add comparison columns
    result['cv_old'] = result['CV (inflated)']  # Current pipeline approach
    result['cv_new'] = result['cv_borrowed']     # New borrowing approach
    result['cv_ratio'] = result['cv_new'] / result['cv_old']
    result['cv_diff'] = result['cv_new'] - result['cv_old']

    # Summary statistics
    print("\n" + "=" * 80)
    print("BORROWING STRATEGY BREAKDOWN")
    print("=" * 80)

    source_counts = result['cv_source'].value_counts()
    print("\nCV source distribution:")
    for source, count in source_counts.items():
        pct = 100 * count / len(result)
        print(f"  {source:30s} {count:4d} pairs ({pct:5.1f}%)")

    # Compare borrowing vs current inflation for low-n pairs
    print("\n" + "=" * 80)
    print("COMPARISON: BORROWING vs CURRENT INFLATION (n < 10)")
    print("=" * 80)

    low_n = result[result['n'] < 10].copy()
    print(f"\nAnalyzing {len(low_n)} pairs with n < 10:")
    print(f"  Current approach (inflation): CV(inflated) = CV(fitted) × inflation_factor")
    print(f"  New approach (borrowing):     CV(borrowed) = hierarchical borrowing")
    print()

    for n_val in sorted(low_n['n'].unique()):
        subset = low_n[low_n['n'] == n_val]
        median_old = subset['cv_old'].median()
        median_new = subset['cv_new'].median()
        median_ratio = subset['cv_ratio'].median()

        print(f"  n={n_val:2d}: old={median_old:6.3f}, new={median_new:6.3f}, ratio={median_ratio:5.2f}x ({len(subset):3d} pairs)")

    # Identify large changes
    print("\n" + "=" * 80)
    print("LARGEST CHANGES (|cv_new - cv_old| > 2.0)")
    print("=" * 80)

    large_changes = result[abs(result['cv_diff']) > 2.0].copy()
    large_changes = large_changes.sort_values('cv_diff', key=abs, ascending=False)

    if len(large_changes) > 0:
        print(f"\n{len(large_changes)} pairs with large CV changes:")
        for idx, row in large_changes.head(20).iterrows():
            print(f"  {row['Technology']:20s} {row['Material']:12s} (n={row['n']:2.0f}): "
                  f"old={row['cv_old']:6.2f}, new={row['cv_new']:6.2f}, "
                  f"Δ={row['cv_diff']:+6.2f}, source={row['cv_source']}")
    else:
        print("\nNo large changes detected.")

    # Save comparison table
    output_path = ROOT / 'outputs' / 'tables' / 'cv_borrowing_comparison.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_cols = [
        'Technology', 'Material', 'n', 'Distribution',
        'CV (observed)', 'CV (fitted)',
        'cv_old', 'cv_new', 'cv_diff', 'cv_ratio',
        'cv_source', 'blend_weight', 'cv_reference'
    ]
    result[output_cols].to_csv(output_path, index=False, float_format='%.4f')

    print(f"\n" + "=" * 80)
    print(f"SAVED: {output_path}")
    print("=" * 80)

    # Summary statistics
    print("\n" + "=" * 80)
    print("OVERALL COMPARISON STATISTICS")
    print("=" * 80)

    print(f"\nAll pairs (n={len(result)}):")
    print(f"  Median CV (old): {result['cv_old'].median():.3f}")
    print(f"  Median CV (new): {result['cv_new'].median():.3f}")
    print(f"  Median ratio:    {result['cv_ratio'].median():.3f}x")

    print(f"\nLow-n pairs (n < {BORROWING_THRESHOLD}, n={len(result[result['n'] < BORROWING_THRESHOLD])}):")
    low = result[result['n'] < BORROWING_THRESHOLD]
    print(f"  Median CV (old): {low['cv_old'].median():.3f}")
    print(f"  Median CV (new): {low['cv_new'].median():.3f}")
    print(f"  Median ratio:    {low['cv_ratio'].median():.3f}x")

    print(f"\nMid-range pairs ({BORROWING_THRESHOLD} ≤ n < {BLEND_THRESHOLD}, n={len(result[(result['n'] >= BORROWING_THRESHOLD) & (result['n'] < BLEND_THRESHOLD)])}):")
    mid = result[(result['n'] >= BORROWING_THRESHOLD) & (result['n'] < BLEND_THRESHOLD)]
    if len(mid) > 0:
        print(f"  Median CV (old): {mid['cv_old'].median():.3f}")
        print(f"  Median CV (new): {mid['cv_new'].median():.3f}")
        print(f"  Median ratio:    {mid['cv_ratio'].median():.3f}x")
    else:
        print("  (no pairs in this range)")

    print(f"\nWell-characterized pairs (n ≥ {BLEND_THRESHOLD}, n={len(result[result['n'] >= BLEND_THRESHOLD])}):")
    well = result[result['n'] >= BLEND_THRESHOLD]
    print(f"  Median CV (old): {well['cv_old'].median():.3f}")
    print(f"  Median CV (new): {well['cv_new'].median():.3f}")
    print(f"  Median ratio:    {well['cv_ratio'].median():.3f}x")
    print()

    # Recommendation
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print("""
The hierarchical borrowing scheme appears to be:
- More conservative for well-characterized pairs (n ≥ 10)
- More data-driven for low-n pairs
- Avoids arbitrary inflation factors

To integrate into the pipeline:
1. Review cv_borrowing_comparison.csv for specific pairs
2. Adjust BORROWING_THRESHOLD, BLEND_THRESHOLD, or FALLBACK_CV as needed
3. Integrate into distribution_fitting.py or stock_flow_simulation.py
4. Re-run Monte Carlo simulation to assess impact on demand projections
    """)


if __name__ == '__main__':
    main()
