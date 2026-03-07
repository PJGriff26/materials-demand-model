#!/usr/bin/env python3
"""
Test CV Borrowing Implementation
=================================

Quick test to verify that CV borrowing is working correctly:
1. Load intensity data
2. Fit distributions
3. Apply CV borrowing
4. Check results

Run from project root:
    python diagnostics/test_cv_borrowing.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_ingestion import load_all_data
from src.distribution_fitting import DistributionFitter
from src.cv_borrowing import (
    apply_cv_borrowing,
    create_cv_borrowing_report,
    compute_lognormal_cv
)


def main():
    print("=" * 80)
    print("TESTING CV BORROWING IMPLEMENTATION")
    print("=" * 80)
    print()

    # Load data
    print("Step 1: Loading intensity data...")
    data_path = ROOT / 'data' / 'intensity_data.csv'
    if not data_path.exists():
        print(f"ERROR: Cannot find {data_path}")
        sys.exit(1)

    data = load_all_data(
        intensity_path=data_path,
        national_capacity_path=ROOT / 'data' / 'StdScen24_annual_national.csv'
    )
    print(f"✓ Loaded {len(data['intensity'])} intensity observations")

    # Fit distributions
    print("\nStep 2: Fitting distributions...")
    fitter = DistributionFitter()
    fitted_dists = fitter.fit_all(data['intensity'])
    print(f"✓ Fitted {len(fitted_dists)} material-technology pairs")

    # Compute statistics BEFORE CV borrowing
    print("\n" + "=" * 80)
    print("STATISTICS BEFORE CV BORROWING")
    print("=" * 80)

    n_lognormal = 0
    n_low_n = 0
    cvs_before = []

    for (tech, mat), dist_info in fitted_dists.items():
        if dist_info.best_fit is not None:
            if dist_info.best_fit.distribution_name == 'lognormal':
                n_lognormal += 1
                sigma = dist_info.best_fit.parameters.get('s', 0)
                cv = compute_lognormal_cv(sigma)
                cvs_before.append((tech, mat, dist_info.n_samples, cv))

                if dist_info.n_samples < 5:
                    n_low_n += 1

    print(f"Total lognormal distributions: {n_lognormal}")
    print(f"Lognormal with n < 5: {n_low_n} ({100*n_low_n/n_lognormal:.1f}%)")

    # Show CV statistics by sample size bin
    cvs_df = pd.DataFrame(cvs_before, columns=['Technology', 'Material', 'n', 'CV'])

    print("\nCV statistics by sample size bin (BEFORE borrowing):")
    for bin_name, condition in [
        ('n=1', cvs_df['n'] == 1),
        ('n=2', cvs_df['n'] == 2),
        ('n=3', cvs_df['n'] == 3),
        ('n=4', cvs_df['n'] == 4),
        ('n>5', cvs_df['n'] > 5),
    ]:
        subset = cvs_df[condition]
        if len(subset) > 0:
            print(f"  {bin_name:5s}: {len(subset):3d} pairs, "
                  f"median CV = {subset['CV'].median():.3f}, "
                  f"range = [{subset['CV'].min():.3f}, {subset['CV'].max():.3f}]")

    # Compute reference CV (median from n > 5)
    reference_cvs = cvs_df[cvs_df['n'] > 5]['CV']
    median_cv = reference_cvs.median()
    print(f"\nReference CV (median from n > 5): {median_cv:.3f}")

    # Apply CV borrowing
    print("\n" + "=" * 80)
    print("APPLYING CV BORROWING")
    print("=" * 80)

    fitted_dists = apply_cv_borrowing(
        fitted_distributions=fitted_dists,
        threshold_n=5,
        reference_n=5
    )

    # Compute statistics AFTER CV borrowing
    print("\n" + "=" * 80)
    print("STATISTICS AFTER CV BORROWING")
    print("=" * 80)

    cvs_after = []

    for (tech, mat), dist_info in fitted_dists.items():
        if dist_info.best_fit is not None:
            if dist_info.best_fit.distribution_name == 'lognormal':
                sigma = dist_info.best_fit.parameters.get('s', 0)
                cv = compute_lognormal_cv(sigma)
                cvs_after.append((tech, mat, dist_info.n_samples, cv))

    cvs_after_df = pd.DataFrame(cvs_after, columns=['Technology', 'Material', 'n', 'CV'])

    print("\nCV statistics by sample size bin (AFTER borrowing):")
    for bin_name, condition in [
        ('n=1', cvs_after_df['n'] == 1),
        ('n=2', cvs_after_df['n'] == 2),
        ('n=3', cvs_after_df['n'] == 3),
        ('n=4', cvs_after_df['n'] == 4),
        ('n>5', cvs_after_df['n'] > 5),
    ]:
        subset = cvs_after_df[condition]
        if len(subset) > 0:
            print(f"  {bin_name:5s}: {len(subset):3d} pairs, "
                  f"median CV = {subset['CV'].median():.3f}, "
                  f"range = [{subset['CV'].min():.3f}, {subset['CV'].max():.3f}]")

    # Check that borrowing worked
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    low_n_after = cvs_after_df[cvs_after_df['n'] < 5]
    borrowed_cvs = low_n_after['CV'].unique()

    print(f"Number of pairs with n < 5: {len(low_n_after)}")
    print(f"Unique CV values for n < 5 pairs: {len(borrowed_cvs)}")

    if len(borrowed_cvs) == 1:
        borrowed_cv = borrowed_cvs[0]
        if abs(borrowed_cv - median_cv) < 0.001:
            print(f"✓ CV borrowing working correctly!")
            print(f"  All n < 5 pairs now have CV = {borrowed_cv:.3f}")
            print(f"  Reference median CV = {median_cv:.3f}")
        else:
            print(f"✗ ERROR: Borrowed CV ({borrowed_cv:.3f}) != Reference CV ({median_cv:.3f})")
            sys.exit(1)
    else:
        print(f"✗ ERROR: Expected all n < 5 pairs to have same CV")
        print(f"  Found {len(borrowed_cvs)} unique values: {borrowed_cvs}")
        sys.exit(1)

    # Show example pairs
    print("\n" + "=" * 80)
    print("EXAMPLE PAIRS (n < 5)")
    print("=" * 80)

    low_n_pairs = cvs_after_df[cvs_after_df['n'] < 5].head(10)
    print("\nFirst 10 pairs with n < 5:")
    print(low_n_pairs.to_string(index=False))

    # Create detailed report
    print("\n" + "=" * 80)
    print("CREATING DETAILED REPORT")
    print("=" * 80)

    output_path = ROOT / 'outputs' / 'data' / 'cv_borrowing_test_report.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    create_cv_borrowing_report(
        fitted_distributions=fitted_dists,
        output_path=str(output_path),
        threshold_n=5,
        reference_n=5
    )

    print(f"\n✓ Detailed report saved: {output_path}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE — CV BORROWING WORKING CORRECTLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
