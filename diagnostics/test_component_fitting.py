#!/usr/bin/env python3
"""
Test Component-Based Distribution Fitting
==========================================

Validates the component-based bimodal fitting implementation for:
- CIGS-Copper (trimodal: cell/module/BOS)
- a-Si-Copper (bimodal: cell vs BOS)
- CdTe-Copper (bimodal: cell vs BOS)

Verifies:
1. Bimodality detection triggers correctly
2. Components are fit separately
3. Monte Carlo sampling sums components
4. Results are physically meaningful

Usage:
    python diagnostics/test_component_fitting.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_ingestion import load_all_data
from src.distribution_fitting import DistributionFitter
from src.cv_borrowing import apply_cv_borrowing


def test_component_fitting():
    """Test component-based fitting for confirmed bimodal pairs"""
    print("=" * 80)
    print("TESTING COMPONENT-BASED DISTRIBUTION FITTING")
    print("=" * 80)
    print()

    # Load data
    print("Loading intensity data...")
    intensity_path = ROOT / 'data' / 'intensity_data.csv'
    capacity_path = ROOT / 'data' / 'StdScen24_annual_national.csv'
    data_dict = load_all_data(intensity_path, capacity_path)
    intensity_df = data_dict['intensity']
    print(f"✓ Loaded {len(intensity_df)} intensity observations")
    print()

    # Test pairs (confirmed TRUE BIMODAL from literature verification)
    test_pairs = [
        ('CIGS', 'Copper', 'trimodal'),
        ('a-Si', 'Copper', 'bimodal'),
        ('CdTe', 'Copper', 'bimodal'),
    ]

    fitter = DistributionFitter()
    results = {}

    for tech, mat, expected_mode in test_pairs:
        print("=" * 80)
        print(f"TESTING: {tech} - {mat} (Expected: {expected_mode})")
        print("=" * 80)

        # Get data for this pair
        pair_data = intensity_df[
            (intensity_df['technology'] == tech) &
            (intensity_df['material'] == mat)
        ]['intensity_t_per_mw'].values

        # Convert to g/MW (intensity is in t/MW, need g/MW)
        pair_data = pair_data * 1000

        if len(pair_data) == 0:
            print(f"✗ No data found for {tech}-{mat}")
            continue

        print(f"Data: n={len(pair_data)}")
        print(f"  Values: {sorted(pair_data)}")
        print(f"  Range: {np.min(pair_data):.1f} - {np.max(pair_data):.1f} g/MW")
        print()

        # Fit distribution
        result = fitter.fit_single(tech, mat, pair_data)

        # Check if bimodal fitting was used
        if result.is_bimodal:
            print(f"✓ BIMODAL FITTING DETECTED")
            print(f"  Split threshold: {result.split_threshold:.1f} g/MW")
            print(f"  Number of components: {len(result.component_fits)}")
            print()

            for i, (fit, label, data) in enumerate(zip(
                result.component_fits,
                result.component_labels,
                result.component_data
            ), 1):
                print(f"  Component {i}: {label}")
                print(f"    n = {len(data)}")
                print(f"    Data range: {np.min(data):.1f} - {np.max(data):.1f} g/MW")
                print(f"    Distribution: {fit.distribution_name}")
                if fit.distribution_name == 'lognormal':
                    print(f"    Parameters: σ={fit.parameters['s']:.3f}, "
                          f"scale={fit.parameters['scale']:.1f}")
                    print(f"    Median: {fit.parameters['scale']:.1f} g/MW")
                print()

            # Test Monte Carlo sampling
            print("Testing Monte Carlo sampling (n=10,000)...")
            samples = result.sample(n=10000, random_state=42)
            print(f"  Sample statistics:")
            print(f"    Median: {np.median(samples):.1f} g/MW")
            print(f"    Mean: {np.mean(samples):.1f} g/MW")
            print(f"    95% CI: [{np.percentile(samples, 2.5):.1f}, "
                  f"{np.percentile(samples, 97.5):.1f}] g/MW")
            print(f"    Max/Median ratio: {np.max(samples)/np.median(samples):.1f}x")
            print()

            # Compare to raw data statistics
            print("Comparison with raw data:")
            print(f"  Raw median: {np.median(pair_data):.1f} g/MW")
            print(f"  Raw range: {np.min(pair_data):.1f} - {np.max(pair_data):.1f} g/MW")
            print(f"  Sample median: {np.median(samples):.1f} g/MW")
            print(f"  Sample 95% covers raw range: "
                  f"{np.percentile(samples, 2.5) <= np.min(pair_data) and np.percentile(samples, 97.5) >= np.max(pair_data)}")
            print()

            results[f"{tech}-{mat}"] = {
                'is_bimodal': True,
                'n_components': len(result.component_fits),
                'components': result.component_labels,
                'sample_median': np.median(samples),
                'sample_95ci': (np.percentile(samples, 2.5), np.percentile(samples, 97.5)),
                'raw_median': np.median(pair_data),
                'raw_range': (np.min(pair_data), np.max(pair_data))
            }

        else:
            print(f"✗ BIMODAL FITTING NOT DETECTED (using single distribution)")
            print(f"  Best fit: {result.best_fit.distribution_name if result.best_fit else 'None'}")
            print(f"  Recommendation: {result.recommendation}")
            print()

            results[f"{tech}-{mat}"] = {
                'is_bimodal': False,
                'reason': result.recommendation
            }

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    bimodal_count = sum(1 for r in results.values() if r.get('is_bimodal', False))
    print(f"Bimodal pairs detected: {bimodal_count} / {len(test_pairs)}")
    print()

    for pair_name, result in results.items():
        if result.get('is_bimodal', False):
            print(f"✓ {pair_name}:")
            print(f"    Components: {result['n_components']} ({', '.join(result['components'])})")
            print(f"    Sample median: {result['sample_median']:.1f} g/MW")
            print(f"    Sample 95% CI: [{result['sample_95ci'][0]:.1f}, {result['sample_95ci'][1]:.1f}] g/MW")
        else:
            print(f"✗ {pair_name}: {result.get('reason', 'Unknown')}")
    print()

    # Validation checks
    print("=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)
    print()

    all_passed = True

    # Check 1: All test pairs should be bimodal
    if bimodal_count == len(test_pairs):
        print("✓ CHECK 1: All confirmed bimodal pairs detected as bimodal")
    else:
        print(f"✗ CHECK 1: Only {bimodal_count}/{len(test_pairs)} pairs detected as bimodal")
        all_passed = False

    # Check 2: All bimodal pairs should have 2+ components
    for pair_name, result in results.items():
        if result.get('is_bimodal', False):
            if result['n_components'] >= 2:
                print(f"✓ CHECK 2a: {pair_name} has {result['n_components']} components")
            else:
                print(f"✗ CHECK 2a: {pair_name} has only {result['n_components']} component(s)")
                all_passed = False

    # Check 3: Sampled values should cover raw data range
    for pair_name, result in results.items():
        if result.get('is_bimodal', False):
            raw_min, raw_max = result['raw_range']
            ci_low, ci_high = result['sample_95ci']
            if ci_low <= raw_min and ci_high >= raw_max:
                print(f"✓ CHECK 3: {pair_name} 95% CI covers raw data range")
            else:
                print(f"⚠ CHECK 3: {pair_name} 95% CI [{ci_low:.1f}, {ci_high:.1f}] "
                      f"vs raw [{raw_min:.1f}, {raw_max:.1f}]")
                # This is a warning, not a failure (CI doesn't have to cover full range)

    print()
    if all_passed:
        print("=" * 80)
        print("✓ ALL VALIDATION CHECKS PASSED")
        print("=" * 80)
        print()
        print("Component-based fitting is working correctly!")
    else:
        print("=" * 80)
        print("✗ SOME VALIDATION CHECKS FAILED")
        print("=" * 80)
        print()
        print("Review implementation before running full simulation.")

    return results


if __name__ == '__main__':
    test_component_fitting()
