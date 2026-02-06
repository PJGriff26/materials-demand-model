#!/usr/bin/env python3
"""
Unit Validation Script
======================

This script validates that the unit conversion from t/GW to t/MW is working correctly
and that output magnitudes are reasonable.

Expected ranges (order of magnitude checks):
- Copper demand for US grid (2035): ~1-10 million tonnes/year
- Steel demand for US grid (2035): ~10-100 million tonnes/year
- Cement demand for US grid (2035): ~100-1000 million tonnes/year

Author: Materials Demand Research Team
Date: 2024
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Import our modules
from data_ingestion import MaterialIntensityLoader

# Configuration
DATA_DIR = Path('Python/11.30.25/src')
INTENSITY_FILE = DATA_DIR / 'intensity_data.csv'


def validate_unit_conversion():
    """Validate that intensity values are converted correctly"""

    print("="*80)
    print("UNIT CONVERSION VALIDATION")
    print("="*80)
    print()

    # Load intensity data
    loader = MaterialIntensityLoader()
    data = loader.load(INTENSITY_FILE)

    print("\nSample of converted intensity values (t/MW):")
    print("-" * 80)

    # Show a few examples
    sample = data.sample(min(10, len(data)))
    for _, row in sample.iterrows():
        print(f"{row['technology']:20s} - {row['material']:15s}: {row['intensity_t_per_mw']:10.4f} t/MW")

    # Statistical checks
    print("\n" + "-" * 80)
    print("Statistical Summary of Intensity Values (t/MW):")
    print("-" * 80)

    stats = data['intensity_t_per_mw'].describe()
    print(f"  Count:  {stats['count']:.0f}")
    print(f"  Mean:   {stats['mean']:.4f} t/MW")
    print(f"  Std:    {stats['std']:.4f} t/MW")
    print(f"  Min:    {stats['min']:.4f} t/MW")
    print(f"  25%:    {stats['25%']:.4f} t/MW")
    print(f"  Median: {stats['50%']:.4f} t/MW")
    print(f"  75%:    {stats['75%']:.4f} t/MW")
    print(f"  Max:    {stats['max']:.4f} t/MW")

    # Reasonableness checks
    print("\n" + "-" * 80)
    print("Reasonableness Checks:")
    print("-" * 80)

    # Check for specific materials
    copper = data[data['material'] == 'Copper']['intensity_t_per_mw']
    if len(copper) > 0:
        print(f"\nCopper intensity range: {copper.min():.2f} - {copper.max():.2f} t/MW")
        print(f"  Expected: ~2-10 t/MW for solar/wind")
        if copper.min() > 1 and copper.max() < 100:
            print(f"  ✓ PASS: Values are in reasonable range")
        else:
            print(f"  ✗ FAIL: Values outside expected range")

    steel = data[data['material'] == 'Steel']['intensity_t_per_mw']
    if len(steel) > 0:
        print(f"\nSteel intensity range: {steel.min():.2f} - {steel.max():.2f} t/MW")
        print(f"  Expected: ~50-200 t/MW for wind")
        if steel.min() > 10 and steel.max() < 1000:
            print(f"  ✓ PASS: Values are in reasonable range")
        else:
            print(f"  ✗ FAIL: Values outside expected range")

    cement = data[data['material'] == 'Cement']['intensity_t_per_mw']
    if len(cement) > 0:
        print(f"\nCement intensity range: {cement.min():.2f} - {cement.max():.2f} t/MW")
        print(f"  Expected: ~5-50 t/MW for various technologies")
        if cement.min() > 0.1 and cement.max() < 500:
            print(f"  ✓ PASS: Values are in reasonable range")
        else:
            print(f"  ✗ FAIL: Values outside expected range")

    # Check that no values are unreasonably large (would indicate unit error)
    max_intensity = data['intensity_t_per_mw'].max()
    print(f"\n" + "-" * 80)
    print(f"Maximum intensity value: {max_intensity:.2f} t/MW")
    if max_intensity < 10000:  # Should never exceed 10,000 t/MW
        print(f"  ✓ PASS: Maximum value is reasonable")
    else:
        print(f"  ✗ FAIL: Maximum value suggests unit conversion error!")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print()


def estimate_demand_magnitude():
    """
    Quick back-of-envelope calculation to estimate expected demand magnitudes.

    Example: If US adds 100 GW of solar in a year with 5 t/MW copper intensity:
    Demand = 100,000 MW × 5 t/MW = 500,000 tonnes copper
    """

    print("\n" + "="*80)
    print("EXPECTED DEMAND MAGNITUDE ESTIMATES")
    print("="*80)
    print()

    print("Back-of-envelope calculations:")
    print("-" * 80)

    print("\nExample 1: Solar PV buildout")
    print("  Assumption: 100 GW/year solar additions (aggressive scenario)")
    print("  Copper intensity: ~5 t/MW")
    print("  Expected copper demand: 100,000 MW × 5 t/MW = 500,000 tonnes/year")
    print("  Order of magnitude: ~10^5 to 10^6 tonnes/year")

    print("\nExample 2: Wind buildout")
    print("  Assumption: 50 GW/year wind additions")
    print("  Steel intensity: ~100 t/MW")
    print("  Expected steel demand: 50,000 MW × 100 t/MW = 5,000,000 tonnes/year")
    print("  Order of magnitude: ~10^6 to 10^7 tonnes/year")

    print("\nExample 3: Total cement (all technologies)")
    print("  Assumption: 200 GW/year total additions, ~20 t/MW average")
    print("  Expected cement demand: 200,000 MW × 20 t/MW = 4,000,000 tonnes/year")
    print("  Order of magnitude: ~10^6 to 10^7 tonnes/year")

    print("\n" + "-" * 80)
    print("If your simulation results show:")
    print("  - Copper: 10^5 to 10^7 tonnes/year → REASONABLE ✓")
    print("  - Copper: 10^10+ tonnes/year → UNIT ERROR ✗")
    print()


if __name__ == '__main__':

    # Verify files exist
    if not INTENSITY_FILE.exists():
        print(f"ERROR: Cannot find {INTENSITY_FILE}")
        print(f"Please run this script from the Materials Demand directory")
        sys.exit(1)

    # Run validation
    validate_unit_conversion()

    # Show expected magnitudes
    estimate_demand_magnitude()

    print("="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Re-run your Monte Carlo simulation: python demo_stock_flow_simulation.py")
    print("2. Check that output values are now in reasonable ranges (10^6 - 10^8 tonnes)")
    print("3. If values are still too high, there may be additional unit issues to investigate")
    print()
