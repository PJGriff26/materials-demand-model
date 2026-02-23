"""
Trace through a single calculation to see where the unit error occurs
"""
import sys
sys.path.insert(0, '../src')

import pandas as pd
import numpy as np
from data_ingestion import load_all_data
from distribution_fitting import DistributionFitter

print("="*80)
print("TRACING CALCULATION FOR ONE MATERIAL")
print("="*80)

# Load data
print("\n1. Loading data...")
data = load_all_data(
    '../data/intensity_data.csv',
    '../data/StdScen24_annual_national.csv'
)

intensity_df = data['intensity']
capacity_df = data['capacity_national']

# Check what column exists
print(f"\n2. Intensity DataFrame columns: {list(intensity_df.columns)}")

# Get Aluminum data as an example
aluminum_data = intensity_df[
    (intensity_df['technology'] == 'ASIGE') &
    (intensity_df['material'] == 'Aluminum')
]

print(f"\n3. Sample Aluminum intensity data (ASIGE technology):")
print(aluminum_data[['technology', 'material', 'intensity_t_per_mw']].head())
print(f"\n   Note: Only converted values available (intensity_t_per_mw)")

# Fit distributions
print(f"\n4. Fitting distributions...")
fitter = DistributionFitter()
fitted_dists = fitter.fit_all(intensity_df)

# Get the distribution for ASIGE + Aluminum
if ('ASIGE', 'Aluminum') in fitted_dists:
    dist_info = fitted_dists[('ASIGE', 'Aluminum')]
    print(f"\n5. Distribution info for ASIGE + Aluminum:")
    print(f"   Raw data (first 5 values): {dist_info.raw_data[:5]}")
    print(f"   Mean: {np.mean(dist_info.raw_data)}")
    print(f"   Expected (if converted): ~31.3 t/MW (31300 t/GW / 1000)")

    # Sample once
    np.random.seed(42)
    sample = dist_info.sample(1)
    print(f"\n6. Sampled value: {sample}")

    # Check capacity for one scenario
    mid_case_2035 = capacity_df[
        (capacity_df['scenario'] == 'Mid_Case') &
        (capacity_df['year'] == 2035)
    ]

    # Get utility-scale solar capacity (maps to ASIGE)
    solar_capacity = mid_case_2035['Utility-scale PV capacity (MW)'].values[0]
    print(f"\n7. Sample capacity (Utility-scale PV, Mid_Case, 2035): {solar_capacity:,.0f} MW")

    # Calculate demand
    demand_tonnes = solar_capacity * sample
    print(f"\n8. Calculated demand:")
    print(f"   Formula: capacity (MW) × intensity (t/MW)")
    print(f"   = {solar_capacity:,.0f} MW × {sample:.6f} t/MW")
    print(f"   = {demand_tonnes:,.2f} tonnes")
    print(f"\n   Is this reasonable?")
    print(f"   Expected range: 10,000-100,000 tonnes for large solar buildout")
    if demand_tonnes > 1e10:
        print(f"   ❌ Value is {demand_tonnes/1e9:.0f} BILLION tonnes - WAY TOO HIGH!")
    elif demand_tonnes > 1e6:
        print(f"   ⚠️  Value is {demand_tonnes/1e6:.1f} million tonnes - seems high")
    elif demand_tonnes > 1000:
        print(f"   ✓ Value seems reasonable")

print("\n" + "="*80)
