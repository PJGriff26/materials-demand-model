"""
Diagnostic script to trace unit conversion through the pipeline
"""
import sys
sys.path.insert(0, '../src')

import pandas as pd
import numpy as np
from data_ingestion import load_all_data

print("=" * 80)
print("UNIT CONVERSION DIAGNOSTIC")
print("=" * 80)

# Load data using the same method as the simulation
print("\n1. Loading data with load_all_data()...")
intensity_df, capacity_proj_df, capacity_loader = load_all_data(
    '../data/intensity_data.csv',
    '../data/StdScen24_annual_national.csv'
)

print(f"\n2. Checking intensity_df columns:")
print(f"   Columns: {list(intensity_df.columns)}")

# Check if unit conversion column exists
if 'intensity_t_per_mw' in intensity_df.columns:
    print("   ✓ Found 'intensity_t_per_mw' column (converted)")
else:
    print("   ✗ 'intensity_t_per_mw' column NOT FOUND")

# Check raw values
print(f"\n3. Sample intensity values:")
sample_materials = ['Silicon', 'Copper', 'Aluminum', 'Steel', 'Cement']
for material in sample_materials:
    mat_data = intensity_df[intensity_df['material'] == material]
    if not mat_data.empty:
        if 'intensity_raw' in mat_data.columns:
            raw_val = mat_data['intensity_raw'].iloc[0]
            print(f"   {material:<15} - Raw: {raw_val}")
        if 'intensity_t_per_mw' in mat_data.columns:
            conv_val = mat_data['intensity_t_per_mw'].iloc[0]
            print(f"   {material:<15} - Converted: {conv_val} (should be raw/1000)")

# Check what column is actually used
print(f"\n4. Which column is used for simulation?")
value_cols = [col for col in intensity_df.columns if 'value' in col.lower() or 'intensity' in col.lower()]
print(f"   Available value columns: {value_cols}")

# Try to simulate what happens in distribution fitting
print(f"\n5. Checking what distribution_fitting.py expects...")
print(f"   Looking for value_col parameter usage...")

# Show first few rows
print(f"\n6. First 5 rows of intensity_df:")
print(intensity_df.head())

# Check for the problematic column name
print(f"\n7. Checking column naming:")
if 'intensity_mt_per_mw' in intensity_df.columns:
    print("   ⚠️  Found 'intensity_mt_per_mw' (incorrect)")
if 'intensity_t_per_mw' in intensity_df.columns:
    print("   ✓ Found 'intensity_t_per_mw' (correct)")
if 'value' in intensity_df.columns:
    print("   Found 'value' column")
    print(f"   Sample values: {intensity_df['value'].head()}")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
