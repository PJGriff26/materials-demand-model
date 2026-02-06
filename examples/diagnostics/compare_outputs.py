"""
Compare Old vs New Monte Carlo Outputs
=======================================

This script compares the old materials_demand.csv output (without Monte Carlo)
to the new material_demand_by_scenario.csv (with Monte Carlo) to check for
order of magnitude agreement.

Usage:
    python compare_outputs.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
OLD_OUTPUT = Path("../Old Reference Outputs/Outputs/materials_demand.csv")
NEW_OUTPUT = Path("../outputs/material_demand_by_scenario.csv")

print("=" * 80)
print("COMPARING OLD VS NEW MONTE CARLO OUTPUTS")
print("=" * 80)

# Load old output (wide format with _mean columns)
print("\nLoading old output (non-Monte Carlo)...")
old_df = pd.read_csv(OLD_OUTPUT)
print(f"  Loaded: {len(old_df)} rows")
print(f"  Columns: {len(old_df.columns)}")
print(f"  Years: {sorted(old_df['year'].unique())}")
print(f"  Scenarios: {sorted(old_df['scenario'].unique())}")

# Load new output (tidy format with mean column)
print("\nLoading new output (Monte Carlo)...")
new_df = pd.read_csv(NEW_OUTPUT)
print(f"  Loaded: {len(new_df)} rows")
print(f"  Years: {sorted(new_df['year'].unique())}")
print(f"  Scenarios: {len(new_df['scenario'].unique())} (showing first 5: {sorted(new_df['scenario'].unique())[:5]})")

# Reshape old data from wide to tidy
print("\nReshaping old data to tidy format...")
old_tidy = []

# Get all material columns (those ending with _mean)
material_cols = [col for col in old_df.columns if col.endswith('_mean')]
materials = [col.replace('_mean', '') for col in material_cols]

print(f"  Found {len(materials)} materials in old output")

for _, row in old_df.iterrows():
    for material in materials:
        col_name = f"{material}_mean"
        if col_name in old_df.columns:
            old_tidy.append({
                'year': row['year'],
                'scenario': row['scenario'],
                'technology': row.get('technology', 'ALL'),
                'material': material,
                'mean': row[col_name]
            })

old_tidy_df = pd.DataFrame(old_tidy)

# Aggregate old data by year, scenario, material (sum across technologies)
old_agg = old_tidy_df.groupby(['year', 'scenario', 'material'])['mean'].sum().reset_index()
old_agg = old_agg.rename(columns={'mean': 'old_mean'})

print(f"  Aggregated to {len(old_agg)} rows")

# Map scenario names (old uses "IRA" and "Ref", new has many more)
# We'll focus on comparable scenarios
scenario_mapping = {
    'IRA': ['Mid_Case_100by2035', 'Mid_Case', 'Adv_RE'],  # Try these as proxies
    'Ref': ['Mid_Case_No_IRA', 'High_Demand_Growth', 'Low_Demand_Growth']
}

print("\n" + "=" * 80)
print("COMPARISON BY YEAR, SCENARIO, AND MATERIAL")
print("=" * 80)

# Compare common years
common_years = [2025, 2030, 2035]

for year in common_years:
    print(f"\n{'='*80}")
    print(f"YEAR: {year}")
    print(f"{'='*80}")

    for old_scenario, new_scenarios in scenario_mapping.items():
        print(f"\n{old_scenario} (old) vs {new_scenarios} (new proxy):\n")
        print(f"{'Material':<20} {'Old':<20} {'New (avg)':<20} {'Ratio':<15} {'Agreement'}")
        print("-" * 90)

        # Get old data
        old_year = old_agg[(old_agg['year'] == year) & (old_agg['scenario'] == old_scenario)]

        # Get new data (average across proxy scenarios)
        new_year = new_df[
            (new_df['year'] == year) &
            (new_df['scenario'].isin(new_scenarios))
        ]

        if new_year.empty:
            print(f"  No data in new output for year {year}, scenarios {new_scenarios}")
            continue

        new_year_agg = new_year.groupby('material')['mean'].mean().reset_index()
        new_year_agg = new_year_agg.rename(columns={'mean': 'new_mean'})

        # Merge
        comparison = pd.merge(
            old_year,
            new_year_agg,
            on='material',
            how='outer'
        )

        # Calculate ratio and order of magnitude
        comparison['ratio'] = comparison['new_mean'] / comparison['old_mean']
        comparison['log_ratio'] = np.log10(comparison['ratio'].replace([np.inf, -np.inf], np.nan))

        # Categorize agreement
        def categorize_agreement(log_ratio):
            if pd.isna(log_ratio):
                return "N/A"
            abs_log = abs(log_ratio)
            if abs_log < 0.5:  # Within ~3x
                return "✓ GOOD"
            elif abs_log < 1.0:  # Within ~10x
                return "~ OK"
            elif abs_log < 2.0:  # Within ~100x
                return "⚠ POOR"
            else:  # > 100x difference
                return "✗ BAD"

        comparison['agreement'] = comparison['log_ratio'].apply(categorize_agreement)

        # Sort by material
        comparison = comparison.sort_values('material')

        # Print results
        for _, row in comparison.iterrows():
            old_val = row['old_mean']
            new_val = row['new_mean']
            ratio = row['ratio']
            agreement = row['agreement']

            # Format values
            if pd.isna(old_val) or old_val == 0:
                old_str = "0 or N/A"
            else:
                old_str = f"{old_val:,.2f}" if old_val < 1000 else f"{old_val:.2e}"

            if pd.isna(new_val) or new_val == 0:
                new_str = "0 or N/A"
            else:
                new_str = f"{new_val:,.2f}" if new_val < 1000 else f"{new_val:.2e}"

            if pd.isna(ratio) or np.isinf(ratio):
                ratio_str = "N/A"
            else:
                ratio_str = f"{ratio:,.1f}x"

            print(f"{row['material']:<20} {old_str:<20} {new_str:<20} {ratio_str:<15} {agreement}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

all_comparisons = []
for year in common_years:
    for old_scenario, new_scenarios in scenario_mapping.items():
        old_year = old_agg[(old_agg['year'] == year) & (old_agg['scenario'] == old_scenario)]
        new_year = new_df[
            (new_df['year'] == year) &
            (new_df['scenario'].isin(new_scenarios))
        ]

        if new_year.empty:
            continue

        new_year_agg = new_year.groupby('material')['mean'].mean().reset_index()
        new_year_agg = new_year_agg.rename(columns={'mean': 'new_mean'})

        comparison = pd.merge(old_year, new_year_agg, on='material', how='inner')
        comparison = comparison[(comparison['old_mean'] > 0) & (comparison['new_mean'] > 0)]
        comparison['ratio'] = comparison['new_mean'] / comparison['old_mean']
        comparison['log_ratio'] = np.log10(comparison['ratio'])

        all_comparisons.append(comparison)

if all_comparisons:
    all_comp_df = pd.concat(all_comparisons, ignore_index=True)

    print(f"\nTotal comparisons: {len(all_comp_df)}")
    print(f"Median ratio (new/old): {all_comp_df['ratio'].median():.2f}x")
    print(f"Mean log10(ratio): {all_comp_df['log_ratio'].mean():.2f}")

    # Count agreement categories
    def categorize(log_r):
        abs_log = abs(log_r)
        if abs_log < 0.5:
            return "GOOD (<3x)"
        elif abs_log < 1.0:
            return "OK (<10x)"
        elif abs_log < 2.0:
            return "POOR (<100x)"
        else:
            return "BAD (>100x)"

    all_comp_df['category'] = all_comp_df['log_ratio'].apply(categorize)

    print("\nAgreement breakdown:")
    for cat, count in all_comp_df['category'].value_counts().sort_index().items():
        pct = 100 * count / len(all_comp_df)
        print(f"  {cat}: {count} ({pct:.1f}%)")

    # Identify materials with large discrepancies
    print("\nMaterials with largest discrepancies (>100x):")
    large_disc = all_comp_df[abs(all_comp_df['log_ratio']) > 2].sort_values('log_ratio', ascending=False)
    if len(large_disc) > 0:
        print(f"{'Material':<20} {'Typical Ratio':<20} {'Issue'}")
        print("-" * 60)
        for material in large_disc['material'].unique():
            mat_data = large_disc[large_disc['material'] == material]
            avg_ratio = mat_data['ratio'].median()
            if avg_ratio > 100:
                issue = "NEW >> OLD"
            elif avg_ratio < 0.01:
                issue = "OLD >> NEW"
            else:
                issue = "Variable"
            print(f"{material:<20} {avg_ratio:>19.2e}x  {issue}")
    else:
        print("  None! All materials within 100x agreement.")

print("\n" + "=" * 80)
print("NOTES:")
print("=" * 80)
print("""
- Old output: non-Monte Carlo, aggregated by technology
- New output: Monte Carlo with 10,000 iterations
- Some extreme values in new output may indicate unit errors still present
- Scenario mapping is approximate (IRA ≈ Mid_Case, Ref ≈ Mid_Case_No_IRA)
- Year 2025 in old output vs 2026 in new output (different base years)
""")

print("\n" + "=" * 80)
print("END OF COMPARISON")
print("=" * 80)
