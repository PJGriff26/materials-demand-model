"""
Sensitivity Analysis for Materials Demand
==========================================

Analyzes which factors most influence material demand estimates:
1. Technology contribution - which technologies drive demand for each material
2. Scenario sensitivity - how demand varies across scenarios
3. Uncertainty decomposition - intensity vs capacity uncertainty
4. Temporal sensitivity - how demand changes over time
5. Cross-material correlations - which materials move together

Usage:
    python sensitivity_analysis.py
    python sensitivity_analysis.py --material Copper
    python sensitivity_analysis.py --output_dir ../outputs/sensitivity
"""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_dir)

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from data_ingestion import load_all_data
from distribution_fitting import DistributionFitter


def load_simulation_data(output_dir='../outputs'):
    """Load Monte Carlo simulation outputs."""
    summary_df = pd.read_csv(f'{output_dir}/material_demand_summary.csv')
    scenario_df = pd.read_csv(f'{output_dir}/material_demand_by_scenario.csv')
    return summary_df, scenario_df


def technology_contribution_analysis(intensity_df, capacity_df, fitted_dists, output_dir):
    """
    Analyze which technologies contribute most to each material's demand.

    This helps identify:
    - Which technologies drive demand for critical materials
    - Diversification of material sources across technologies
    """
    print("\n" + "="*80)
    print("1. TECHNOLOGY CONTRIBUTION ANALYSIS")
    print("="*80)

    # Capacity data is in wide format - melt it to long format
    # Columns are like 'wind_onshore_MW', 'upv_MW', etc.
    capacity_cols = [c for c in capacity_df.columns if c.endswith('_MW')]
    capacity_long = capacity_df.melt(
        id_vars=['scenario', 'year'],
        value_vars=capacity_cols,
        var_name='tech_col',
        value_name='capacity_mw'
    )
    # Extract technology name (remove '_MW' suffix)
    capacity_long['technology'] = capacity_long['tech_col'].str.replace('_MW', '')

    # Create mapping from intensity technology names to capacity technology names
    tech_mapping = {
        'onshore wind': 'wind_onshore',
        'offshore wind': 'wind_offshore',
        'utility-scale solar pv': 'upv',
        'Solar Distributed': 'distpv',
        'Hydro': 'hydro',
        'Nuclear New': 'nuclear',
        'Nuclear SMR': 'nuclear_smr',
        'Gas': 'gas_cc',
        'Gas CCS': 'gas_cc_ccs',
        'NGCC': 'gas_cc',
        'NGGT': 'gas_ct',
        'Coal': 'coal',
        'Coal CCS': 'coal_ccs',
        'Biomass': 'bio',
        'Bio CCS': 'bio-ccs',
        'Geothermal': 'geo',
        'CSP': 'csp',
        'ASIGE': 'upv',  # Assume silicon PV
        'CDTE': 'upv',
        'CdTe': 'upv',
        'CIGS': 'upv',
        'a-Si': 'upv',
    }

    # Get unique materials
    materials = intensity_df['material'].unique()

    # Calculate expected demand contribution by technology for each material
    results = []

    for material in materials:
        mat_intensities = intensity_df[intensity_df['material'] == material]

        for _, row in mat_intensities.iterrows():
            tech = row['technology']
            intensity = row['intensity_t_per_mw']

            # Map to capacity technology name
            cap_tech = tech_mapping.get(tech, tech.lower().replace(' ', '_'))

            # Get capacity for this technology (sum across scenarios and years)
            tech_capacity = capacity_long[capacity_long['technology'] == cap_tech]
            if len(tech_capacity) == 0:
                continue

            total_capacity_mw = tech_capacity['capacity_mw'].sum()

            # Expected demand = intensity * capacity
            expected_demand = intensity * total_capacity_mw

            results.append({
                'material': material,
                'technology': tech,
                'intensity_t_per_mw': intensity,
                'total_capacity_mw': total_capacity_mw,
                'expected_demand_tonnes': expected_demand
            })

    contrib_df = pd.DataFrame(results)

    # Aggregate by material and technology
    tech_contrib = contrib_df.groupby(['material', 'technology']).agg({
        'intensity_t_per_mw': 'mean',
        'total_capacity_mw': 'first',
        'expected_demand_tonnes': 'mean'
    }).reset_index()

    # Calculate percentage contribution for each material
    material_totals = tech_contrib.groupby('material')['expected_demand_tonnes'].sum()
    tech_contrib['pct_contribution'] = tech_contrib.apply(
        lambda row: 100 * row['expected_demand_tonnes'] / material_totals[row['material']]
        if material_totals[row['material']] > 0 else 0, axis=1
    )

    # Get top contributors for each material
    print("\nTop technology contributors by material:")
    print("-" * 80)

    top_contributors = {}
    for material in materials:
        mat_data = tech_contrib[tech_contrib['material'] == material].sort_values(
            'pct_contribution', ascending=False
        ).head(3)

        if len(mat_data) > 0:
            top_contributors[material] = mat_data
            top_tech = mat_data.iloc[0]
            print(f"{material:20s}: {top_tech['technology']:25s} ({top_tech['pct_contribution']:.1f}%)")

    # Save detailed results
    tech_contrib.to_csv(f'{output_dir}/technology_contributions.csv', index=False)
    print(f"\nSaved: {output_dir}/technology_contributions.csv")

    return tech_contrib, top_contributors


def scenario_sensitivity_analysis(scenario_df, output_dir):
    """
    Analyze how material demand varies across scenarios.

    This helps identify:
    - Which materials are most sensitive to scenario choice
    - Which scenarios produce highest/lowest demand
    - Scenario clusters with similar material demand profiles
    """
    print("\n" + "="*80)
    print("2. SCENARIO SENSITIVITY ANALYSIS")
    print("="*80)

    # Aggregate demand by scenario and material
    scenario_totals = scenario_df.groupby(['scenario', 'material'])['mean'].sum().unstack(fill_value=0)

    # Calculate coefficient of variation across scenarios for each material
    scenario_cv = scenario_totals.std() / scenario_totals.mean() * 100
    scenario_cv = scenario_cv.sort_values(ascending=False)

    print("\nMaterials most sensitive to scenario choice (by CV across scenarios):")
    print("-" * 80)
    for mat, cv in scenario_cv.head(10).items():
        print(f"{mat:20s}: CV = {cv:.1f}%")

    # Identify extreme scenarios
    print("\nScenarios with highest total material demand:")
    print("-" * 80)
    scenario_sums = scenario_totals.sum(axis=1).sort_values(ascending=False)
    for scenario, total in scenario_sums.head(5).items():
        print(f"{scenario:40s}: {total/1e9:.2f} billion tonnes")

    print("\nScenarios with lowest total material demand:")
    print("-" * 80)
    for scenario, total in scenario_sums.tail(5).items():
        print(f"{scenario:40s}: {total/1e9:.2f} billion tonnes")

    # Calculate scenario range for each material
    scenario_range = pd.DataFrame({
        'min_demand': scenario_totals.min(),
        'max_demand': scenario_totals.max(),
        'mean_demand': scenario_totals.mean(),
        'range': scenario_totals.max() - scenario_totals.min(),
        'range_pct': (scenario_totals.max() - scenario_totals.min()) / scenario_totals.mean() * 100
    }).sort_values('range_pct', ascending=False)

    scenario_range.to_csv(f'{output_dir}/scenario_sensitivity.csv')
    print(f"\nSaved: {output_dir}/scenario_sensitivity.csv")

    return scenario_totals, scenario_cv, scenario_range


def uncertainty_decomposition(scenario_df, fitted_dists, output_dir):
    """
    Decompose total uncertainty into:
    - Intensity uncertainty (from distribution fitting)
    - Scenario uncertainty (from different capacity projections)

    Uses variance decomposition: Var(total) = Var(intensity) + Var(scenario) + 2*Cov
    """
    print("\n" + "="*80)
    print("3. UNCERTAINTY DECOMPOSITION")
    print("="*80)

    results = []

    for material in scenario_df['material'].unique():
        mat_data = scenario_df[scenario_df['material'] == material]

        # Total uncertainty (CV of mean demand across all scenario-years)
        total_mean = mat_data['mean'].mean()
        total_std = mat_data['mean'].std()
        total_cv = total_std / total_mean * 100 if total_mean > 0 else 0

        # Within-scenario uncertainty (average CV within each scenario)
        within_cv = mat_data['cv'].mean() if 'cv' in mat_data.columns else 0

        # Between-scenario uncertainty (CV of scenario means)
        scenario_means = mat_data.groupby('scenario')['mean'].sum()
        between_cv = scenario_means.std() / scenario_means.mean() * 100 if scenario_means.mean() > 0 else 0

        results.append({
            'material': material,
            'total_cv': total_cv,
            'intensity_cv': within_cv,  # Approximation of intensity uncertainty
            'scenario_cv': between_cv,   # Scenario uncertainty
            'dominant_source': 'intensity' if within_cv > between_cv else 'scenario'
        })

    decomp_df = pd.DataFrame(results).sort_values('total_cv', ascending=False)

    print("\nUncertainty decomposition by material:")
    print("-" * 80)
    print(f"{'Material':<20s} {'Total CV':<12s} {'Intensity CV':<14s} {'Scenario CV':<14s} {'Dominant':<12s}")
    print("-" * 80)

    for _, row in decomp_df.head(15).iterrows():
        print(f"{row['material']:<20s} {row['total_cv']:>8.1f}%    {row['intensity_cv']:>10.1f}%    "
              f"{row['scenario_cv']:>10.1f}%    {row['dominant_source']:<12s}")

    # Summary statistics
    intensity_dominant = (decomp_df['dominant_source'] == 'intensity').sum()
    scenario_dominant = (decomp_df['dominant_source'] == 'scenario').sum()

    print(f"\nSummary:")
    print(f"  Materials where intensity uncertainty dominates: {intensity_dominant}")
    print(f"  Materials where scenario uncertainty dominates: {scenario_dominant}")

    decomp_df.to_csv(f'{output_dir}/uncertainty_decomposition.csv', index=False)
    print(f"\nSaved: {output_dir}/uncertainty_decomposition.csv")

    return decomp_df


def temporal_sensitivity_analysis(summary_df, output_dir):
    """
    Analyze how demand and uncertainty evolve over time.
    """
    print("\n" + "="*80)
    print("4. TEMPORAL SENSITIVITY ANALYSIS")
    print("="*80)

    # Calculate year-over-year growth rates
    growth_rates = []

    for material in summary_df['material'].unique():
        mat_data = summary_df[summary_df['material'] == material].sort_values('year').copy()

        if len(mat_data) > 1:
            # Find first non-zero year
            nonzero_data = mat_data[mat_data['mean'] > 0]
            if len(nonzero_data) < 2:
                continue

            first_year = nonzero_data.iloc[0]
            last_year = nonzero_data.iloc[-1]
            n_years = last_year['year'] - first_year['year']

            if first_year['mean'] > 0 and n_years > 0:
                cagr = (last_year['mean'] / first_year['mean']) ** (1/n_years) - 1

                # Uncertainty trend (change in CV over time)
                mat_data.loc[:, 'cv'] = mat_data['std'] / mat_data['mean'] * 100

                # Handle infinite/nan CVs
                valid_cv = mat_data['cv'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid_cv) >= 2:
                    cv_trend = np.polyfit(mat_data.loc[valid_cv.index, 'year'], valid_cv, 1)[0]
                else:
                    cv_trend = 0

                growth_rates.append({
                    'material': material,
                    'cagr': cagr * 100,
                    'demand_2026': first_year['mean'],
                    'demand_2050': last_year['mean'],
                    'growth_factor': last_year['mean'] / first_year['mean'],
                    'cv_2026': mat_data.iloc[0]['cv'] if np.isfinite(mat_data.iloc[0]['cv']) else 0,
                    'cv_2050': mat_data.iloc[-1]['cv'] if np.isfinite(mat_data.iloc[-1]['cv']) else 0,
                    'cv_trend': cv_trend  # Positive = uncertainty increasing
                })

    if len(growth_rates) == 0:
        print("WARNING: No growth rates calculated")
        growth_df = pd.DataFrame(columns=['material', 'cagr', 'demand_2026', 'demand_2050',
                                          'growth_factor', 'cv_2026', 'cv_2050', 'cv_trend'])
    else:
        growth_df = pd.DataFrame(growth_rates).sort_values('cagr', ascending=False)

    print("\nFastest growing materials (CAGR 2026-2050):")
    print("-" * 80)
    for _, row in growth_df.head(10).iterrows():
        print(f"{row['material']:<20s}: CAGR = {row['cagr']:>6.1f}%, "
              f"Growth factor = {row['growth_factor']:.1f}x")

    print("\nMaterials with increasing uncertainty over time:")
    print("-" * 80)
    increasing_uncert = growth_df[growth_df['cv_trend'] > 0.5].sort_values('cv_trend', ascending=False)
    for _, row in increasing_uncert.head(5).iterrows():
        print(f"{row['material']:<20s}: CV trend = +{row['cv_trend']:.2f}%/year")

    growth_df.to_csv(f'{output_dir}/temporal_sensitivity.csv', index=False)
    print(f"\nSaved: {output_dir}/temporal_sensitivity.csv")

    return growth_df


def cross_material_correlation(scenario_df, output_dir):
    """
    Analyze correlations between material demands across scenarios.

    This helps identify:
    - Materials that tend to move together
    - Potential supply chain bottlenecks (correlated critical materials)
    """
    print("\n" + "="*80)
    print("5. CROSS-MATERIAL CORRELATION ANALYSIS")
    print("="*80)

    # Create pivot table: scenarios x materials
    pivot = scenario_df.groupby(['scenario', 'material'])['mean'].sum().unstack(fill_value=0)

    # Calculate correlation matrix
    corr_matrix = pivot.corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    materials = corr_matrix.columns.tolist()

    for i, mat1 in enumerate(materials):
        for mat2 in materials[i+1:]:
            corr = corr_matrix.loc[mat1, mat2]
            if abs(corr) > 0.8:
                high_corr_pairs.append({
                    'material_1': mat1,
                    'material_2': mat2,
                    'correlation': corr
                })

    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)

    print("\nHighly correlated material pairs (|r| > 0.8):")
    print("-" * 80)
    for _, row in high_corr_df.head(15).iterrows():
        print(f"{row['material_1']:<15s} <-> {row['material_2']:<15s}: r = {row['correlation']:.3f}")

    # Save correlation matrix
    corr_matrix.to_csv(f'{output_dir}/material_correlations.csv')
    high_corr_df.to_csv(f'{output_dir}/high_correlation_pairs.csv', index=False)

    print(f"\nSaved: {output_dir}/material_correlations.csv")
    print(f"Saved: {output_dir}/high_correlation_pairs.csv")

    return corr_matrix, high_corr_df


def create_sensitivity_visualization(tech_contrib, scenario_range, decomp_df, growth_df,
                                    corr_matrix, output_dir):
    """Create comprehensive sensitivity analysis visualization."""

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ========================================================================
    # Panel 1: Top technology contributors (stacked bar for top 10 materials)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :2])

    # Get top 10 materials by total demand
    material_totals = tech_contrib.groupby('material')['expected_demand_tonnes'].sum()
    top10_materials = material_totals.nlargest(10).index.tolist()

    # Prepare data for stacked bar
    top_tech_data = tech_contrib[tech_contrib['material'].isin(top10_materials)]

    # Get top 5 technologies overall
    tech_totals = tech_contrib.groupby('technology')['expected_demand_tonnes'].sum()
    top5_techs = tech_totals.nlargest(5).index.tolist()

    # Create pivot for stacking
    pivot_data = top_tech_data.pivot_table(
        index='material', columns='technology', values='expected_demand_tonnes', fill_value=0
    )

    # Sum "other" technologies
    other_techs = [t for t in pivot_data.columns if t not in top5_techs]
    if other_techs:
        pivot_data['Other'] = pivot_data[other_techs].sum(axis=1)
        pivot_data = pivot_data.drop(columns=other_techs)

    # Reorder by total
    pivot_data = pivot_data.loc[top10_materials[::-1]]

    # Plot stacked bar
    pivot_data.plot(kind='barh', stacked=True, ax=ax1, colormap='tab10')
    ax1.set_xlabel('Expected Demand (Tonnes)', fontsize=11)
    ax1.set_title('Technology Contribution to Material Demand (Top 10 Materials)',
                 fontsize=13, fontweight='bold')
    ax1.legend(title='Technology', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax1.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))

    # ========================================================================
    # Panel 2: Scenario sensitivity (range as % of mean)
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 2])

    top_sensitive = scenario_range.head(12)
    y_pos = np.arange(len(top_sensitive))

    colors = ['red' if x > 100 else 'orange' if x > 50 else 'green'
              for x in top_sensitive['range_pct']]
    ax2.barh(y_pos, top_sensitive['range_pct'], color=colors, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_sensitive.index)
    ax2.set_xlabel('Scenario Range (% of Mean)', fontsize=11)
    ax2.set_title('Scenario Sensitivity by Material', fontsize=13, fontweight='bold')
    ax2.axvline(100, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(50, color='orange', linestyle='--', alpha=0.5)

    # ========================================================================
    # Panel 3: Uncertainty decomposition
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    top_uncert = decomp_df.head(12)
    y_pos = np.arange(len(top_uncert))
    width = 0.35

    ax3.barh(y_pos - width/2, top_uncert['intensity_cv'], width,
            label='Intensity Uncertainty', color='steelblue', alpha=0.7)
    ax3.barh(y_pos + width/2, top_uncert['scenario_cv'], width,
            label='Scenario Uncertainty', color='coral', alpha=0.7)

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(top_uncert['material'])
    ax3.set_xlabel('CV (%)', fontsize=11)
    ax3.set_title('Uncertainty Decomposition', fontsize=13, fontweight='bold')
    ax3.legend(loc='lower right')

    # ========================================================================
    # Panel 4: Growth rates (CAGR)
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    top_growth = growth_df.head(12)
    y_pos = np.arange(len(top_growth))

    colors = ['darkgreen' if x > 5 else 'green' if x > 2 else 'gray' if x > 0 else 'red'
              for x in top_growth['cagr']]
    ax4.barh(y_pos, top_growth['cagr'], color=colors, alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(top_growth['material'])
    ax4.set_xlabel('CAGR 2026-2050 (%)', fontsize=11)
    ax4.set_title('Demand Growth Rate by Material', fontsize=13, fontweight='bold')
    ax4.axvline(0, color='black', linewidth=0.5)

    # ========================================================================
    # Panel 5: Growth factor vs uncertainty
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 2])

    ax5.scatter(growth_df['growth_factor'], growth_df['cv_2050'],
               s=80, alpha=0.6, c='steelblue', edgecolor='black')

    # Label outliers
    for _, row in growth_df.iterrows():
        if row['growth_factor'] > 3 or row['cv_2050'] > 60:
            ax5.annotate(row['material'], (row['growth_factor'], row['cv_2050']),
                        fontsize=8, alpha=0.8)

    ax5.set_xlabel('Growth Factor (2050/2026)', fontsize=11)
    ax5.set_ylabel('CV in 2050 (%)', fontsize=11)
    ax5.set_title('Growth vs Uncertainty', fontsize=13, fontweight='bold')
    ax5.grid(alpha=0.3)

    # ========================================================================
    # Panel 6: Correlation heatmap (subset)
    # ========================================================================
    ax6 = fig.add_subplot(gs[2, :])

    # Select top 15 materials by demand variance across scenarios
    material_variance = corr_matrix.index.tolist()[:15]  # First 15 alphabetically, or customize
    subset_corr = corr_matrix.loc[material_variance, material_variance]

    im = ax6.imshow(subset_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax6.set_xticks(range(len(material_variance)))
    ax6.set_yticks(range(len(material_variance)))
    ax6.set_xticklabels(material_variance, rotation=45, ha='right', fontsize=9)
    ax6.set_yticklabels(material_variance, fontsize=9)
    ax6.set_title('Cross-Material Demand Correlation (Across Scenarios)',
                 fontsize=13, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax6, shrink=0.6)
    cbar.set_label('Correlation', fontsize=10)

    # Overall title
    fig.suptitle('Sensitivity Analysis: Material Demand Drivers and Uncertainty',
                fontsize=16, fontweight='bold', y=0.98)

    # Save
    output_path = f'{output_dir}/sensitivity_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Sensitivity analysis for material demand')
    parser.add_argument('--material', type=str, default=None,
                       help='Focus on specific material (optional)')
    parser.add_argument('--output_dir', type=str, default='../outputs/sensitivity',
                       help='Output directory for results')

    args = parser.parse_args()

    print("="*80)
    print("MATERIALS DEMAND SENSITIVITY ANALYSIS")
    print("="*80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("\nLoading data...")
    data = load_all_data('../data/intensity_data.csv', '../data/StdScen24_annual_national.csv')
    intensity_df = data['intensity']
    capacity_df = data['capacity_national']

    # Fit distributions
    print("Fitting distributions...")
    fitter = DistributionFitter()
    fitted_dists = fitter.fit_all(intensity_df)

    # Load simulation outputs
    print("Loading simulation outputs...")
    summary_df, scenario_df = load_simulation_data()

    # Run analyses
    tech_contrib, top_contributors = technology_contribution_analysis(
        intensity_df, capacity_df, fitted_dists, args.output_dir
    )

    scenario_totals, scenario_cv, scenario_range = scenario_sensitivity_analysis(
        scenario_df, args.output_dir
    )

    decomp_df = uncertainty_decomposition(scenario_df, fitted_dists, args.output_dir)

    growth_df = temporal_sensitivity_analysis(summary_df, args.output_dir)

    corr_matrix, high_corr_df = cross_material_correlation(scenario_df, args.output_dir)

    # Create visualization
    print("\nCreating visualization...")
    create_sensitivity_visualization(
        tech_contrib, scenario_range, decomp_df, growth_df, corr_matrix, args.output_dir
    )

    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
