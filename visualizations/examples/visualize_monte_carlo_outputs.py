"""
Monte Carlo Output Visualization
================================

Visualizes the results of the Monte Carlo simulation for material demand.
"""

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def create_monte_carlo_visualization(output_dir='../outputs'):
    """Create comprehensive visualization of Monte Carlo results."""

    # Load data
    summary_df = pd.read_csv(f'{output_dir}/material_demand_summary.csv')
    scenario_df = pd.read_csv(f'{output_dir}/material_demand_by_scenario.csv')

    # Calculate CV
    scenario_df['cv'] = (scenario_df['std'] / scenario_df['mean']) * 100
    summary_df['cv'] = (summary_df['std'] / summary_df['mean']) * 100

    # Create figure with multiple panels
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # ========================================================================
    # Panel 1: Total demand by material (aggregated across scenarios/years)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :2])

    # Aggregate by material
    material_totals = summary_df.groupby('material').agg({
        'mean': 'sum',
        'p50': 'sum',
        'p2': 'sum',
        'p97': 'sum'
    }).sort_values('mean', ascending=True)

    # Top 15 materials by demand
    top_materials = material_totals.tail(15)

    y_pos = np.arange(len(top_materials))
    ax1.barh(y_pos, top_materials['mean'] / 1e6, color='steelblue', alpha=0.7, label='Mean')
    ax1.errorbar(top_materials['mean'] / 1e6, y_pos,
                xerr=[(top_materials['mean'] - top_materials['p2']) / 1e6,
                      (top_materials['p97'] - top_materials['mean']) / 1e6],
                fmt='none', color='black', capsize=3, label='95% CI (2.5th-97.5th percentile)')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_materials.index)
    ax1.set_xlabel('Cumulative Demand 2026-2050 (Million Tonnes)', fontsize=11)
    ax1.set_title('Top 15 Materials by Total Demand (All Scenarios)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(axis='x', alpha=0.3)

    # ========================================================================
    # Panel 2: CV distribution
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 2])

    # Filter out inf/nan
    cv_valid = scenario_df['cv'].replace([np.inf, -np.inf], np.nan).dropna()

    ax2.hist(cv_valid.clip(upper=200), bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(cv_valid.median(), color='red', linestyle='--', linewidth=2,
               label=f'Median CV: {cv_valid.median():.1f}%')
    ax2.axvline(100, color='orange', linestyle=':', linewidth=2, label='CV = 100%')

    ax2.set_xlabel('Coefficient of Variation (%)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of CV Across All Estimates', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.set_xlim(0, 200)

    # ========================================================================
    # Panel 3: Demand over time for top materials
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, :])

    # Get top 5 materials
    top5_materials = material_totals.tail(5).index.tolist()

    colors = plt.cm.tab10(np.linspace(0, 1, len(top5_materials)))

    for i, mat in enumerate(top5_materials):
        mat_data = summary_df[summary_df['material'] == mat].sort_values('year')

        ax3.plot(mat_data['year'], mat_data['mean'] / 1e6,
                color=colors[i], linewidth=2, label=mat, marker='o')
        ax3.fill_between(mat_data['year'],
                        mat_data['p2'] / 1e6,
                        mat_data['p97'] / 1e6,
                        color=colors[i], alpha=0.2)

    ax3.set_xlabel('Year', fontsize=11)
    ax3.set_ylabel('Annual Demand (Million Tonnes)', fontsize=11)
    ax3.set_title('Top 5 Materials: Demand Trajectory with 95% CI (2.5th-97.5th percentile)',
                 fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', ncol=5)
    ax3.grid(alpha=0.3)

    # ========================================================================
    # Panel 4: Mean vs Median comparison (check for skewness)
    # ========================================================================
    ax4 = fig.add_subplot(gs[2, 0])

    # Scatter mean vs median
    ax4.scatter(scenario_df['p50'] / 1e6, scenario_df['mean'] / 1e6,
               alpha=0.3, s=10, c='steelblue')

    # Perfect agreement line
    max_val = max(scenario_df['mean'].max(), scenario_df['p50'].max()) / 1e6
    ax4.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect agreement')

    ax4.set_xlabel('Median (p50) [Million Tonnes]', fontsize=11)
    ax4.set_ylabel('Mean [Million Tonnes]', fontsize=11)
    ax4.set_title('Mean vs Median\n(Points above line = right-skewed)', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.set_xlim(0, max_val * 1.05)
    ax4.set_ylim(0, max_val * 1.05)
    ax4.grid(alpha=0.3)

    # ========================================================================
    # Panel 5: Uncertainty by material (CV)
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, 1])

    # Average CV by material
    cv_by_material = scenario_df.groupby('material')['cv'].median().sort_values(ascending=True)
    top_cv = cv_by_material.tail(15)

    y_pos = np.arange(len(top_cv))
    colors = ['red' if cv > 100 else 'orange' if cv > 50 else 'green' for cv in top_cv.values]
    ax5.barh(y_pos, top_cv.values, color=colors, alpha=0.7)

    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(top_cv.index)
    ax5.set_xlabel('Median CV (%)', fontsize=11)
    ax5.set_title('Materials with Highest Uncertainty', fontsize=13, fontweight='bold')
    ax5.axvline(100, color='red', linestyle='--', alpha=0.5)
    ax5.axvline(50, color='orange', linestyle='--', alpha=0.5)
    ax5.grid(axis='x', alpha=0.3)

    # ========================================================================
    # Panel 6: Scenario comparison for a key material
    # ========================================================================
    ax6 = fig.add_subplot(gs[2, 2])

    # Pick Copper as example
    copper_data = scenario_df[scenario_df['material'] == 'Copper']

    # Aggregate by scenario
    scenario_totals = copper_data.groupby('scenario')['mean'].sum().sort_values()

    # Show top and bottom 5 scenarios
    bottom5 = scenario_totals.head(5)
    top5 = scenario_totals.tail(5)
    selected = pd.concat([bottom5, top5])

    y_pos = np.arange(len(selected))
    colors = ['green'] * 5 + ['red'] * 5
    ax6.barh(y_pos, selected.values / 1e6, color=colors, alpha=0.7)

    ax6.set_yticks(y_pos)
    ax6.set_yticklabels([s[:20] + '...' if len(s) > 20 else s for s in selected.index], fontsize=8)
    ax6.set_xlabel('Total Copper Demand (Million Tonnes)', fontsize=11)
    ax6.set_title('Copper: Lowest vs Highest Demand Scenarios', fontsize=13, fontweight='bold')
    ax6.grid(axis='x', alpha=0.3)

    # Overall title
    fig.suptitle('Monte Carlo Simulation Results: Material Demand Analysis\n'
                '10,000 iterations across 61 scenarios, 2026-2050',
                fontsize=16, fontweight='bold', y=0.98)

    # Save
    output_path = f'{output_dir}/monte_carlo_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {output_path}')

    plt.close()

    # Print summary statistics
    print()
    print('='*80)
    print('MONTE CARLO OUTPUT SUMMARY')
    print('='*80)
    print(f'Total scenario-year-material combinations: {len(scenario_df):,}')
    print(f'Scenarios: {scenario_df["scenario"].nunique()}')
    print(f'Years: {sorted(scenario_df["year"].unique())}')
    print(f'Materials: {scenario_df["material"].nunique()}')
    print()
    print('Demand Statistics (all values in tonnes):')
    print(f'  Max mean: {scenario_df["mean"].max():.2e}')
    print(f'  Max median: {scenario_df["p50"].max():.2e}')
    print(f'  Max 95th percentile: {scenario_df["p95"].max():.2e}')
    print()
    print('Uncertainty Statistics:')
    print(f'  Median CV: {cv_valid.median():.1f}%')
    print(f'  Max CV: {cv_valid.max():.1f}%')
    print(f'  Estimates with CV > 100%: {(cv_valid > 100).sum()} ({100*(cv_valid > 100).sum()/len(cv_valid):.1f}%)')
    print(f'  Estimates with CV > 50%: {(cv_valid > 50).sum()} ({100*(cv_valid > 50).sum()/len(cv_valid):.1f}%)')
    print()

    # Check for any remaining extreme values
    mean_median_ratio = scenario_df['mean'] / scenario_df['p50']
    extreme_ratios = mean_median_ratio[mean_median_ratio > 10]
    if len(extreme_ratios) > 0:
        print(f'WARNING: {len(extreme_ratios)} estimates have mean/median > 10')
        print('  This may indicate remaining distribution issues')
    else:
        print('All estimates have reasonable mean/median ratios (< 10)')


if __name__ == '__main__':
    create_monte_carlo_visualization()
