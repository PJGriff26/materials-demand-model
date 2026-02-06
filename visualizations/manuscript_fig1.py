"""
Manuscript Figure 1: Scenario-Range Material Demand with Uncertainty Bands

Creates publication-quality figure showing annual material demand (2026-2050)
with uncertainty bands representing the range across all 61 NREL scenarios.

For each material:
- Solid line: Median demand across scenarios (p50)
- Shaded band: 95% scenario range (p2.5 to p97.5 across scenario medians)
- Light band: Full range (min to max scenario median)

Author: Materials Demand Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def format_demand(x, pos):
    """Format demand values with K/M/B suffixes."""
    if x >= 1e9:
        return f'{x/1e9:.1f}B'
    elif x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    elif x >= 1:
        return f'{x:.0f}'
    else:
        return f'{x:.2g}'


def load_demand_data():
    """Load material demand data."""
    base_dir = Path(__file__).resolve().parent.parent
    demand_file = base_dir / "outputs" / "data" / "material_demand_by_scenario.csv"

    if not demand_file.exists():
        raise FileNotFoundError(f"Demand data not found: {demand_file}")

    return pd.read_csv(demand_file)


def calculate_scenario_statistics(demand):
    """
    Calculate statistics across scenarios for each material and year.

    Returns DataFrame with columns:
    - material, year
    - median_demand: median of scenario medians (p50)
    - p2_5, p97_5: 2.5th and 97.5th percentiles of scenario medians
    - min_demand, max_demand: full range
    """
    # Use the p50 (median) from each scenario's MC distribution
    # as the "demand" for that scenario
    scenario_medians = demand.pivot_table(
        index=['material', 'year'],
        columns='scenario',
        values='p50',
        aggfunc='first'
    )

    # Calculate statistics across scenarios
    stats = pd.DataFrame({
        'median_demand': scenario_medians.median(axis=1),
        'mean_demand': scenario_medians.mean(axis=1),
        'p2_5': scenario_medians.quantile(0.025, axis=1),
        'p5': scenario_medians.quantile(0.05, axis=1),
        'p25': scenario_medians.quantile(0.25, axis=1),
        'p75': scenario_medians.quantile(0.75, axis=1),
        'p95': scenario_medians.quantile(0.95, axis=1),
        'p97_5': scenario_medians.quantile(0.975, axis=1),
        'min_demand': scenario_medians.min(axis=1),
        'max_demand': scenario_medians.max(axis=1),
        'n_scenarios': scenario_medians.notna().sum(axis=1),
    }).reset_index()

    return stats


def create_fig1_demand_curves(
    demand,
    materials=None,
    output_path=None,
    figsize=(16, 12),
    n_cols=3
):
    """
    Create Figure 1: Annual material demand with scenario uncertainty bands.

    Parameters
    ----------
    demand : DataFrame
        Material demand data with columns: scenario, year, material, mean, p50, etc.
    materials : list, optional
        Materials to plot. If None, uses top 12 by median demand.
    output_path : Path, optional
        Save figure to this path.
    figsize : tuple
        Figure size in inches.
    n_cols : int
        Number of columns in subplot grid.

    Returns
    -------
    fig : Figure
    """
    # Calculate statistics across scenarios
    stats = calculate_scenario_statistics(demand)

    # Select materials
    if materials is None:
        # Top 12 by median cumulative demand
        total_demand = stats.groupby('material')['median_demand'].sum()
        materials = total_demand.nlargest(12).index.tolist()

    # Create subplot grid
    n_materials = len(materials)
    n_rows = (n_materials + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Colors for consistency
    median_color = '#2E86AB'  # Steel blue
    band_95_color = '#A6E1FA'  # Light blue
    band_full_color = '#E8F4F8'  # Very light blue

    for idx, material in enumerate(materials):
        ax = axes[idx]
        mat_stats = stats[stats['material'] == material].sort_values('year')

        if len(mat_stats) == 0:
            ax.text(0.5, 0.5, f'No data for {material}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(material, fontsize=11, fontweight='bold')
            continue

        years = mat_stats['year'].values

        # Plot full range (min to max)
        ax.fill_between(
            years,
            mat_stats['min_demand'].values,
            mat_stats['max_demand'].values,
            color=band_full_color,
            alpha=0.8,
            label='Full range'
        )

        # Plot 95% interval (p2.5 to p97.5)
        ax.fill_between(
            years,
            mat_stats['p2_5'].values,
            mat_stats['p97_5'].values,
            color=band_95_color,
            alpha=0.8,
            label='95% interval'
        )

        # Plot IQR (p25 to p75)
        ax.fill_between(
            years,
            mat_stats['p25'].values,
            mat_stats['p75'].values,
            color=median_color,
            alpha=0.3,
            label='IQR (25-75%)'
        )

        # Plot median line
        ax.plot(
            years,
            mat_stats['median_demand'].values,
            color=median_color,
            linewidth=2,
            label='Median'
        )

        # Formatting
        ax.set_title(material, fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel('Annual Demand (tonnes)', fontsize=9)
        ax.yaxis.set_major_formatter(FuncFormatter(format_demand))

        # Set x-axis to show only key years
        ax.set_xticks([2026, 2032, 2038, 2044, 2050])
        ax.tick_params(axis='both', labelsize=8)

        # Grid and spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)

        # Add number of scenarios annotation
        n_scen = mat_stats['n_scenarios'].iloc[0]
        ax.text(0.02, 0.98, f'n={n_scen} scenarios',
               transform=ax.transAxes, ha='left', va='top',
               fontsize=7, color='gray')

    # Hide unused subplots
    for idx in range(n_materials, len(axes)):
        axes[idx].set_visible(False)

    # Add legend to first subplot
    axes[0].legend(loc='upper left', fontsize=7, frameon=True, framealpha=0.9)

    # Suptitle
    fig.suptitle(
        'Annual Material Demand for U.S. Electricity Infrastructure (2026-2050)\n'
        'Uncertainty bands represent scenario variation across 61 NREL Standard Scenarios',
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

    return fig


def create_fig1_single_panel(
    demand,
    materials=None,
    output_path=None,
    figsize=(14, 8)
):
    """
    Alternative: Single panel with multiple materials overlaid.
    Uses log scale to show materials with different magnitudes.
    """
    stats = calculate_scenario_statistics(demand)

    if materials is None:
        total_demand = stats.groupby('material')['median_demand'].sum()
        materials = total_demand.nlargest(8).index.tolist()

    fig, ax = plt.subplots(figsize=figsize)

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(materials)))

    for idx, material in enumerate(materials):
        mat_stats = stats[stats['material'] == material].sort_values('year')
        years = mat_stats['year'].values

        # Plot band
        ax.fill_between(
            years,
            mat_stats['p2_5'].values,
            mat_stats['p97_5'].values,
            color=colors[idx],
            alpha=0.2
        )

        # Plot median
        ax.plot(
            years,
            mat_stats['median_demand'].values,
            color=colors[idx],
            linewidth=2,
            label=material
        )

    ax.set_yscale('log')
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annual Demand (tonnes, log scale)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Annual Material Demand for U.S. Electricity Infrastructure\n'
        '(Median with 95% scenario interval)',
        fontsize=14, fontweight='bold'
    )

    ax.legend(loc='upper left', frameon=True, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

    return fig


def create_fig1_cumulative(
    demand,
    materials=None,
    output_path=None,
    figsize=(12, 8)
):
    """
    Alternative: Cumulative demand bar chart with uncertainty.
    Shows total 2026-2050 demand for each material.
    """
    stats = calculate_scenario_statistics(demand)

    # Calculate cumulative demand per material
    cumulative = stats.groupby('material').agg({
        'median_demand': 'sum',
        'p2_5': 'sum',
        'p97_5': 'sum',
        'min_demand': 'sum',
        'max_demand': 'sum',
    }).reset_index()

    if materials is None:
        materials = cumulative.nlargest(15, 'median_demand')['material'].tolist()

    cumulative = cumulative[cumulative['material'].isin(materials)]
    cumulative = cumulative.sort_values('median_demand', ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(cumulative))

    # Plot bars with error bars
    ax.barh(
        y_pos,
        cumulative['median_demand'],
        xerr=[
            cumulative['median_demand'] - cumulative['p2_5'],
            cumulative['p97_5'] - cumulative['median_demand']
        ],
        capsize=3,
        color='#2E86AB',
        alpha=0.8,
        error_kw={'elinewidth': 1, 'capthick': 1}
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cumulative['material'])
    ax.set_xlabel('Cumulative Demand 2026-2050 (tonnes)', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_formatter(FuncFormatter(format_demand))
    ax.set_title(
        'Cumulative Material Demand for U.S. Electricity Infrastructure\n'
        '(Median with 95% scenario interval)',
        fontsize=14, fontweight='bold'
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

    return fig


def main():
    """Generate Figure 1 variants."""
    print("=" * 70)
    print("GENERATING MANUSCRIPT FIGURE 1: Demand Curves with Uncertainty")
    print("=" * 70)

    # Setup
    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / "outputs" / "figures" / "manuscript"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading demand data...")
    demand = load_demand_data()
    print(f"  Loaded {len(demand)} rows")
    print(f"  Scenarios: {demand['scenario'].nunique()}")
    print(f"  Materials: {demand['material'].nunique()}")
    print(f"  Years: {sorted(demand['year'].unique())}")

    # Key materials for manuscript
    key_materials = [
        'Copper', 'Steel', 'Aluminum', 'Cement', 'Silicon',
        'Neodymium', 'Dysprosium', 'Glass', 'Fiberglass',
        'Nickel', 'Manganese', 'Chromium'
    ]

    # Generate multi-panel figure (main Fig 1)
    print("\n" + "-" * 50)
    print("Generating multi-panel demand curves (Fig 1)...")
    print("-" * 50)
    create_fig1_demand_curves(
        demand,
        materials=key_materials,
        output_path=output_dir / "fig1_demand_curves.png"
    )

    # Generate single-panel log scale version
    print("\nGenerating single-panel log scale version...")
    create_fig1_single_panel(
        demand,
        materials=['Copper', 'Steel', 'Aluminum', 'Cement', 'Silicon',
                   'Neodymium', 'Nickel', 'Glass'],
        output_path=output_dir / "fig1_demand_curves_log.png"
    )

    # Generate cumulative demand version
    print("\nGenerating cumulative demand bar chart...")
    create_fig1_cumulative(
        demand,
        materials=key_materials,
        output_path=output_dir / "fig1_cumulative_demand.png"
    )

    print("\n" + "=" * 70)
    print(f"All Figure 1 variants saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
