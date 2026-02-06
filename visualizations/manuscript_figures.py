"""
Manuscript-Style Figures

Recreates key visualizations from the materials demand manuscript:
- Fig. 2: Material demand by technology (stacked bar)
- Fig. S1: Capacity projections by technology
- Fig. S3: Material intensity distributions

Author: Generated for Materials Demand Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def format_large_number(x, pos):
    """Format large numbers with K/M/B suffixes."""
    if x >= 1e9:
        return f'{x/1e9:.1f}B'
    elif x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    else:
        return f'{x:.0f}'


def load_data():
    """Load all required data files."""
    base_dir = Path(__file__).resolve().parent.parent

    # Technology contributions
    tech_contrib = pd.read_csv(base_dir / "outputs" / "data" / "sensitivity" / "technology_contributions.csv")

    # Material demand by scenario
    demand = pd.read_csv(base_dir / "outputs" / "data" / "material_demand_by_scenario.csv")

    # NREL capacity data (skip 3 header rows)
    nrel = pd.read_csv(base_dir / "data" / "StdScen24_annual_national.csv", skiprows=3)

    # Intensity data
    intensity = pd.read_csv(base_dir / "data" / "intensity_data.csv")

    return tech_contrib, demand, nrel, intensity


# ============================================================================
# FIG. 2 STYLE: MATERIAL DEMAND BY TECHNOLOGY
# ============================================================================

def create_technology_breakdown_chart(tech_contrib, materials=None, output_path=None, figsize=(14, 10)):
    """
    Create stacked horizontal bar chart showing technology contributions
    to material demand (Fig. 2 style from manuscript).

    Parameters
    ----------
    tech_contrib : DataFrame
        Technology contributions data with columns:
        material, technology, intensity_t_per_mw, total_capacity_mw,
        expected_demand_tonnes, pct_contribution
    materials : list, optional
        Subset of materials to plot. If None, uses top 15 by total demand.
    output_path : Path, optional
        If provided, save figure to this path.
    """
    # Aggregate by material and technology
    pivot = tech_contrib.pivot_table(
        index='material',
        columns='technology',
        values='expected_demand_tonnes',
        aggfunc='sum',
        fill_value=0
    )

    # Get total demand per material and sort
    total_demand = pivot.sum(axis=1).sort_values(ascending=True)

    # Select materials
    if materials:
        materials_to_plot = [m for m in materials if m in total_demand.index]
    else:
        # Top 15 by demand
        materials_to_plot = total_demand.tail(15).index.tolist()

    pivot = pivot.loc[materials_to_plot]

    # Get technologies with non-zero contribution, sorted by total
    tech_totals = pivot.sum(axis=0)
    technologies = tech_totals[tech_totals > 0].sort_values(ascending=False).index.tolist()

    # Limit to top 10 technologies, group rest as "Other"
    if len(technologies) > 10:
        top_techs = technologies[:9]
        other_techs = technologies[9:]
        pivot_plot = pivot[top_techs].copy()
        pivot_plot['Other'] = pivot[other_techs].sum(axis=1)
        technologies = top_techs + ['Other']
    else:
        pivot_plot = pivot[technologies]

    # Color palette for technologies
    colors = plt.cm.Set3(np.linspace(0, 1, len(technologies)))
    tech_colors = dict(zip(technologies, colors))

    # Special colors for key technologies
    special_colors = {
        'utility-scale solar pv': '#FFD700',  # Gold
        'Solar Distributed': '#FFA500',        # Orange
        'onshore wind': '#4169E1',             # Royal Blue
        'offshore wind': '#1E90FF',            # Dodger Blue
        'Nuclear New': '#9370DB',              # Medium Purple
        'Hydro': '#20B2AA',                    # Light Sea Green
        'Coal': '#696969',                     # Dim Gray
        'Gas': '#A9A9A9',                      # Dark Gray
        'NGCC': '#A9A9A9',
        'NGGT': '#C0C0C0',
        'Other': '#D3D3D3',                    # Light Gray
    }
    for tech, color in special_colors.items():
        if tech in tech_colors:
            tech_colors[tech] = color

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot stacked horizontal bars
    left = np.zeros(len(materials_to_plot))
    for tech in technologies:
        values = pivot_plot[tech].values
        ax.barh(
            range(len(materials_to_plot)),
            values,
            left=left,
            label=tech,
            color=tech_colors[tech],
            edgecolor='white',
            linewidth=0.5,
        )
        left += values

    # Formatting
    ax.set_yticks(range(len(materials_to_plot)))
    ax.set_yticklabels(materials_to_plot, fontsize=10)
    ax.set_xlabel('Material Demand (tonnes)', fontsize=12, fontweight='bold')
    ax.set_title('Material Demand by Technology\n(Cumulative across all scenarios and years)',
                 fontsize=14, fontweight='bold', pad=20)

    ax.xaxis.set_major_formatter(FuncFormatter(format_large_number))

    # Legend outside plot
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.95,
        fontsize=9,
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, ax


def create_technology_pie_charts(tech_contrib, materials=None, output_path=None):
    """
    Create pie charts showing technology breakdown for each material.

    Parameters
    ----------
    tech_contrib : DataFrame
        Technology contributions data.
    materials : list, optional
        Materials to plot. If None, uses top 12 by demand.
    """
    # Get total demand per material
    total_by_mat = tech_contrib.groupby('material')['expected_demand_tonnes'].sum()

    if materials is None:
        materials = total_by_mat.nlargest(12).index.tolist()

    # Create subplot grid
    n_cols = 4
    n_rows = (len(materials) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for idx, material in enumerate(materials):
        ax = axes[idx]

        # Get technology breakdown for this material
        mat_data = tech_contrib[tech_contrib['material'] == material].copy()
        mat_data = mat_data[mat_data['expected_demand_tonnes'] > 0]
        mat_data = mat_data.sort_values('expected_demand_tonnes', ascending=False)

        # Group small contributors into "Other"
        total = mat_data['expected_demand_tonnes'].sum()
        mat_data['pct'] = mat_data['expected_demand_tonnes'] / total * 100

        main_techs = mat_data[mat_data['pct'] >= 5].copy()
        other = mat_data[mat_data['pct'] < 5]['expected_demand_tonnes'].sum()

        if other > 0:
            other_row = pd.DataFrame({
                'technology': ['Other'],
                'expected_demand_tonnes': [other],
                'pct': [other / total * 100]
            })
            main_techs = pd.concat([main_techs, other_row], ignore_index=True)

        # Plot pie
        wedges, texts, autotexts = ax.pie(
            main_techs['expected_demand_tonnes'],
            labels=None,
            autopct=lambda p: f'{p:.1f}%' if p >= 5 else '',
            pctdistance=0.75,
            colors=plt.cm.Set3(np.linspace(0, 1, len(main_techs))),
        )

        ax.set_title(material, fontsize=11, fontweight='bold')

        # Add legend for first plot only
        if idx == 0:
            ax.legend(
                main_techs['technology'],
                loc='center left',
                bbox_to_anchor=(-0.5, 0.5),
                fontsize=8,
            )

    # Hide unused subplots
    for idx in range(len(materials), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Technology Contribution to Material Demand by Material',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


# ============================================================================
# FIG. S1 STYLE: CAPACITY PROJECTIONS
# ============================================================================

def create_capacity_projections_chart(nrel, scenario='Mid_Case', output_path=None, figsize=(12, 8)):
    """
    Create stacked area chart showing capacity projections by technology (Fig. S1 style).

    Parameters
    ----------
    nrel : DataFrame
        NREL Standard Scenarios capacity data.
    scenario : str
        Scenario to plot.
    output_path : Path, optional
        If provided, save figure to this path.
    """
    # Filter to scenario
    data = nrel[nrel['scenario'] == scenario].copy()

    if len(data) == 0:
        print(f"Scenario '{scenario}' not found. Available: {nrel['scenario'].unique()[:10]}...")
        return None

    # Get MW columns
    mw_cols = [c for c in data.columns if c.endswith('_MW')]

    # Reshape for plotting
    years = data['t'].values

    # Group technologies
    tech_groups = {
        'Solar PV': ['upv_MW', 'distpv_MW'],
        'Wind': ['wind_onshore_MW', 'wind_offshore_MW'],
        'Nuclear': ['nuclear_MW', 'nuclear_smr_MW'],
        'Hydro': ['hydro_MW', 'pumped-hydro_MW'],
        'Natural Gas': ['gas_cc_MW', 'gas_ct_MW', 'gas_cc_ccs_MW'],
        'Coal': ['coal_MW', 'coal_ccs_MW'],
        'Battery': ['battery_4_MW', 'battery_8_MW'],
        'Other': ['bio_MW', 'bio-ccs_MW', 'geo_MW', 'csp_MW', 'h2-ct_MW'],
    }

    # Aggregate by group
    grouped = {}
    for group, cols in tech_groups.items():
        existing_cols = [c for c in cols if c in data.columns]
        if existing_cols:
            grouped[group] = data[existing_cols].sum(axis=1).values

    # Colors
    group_colors = {
        'Solar PV': '#FFD700',
        'Wind': '#4169E1',
        'Nuclear': '#9370DB',
        'Hydro': '#20B2AA',
        'Natural Gas': '#A9A9A9',
        'Coal': '#696969',
        'Battery': '#32CD32',
        'Other': '#D3D3D3',
    }

    # Plot stacked area
    fig, ax = plt.subplots(figsize=figsize)

    bottom = np.zeros(len(years))
    for group in ['Coal', 'Natural Gas', 'Nuclear', 'Hydro', 'Other', 'Wind', 'Solar PV', 'Battery']:
        if group in grouped:
            values = grouped[group] / 1000  # Convert to GW
            ax.fill_between(years, bottom, bottom + values,
                           label=group, color=group_colors[group], alpha=0.8)
            bottom += values

    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Installed Capacity (GW)', fontsize=12, fontweight='bold')
    ax.set_title(f'U.S. Electricity Generation Capacity Projections\n{scenario.replace("_", " ")}',
                 fontsize=14, fontweight='bold', pad=20)

    ax.legend(loc='upper left', frameon=True, framealpha=0.95)
    ax.set_xlim(years.min(), years.max())
    ax.set_ylim(0, None)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, ax


def create_capacity_additions_chart(nrel, scenario='Mid_Case', output_path=None, figsize=(12, 8)):
    """
    Create bar chart showing annual capacity additions by technology.
    """
    # Filter to scenario
    data = nrel[nrel['scenario'] == scenario].copy().sort_values('t')

    if len(data) == 0:
        return None

    years = data['t'].values

    # Get MW columns and calculate year-over-year additions
    mw_cols = [c for c in data.columns if c.endswith('_MW')]

    additions = data[mw_cols].diff().fillna(0)
    additions['year'] = years
    additions = additions[additions['year'] >= years.min() + 1]  # Skip first year

    # Group technologies
    tech_groups = {
        'Solar PV': ['upv_MW', 'distpv_MW'],
        'Wind': ['wind_onshore_MW', 'wind_offshore_MW'],
        'Nuclear': ['nuclear_MW', 'nuclear_smr_MW'],
        'Battery': ['battery_4_MW', 'battery_8_MW'],
        'Gas': ['gas_cc_MW', 'gas_ct_MW'],
        'Other': ['bio_MW', 'geo_MW', 'csp_MW', 'hydro_MW'],
    }

    group_colors = {
        'Solar PV': '#FFD700',
        'Wind': '#4169E1',
        'Nuclear': '#9370DB',
        'Battery': '#32CD32',
        'Gas': '#A9A9A9',
        'Other': '#D3D3D3',
    }

    # Aggregate
    grouped = {}
    for group, cols in tech_groups.items():
        existing_cols = [c for c in cols if c in additions.columns]
        if existing_cols:
            grouped[group] = additions[existing_cols].sum(axis=1).values / 1000  # GW

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(additions))
    width = 0.8
    bottom = np.zeros(len(additions))

    for group in ['Other', 'Gas', 'Nuclear', 'Battery', 'Wind', 'Solar PV']:
        if group in grouped:
            values = np.maximum(grouped[group], 0)  # Only positive additions
            ax.bar(x, values, width, bottom=bottom, label=group, color=group_colors[group])
            bottom += values

    ax.set_xticks(x[::2])
    ax.set_xticklabels(additions['year'].values[::2].astype(int), rotation=45)
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Capacity Additions (GW)', fontsize=12, fontweight='bold')
    ax.set_title(f'Annual Capacity Additions\n{scenario.replace("_", " ")}',
                 fontsize=14, fontweight='bold', pad=20)

    ax.legend(loc='upper left', frameon=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, ax


# ============================================================================
# FIG. S3 STYLE: MATERIAL INTENSITY DISTRIBUTIONS
# ============================================================================

def create_intensity_distributions_chart(intensity, materials=None, output_path=None):
    """
    Create violin/box plots showing material intensity distributions (Fig. S3 style).

    Parameters
    ----------
    intensity : DataFrame
        Raw intensity data with columns: technology, Material, value (t/GW).
    materials : list, optional
        Materials to plot. If None, uses materials with most data points.
    """
    # Standardize column names
    if 'Material' in intensity.columns:
        intensity = intensity.rename(columns={'Material': 'material'})

    # Get materials with most data points
    mat_counts = intensity.groupby('material').size().sort_values(ascending=False)

    if materials is None:
        materials = mat_counts.head(16).index.tolist()

    # Filter data
    data = intensity[intensity['material'].isin(materials)].copy()

    # Create subplot grid
    n_cols = 4
    n_rows = (len(materials) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()

    for idx, material in enumerate(materials):
        ax = axes[idx]
        mat_data = data[data['material'] == material]['value'].dropna()

        if len(mat_data) > 0:
            # Create violin plot
            parts = ax.violinplot(mat_data, positions=[0], showmeans=True, showmedians=True)

            # Color the violin
            for pc in parts['bodies']:
                pc.set_facecolor('#4169E1')
                pc.set_alpha(0.6)

            # Add individual points
            ax.scatter(
                np.zeros(len(mat_data)) + np.random.uniform(-0.1, 0.1, len(mat_data)),
                mat_data,
                alpha=0.5,
                s=20,
                c='black',
            )

            # Add statistics text
            ax.text(0.95, 0.95,
                   f'n={len(mat_data)}\nmean={mat_data.mean():.1f}\nmed={mat_data.median():.1f}',
                   transform=ax.transAxes, ha='right', va='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title(material, fontsize=10, fontweight='bold')
        ax.set_ylabel('Intensity (t/GW)', fontsize=9)
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused subplots
    for idx in range(len(materials), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Material Intensity Distributions\n(from literature review)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def create_intensity_by_technology_chart(intensity, material='Copper', output_path=None, figsize=(12, 6)):
    """
    Create box plot showing intensity variation by technology for a specific material.
    """
    # Standardize column names
    if 'Material' in intensity.columns:
        intensity = intensity.rename(columns={'Material': 'material'})

    # Filter to material
    data = intensity[intensity['material'] == material].copy()

    if len(data) == 0:
        print(f"No data for {material}")
        return None

    # Get technologies with data
    tech_counts = data.groupby('technology').size().sort_values(ascending=False)
    technologies = tech_counts[tech_counts >= 2].index.tolist()[:15]

    data = data[data['technology'].isin(technologies)]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Box plot
    bp = ax.boxplot(
        [data[data['technology'] == t]['value'].values for t in technologies],
        labels=technologies,
        patch_artist=True,
    )

    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(technologies)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(technologies, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(f'{material} Intensity (t/GW)', fontsize=12, fontweight='bold')
    ax.set_title(f'{material} Intensity by Technology',
                 fontsize=14, fontweight='bold', pad=20)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, ax


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Generate all manuscript-style figures."""
    print("=" * 70)
    print("GENERATING MANUSCRIPT-STYLE FIGURES")
    print("=" * 70)

    # Setup output directory
    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / "outputs" / "figures" / "manuscript"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    tech_contrib, demand, nrel, intensity = load_data()
    print(f"  Technology contributions: {len(tech_contrib)} rows")
    print(f"  Material demand: {len(demand)} rows")
    print(f"  NREL capacity: {len(nrel)} rows")
    print(f"  Intensity data: {len(intensity)} rows")

    # --- Fig. 2 style: Technology breakdown ---
    print("\n" + "-" * 50)
    print("FIG. 2 STYLE: Material Demand by Technology")
    print("-" * 50)

    print("\nGenerating technology breakdown chart...")
    create_technology_breakdown_chart(
        tech_contrib,
        output_path=output_dir / "fig2_technology_breakdown.png"
    )

    print("\nGenerating technology pie charts...")
    create_technology_pie_charts(
        tech_contrib,
        output_path=output_dir / "fig2_technology_pies.png"
    )

    # --- Fig. S1 style: Capacity projections ---
    print("\n" + "-" * 50)
    print("FIG. S1 STYLE: Capacity Projections")
    print("-" * 50)

    # Try a few scenarios
    for scenario in ['Mid_Case', 'Adv_RE', 'High_NG']:
        if scenario in nrel['scenario'].values:
            print(f"\nGenerating capacity chart for {scenario}...")
            create_capacity_projections_chart(
                nrel,
                scenario=scenario,
                output_path=output_dir / f"figs1_capacity_{scenario.lower()}.png"
            )

            print(f"Generating capacity additions for {scenario}...")
            create_capacity_additions_chart(
                nrel,
                scenario=scenario,
                output_path=output_dir / f"figs1_additions_{scenario.lower()}.png"
            )
            break

    # --- Fig. S3 style: Intensity distributions ---
    print("\n" + "-" * 50)
    print("FIG. S3 STYLE: Material Intensity Distributions")
    print("-" * 50)

    print("\nGenerating intensity distributions...")
    create_intensity_distributions_chart(
        intensity,
        output_path=output_dir / "figs3_intensity_distributions.png"
    )

    print("\nGenerating intensity by technology (Copper)...")
    create_intensity_by_technology_chart(
        intensity,
        material='Copper',
        output_path=output_dir / "figs3_intensity_copper_by_tech.png"
    )

    # Additional intensity figures for key materials
    key_materials = ['Steel', 'Aluminum', 'Silicon', 'Neodymium', 'Dysprosium']
    for material in key_materials:
        print(f"\nGenerating intensity by technology ({material})...")
        fig = create_intensity_by_technology_chart(
            intensity,
            material=material,
            output_path=output_dir / f"figs3_intensity_{material.lower()}_by_tech.png"
        )
        if fig is None:
            print(f"  Skipped {material} - insufficient data")

    print("\n" + "=" * 70)
    print(f"All figures saved to: {output_dir}")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
