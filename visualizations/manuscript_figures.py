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

def create_technology_breakdown_chart(tech_contrib, materials=None, output_path=None, figsize=(14, 12)):
    """
    100% stacked horizontal bar chart showing the percentage contribution
    of each technology to total demand for every material.

    Each material gets an equal-length bar (0–100%), making the technology
    concentration pattern visible regardless of absolute demand magnitude.

    Parameters
    ----------
    tech_contrib : DataFrame
        Technology contributions data with columns:
        material, technology, expected_demand_tonnes, pct_contribution
    materials : list, optional
        Subset of materials to plot. If None, plots all materials.
    output_path : Path, optional
        If provided, save figure to this path.
    """
    # Consolidate UPV sub-technologies: ASIGE, CDTE, CIGS all represent the
    # 70/15/15 split of utility-scale solar PV.  For materials where all three
    # sub-techs carry equal contributions (structural BOS materials like
    # aluminum, glass, copper, steel, cement), merge them into a single
    # "utility-scale solar pv" entry so the figure doesn't misleadingly show
    # three identical segments.  Keep them separate for materials where the
    # sub-techs have genuinely different intensities (cadmium, tellurium,
    # indium, gallium, selenium, germanium — true thin-film-specific materials).
    tc = tech_contrib.copy()
    upv_subtechs = ['ASIGE', 'CDTE', 'CIGS']
    thin_film_materials = {
        'Cadmium', 'Tellurium', 'Indium', 'Gallium', 'Selenium',
        'Germanium', 'Gadium', 'Molybdenum',
    }
    rows_to_drop = []
    rows_to_add = []
    for mat in tc['material'].unique():
        if mat in thin_film_materials:
            continue
        mask = (tc['material'] == mat) & (tc['technology'].isin(upv_subtechs))
        idx = tc.index[mask]
        if len(idx) > 0:
            merged_demand = tc.loc[idx, 'expected_demand_tonnes'].sum()
            merged_pct = tc.loc[idx, 'pct_contribution'].sum()
            # Check if there's already a 'utility-scale solar pv' row
            upv_idx = tc.index[
                (tc['material'] == mat) & (tc['technology'] == 'utility-scale solar pv')
            ]
            if len(upv_idx) > 0:
                tc.loc[upv_idx[0], 'expected_demand_tonnes'] += merged_demand
                tc.loc[upv_idx[0], 'pct_contribution'] += merged_pct
            else:
                new_row = tc.loc[idx[0]].copy()
                new_row['technology'] = 'utility-scale solar pv'
                new_row['expected_demand_tonnes'] = merged_demand
                new_row['pct_contribution'] = merged_pct
                rows_to_add.append(new_row)
            rows_to_drop.extend(idx.tolist())

    if rows_to_drop:
        tc = tc.drop(index=rows_to_drop)
    if rows_to_add:
        tc = pd.concat([tc, pd.DataFrame(rows_to_add)], ignore_index=True)

    # Drop "Gadium" (data typo — likely Gadolinium, but only 1 entry with
    # 100% CIGS; not meaningful for the figure)
    tc = tc[tc['material'] != 'Gadium']

    # Pivot to percentage contributions
    pivot_pct = tc.pivot_table(
        index='material',
        columns='technology',
        values='pct_contribution',
        aggfunc='sum',
        fill_value=0
    )

    # Select materials
    if materials:
        materials_to_plot = [m for m in materials if m in pivot_pct.index]
    else:
        materials_to_plot = sorted(pivot_pct.index.tolist())

    pivot_pct = pivot_pct.loc[materials_to_plot]

    # Identify the dominant technology per material for sorting
    dominant_tech = pivot_pct.idxmax(axis=1)

    # Define technology display order (most important first)
    tech_order = [
        'onshore wind', 'utility-scale solar pv', 'offshore wind',
        'Hydro', 'CDTE', 'CIGS', 'ASIGE', 'a-Si',
        'Solar Distributed', 'Nuclear New', 'Gas', 'NGCC', 'NGGT',
        'CSP', 'Geothermal', 'Biomass', 'Bio CCS',
        'Coal', 'Coal CCS', 'Gas CCS',
    ]

    # Sort materials: group by dominant technology, then alphabetically within
    sort_key = []
    for mat in materials_to_plot:
        dom = dominant_tech[mat]
        order_idx = tech_order.index(dom) if dom in tech_order else 99
        sort_key.append((order_idx, mat))
    sort_key.sort(key=lambda x: (x[0], x[1]))
    materials_sorted = [s[1] for s in sort_key]
    # Reverse so first group is at top of plot
    materials_sorted = materials_sorted[::-1]

    pivot_pct = pivot_pct.loc[materials_sorted]

    # Get technologies with any non-zero contribution
    tech_totals = pivot_pct.sum(axis=0)
    active_techs = tech_totals[tech_totals > 0].sort_values(ascending=False).index.tolist()

    # Reorder to match tech_order where possible
    ordered_techs = [t for t in tech_order if t in active_techs]
    remaining = [t for t in active_techs if t not in ordered_techs]
    technologies = ordered_techs + remaining

    # Group minor technologies (<3% max contribution) as "Other"
    major_techs = []
    minor_techs = []
    for tech in technologies:
        if pivot_pct[tech].max() >= 3.0:
            major_techs.append(tech)
        else:
            minor_techs.append(tech)

    pivot_plot = pivot_pct[major_techs].copy()
    if minor_techs:
        other_sum = pivot_pct[minor_techs].sum(axis=1)
        if other_sum.max() > 0:
            pivot_plot['Other'] = other_sum
            major_techs.append('Other')

    technologies = major_techs

    # Technology display names (clean up for legend)
    tech_labels = {
        'onshore wind': 'Onshore Wind',
        'offshore wind': 'Offshore Wind',
        'utility-scale solar pv': 'Utility PV (c-Si)',
        'Solar Distributed': 'Distributed PV',
        'CDTE': 'CdTe',
        'CdTe': 'CdTe',
        'CIGS': 'CIGS',
        'ASIGE': 'a-Si/Ge',
        'a-Si': 'a-Si',
        'Hydro': 'Hydropower',
        'Nuclear New': 'Nuclear',
        'Gas': 'Natural Gas',
        'NGCC': 'Natural Gas (CC)',
        'NGGT': 'Natural Gas (CT)',
        'CSP': 'Concentrated Solar',
        'Geothermal': 'Geothermal',
        'Biomass': 'Biomass',
        'Bio CCS': 'Biomass + CCS',
        'Coal': 'Coal',
        'Coal CCS': 'Coal + CCS',
        'Gas CCS': 'Gas + CCS',
        'Other': 'Other',
    }

    # Color palette — perceptually distinct, grouped by energy family.
    # Wind:    blue family (cool)
    # Solar:   warm family with wide hue separation
    # Hydro:   teal/cyan (water, distinct from wind blue)
    # Nuclear: purple
    # Fossil:  gray family (3 distinct grays)
    # Biomass: brown
    tech_colors = {
        # Wind — blue family
        'onshore wind': '#2166AC',          # Strong blue
        'offshore wind': '#72B4D9',          # Sky blue (lighter, distinct)
        # Solar — warm but spread across hue range
        'utility-scale solar pv': '#E8A525', # Amber/gold (sun)
        'Solar Distributed': '#F4D03F',      # Bright yellow (distinct from amber)
        'CDTE': '#C0392B',                   # Crimson red
        'CdTe': '#C0392B',                   # Crimson red (alias)
        'CIGS': '#E67E22',                   # Tangerine orange
        'ASIGE': '#F5CBA7',                  # Peach (pale, distinct)
        'a-Si': '#D4AC0D',                   # Dark gold
        'CSP': '#CD6155',                    # Dusty rose (warm, distinct from orange)
        # Hydro — teal/cyan (water-intuitive, distinct from wind blues)
        'Hydro': '#17A589',                  # Teal
        # Nuclear — purple (unique hue, no confusion)
        'Nuclear New': '#7D3C98',            # Royal purple
        # Fossil/thermal — gray family (perceptually stepped)
        'Gas': '#7B8D8E',                    # Blue-gray
        'NGCC': '#5D6D7E',                   # Steel gray (darker)
        'NGGT': '#AEB6BF',                   # Silver (lighter)
        'Coal': '#2C3E50',                   # Charcoal
        'Coal CCS': '#566573',              # Slate gray
        'Gas CCS': '#95A5A6',               # Ash gray
        # Biomass — earth tones
        'Biomass': '#7E5109',                # Dark brown
        'Bio CCS': '#B87333',               # Copper/light brown
        # Geothermal — distinct green (earth heat)
        'Geothermal': '#27AE60',             # Emerald green
        # Catch-all
        'Other': '#D5D8DC',                 # Very light gray
    }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(materials_sorted))
    bar_height = 0.7

    # Plot 100% stacked horizontal bars
    left = np.zeros(len(materials_sorted))
    for tech in technologies:
        values = pivot_plot[tech].values
        label = tech_labels.get(tech, tech)
        color = tech_colors.get(tech, '#D9D9D9')
        ax.barh(
            y_pos,
            values,
            height=bar_height,
            left=left,
            label=label,
            color=color,
            edgecolor='white',
            linewidth=0.4,
        )

        left += values

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(materials_sorted, fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Contribution to Total Material Demand (%)',
                  fontsize=11, fontweight='bold')
    ax.set_title(
        'Technology Decomposition of Material Demand',
        fontsize=14, fontweight='bold', pad=15,
    )

    # Vertical gridlines at 25% intervals
    for x in [25, 50, 75]:
        ax.axvline(x, color='#cccccc', linewidth=0.5, linestyle='--', zorder=0)

    # Legend — deduplicate and place below
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)

    ax.legend(
        unique_handles, unique_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.06),
        ncol=4,
        frameon=True,
        framealpha=0.95,
        fontsize=8.5,
        columnspacing=1.0,
        handlelength=1.5,
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.tick_params(axis='y', length=0)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def create_intensity_by_technology_chart(intensity, material='Copper', output_path=None, figsize=None):
    """
    Create box plot showing intensity variation by technology for a specific material.

    Figure size is dynamically scaled to the number of technologies so that
    2-box figures don't waste a full page of white space.
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

    # Technology display names (clean model identifiers → readable labels)
    tech_labels = {
        'onshore wind': 'Onshore Wind',
        'offshore wind': 'Offshore Wind',
        'utility-scale solar pv': 'Utility PV (c-Si)',
        'Solar Distributed': 'Distributed PV',
        'CDTE': 'CdTe', 'CdTe': 'CdTe',
        'CIGS': 'CIGS',
        'ASIGE': 'a-Si/Ge',
        'a-Si': 'a-Si',
        'Hydro': 'Hydropower',
        'Nuclear New': 'Nuclear',
        'Gas': 'Natural Gas',
        'NGCC': 'Natural Gas (CC)',
        'NGGT': 'Natural Gas (CT)',
        'CSP': 'Concentrated Solar',
        'Geothermal': 'Geothermal',
        'Biomass': 'Biomass',
        'Bio CCS': 'Biomass + CCS',
        'Coal': 'Coal', 'Coal CCS': 'Coal + CCS',
        'Gas CCS': 'Gas + CCS',
    }

    # Energy-family color palette (consistent with fig2)
    tech_colors = {
        'onshore wind': '#2166AC', 'offshore wind': '#72B4D9',
        'utility-scale solar pv': '#E8A525', 'Solar Distributed': '#F4D03F',
        'CDTE': '#C0392B', 'CdTe': '#C0392B',
        'CIGS': '#E67E22', 'ASIGE': '#F5CBA7', 'a-Si': '#D4AC0D',
        'CSP': '#CD6155',
        'Hydro': '#17A589',
        'Nuclear New': '#7D3C98',
        'Gas': '#7B8D8E', 'NGCC': '#5D6D7E', 'NGGT': '#AEB6BF',
        'Coal': '#2C3E50', 'Coal CCS': '#566573', 'Gas CCS': '#95A5A6',
        'Biomass': '#7E5109', 'Bio CCS': '#B87333',
        'Geothermal': '#27AE60',
    }

    # Dynamic figsize: width scales with # of technologies, min 5"
    n_tech = len(technologies)
    if figsize is None:
        width = max(5, n_tech * 1.2 + 2)
        figsize = (width, 6)

    fig, ax = plt.subplots(figsize=figsize)

    # Box plot
    display_labels = [tech_labels.get(t, t) for t in technologies]
    bp = ax.boxplot(
        [data[data['technology'] == t]['value'].values for t in technologies],
        tick_labels=display_labels,
        patch_artist=True,
    )

    # Color boxes by energy family
    for patch, tech in zip(bp['boxes'], technologies):
        color = tech_colors.get(tech, '#B0BEC5')
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(f'{material} Intensity (t/GW)', fontsize=12, fontweight='bold')
    ax.set_title(f'{material} Intensity by Technology',
                 fontsize=14, fontweight='bold', pad=20)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
