"""
Publication-Ready Visualizations for Materials Demand Analysis
===============================================================

Creates publication-quality figures for materials demand projections:
1. Spaghetti plots - All scenarios with highlighted key scenarios
2. Time series stacked charts - Technology contributions with uncertainty
3. Material comparison charts - Key materials across scenarios

Design principles:
- Clean, minimal aesthetic
- No legends on spaghetti plots (explained in caption)
- Black dotted lines for uncertainty (not gray shading)
- 4 highlighted scenarios in color, rest in light gray
- Professional typography and colors

Author: Materials Demand Research Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


# ============================================================================
# COLOR SCHEMES
# ============================================================================

# Highlighted scenarios (4 colors)
HIGHLIGHT_COLORS = {
    'Mid_Case': '#2E86AB',           # Blue - baseline
    'Mid_Case_100by2035': '#A23B72', # Magenta - aggressive net-zero
    'Mid_Case_No_IRA': '#F18F01',    # Orange - no IRA counterfactual
    'High_Demand_Growth_100by2035': '#C73E1D'  # Red - maximum demand
}

# Technology colors (for stacked charts)
TECHNOLOGY_COLORS = {
    'Solar': '#FFB627',      # Gold
    'Wind': '#6FCDDD',       # Cyan
    'Nuclear': '#7E52A0',    # Purple
    'Hydro': '#3E7CB1',      # Blue
    'Geothermal': '#81A684', # Green
    'Biomass': '#8B5A3C',    # Brown
    'Coal': '#2F2F2F',       # Dark gray
    'Gas': '#94ADC7',        # Light blue
    'Other': '#CCCCCC'       # Gray
}

# Gray for background scenarios
BACKGROUND_GRAY = '#DDDDDD'
BACKGROUND_GRAY_ALPHA = 0.3


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_figure(figsize=(8, 6)):
    """Create figure with publication settings"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    return fig, ax


def format_large_number(x, pos=None):
    """Format large numbers for axis labels"""
    if x >= 1e9:
        return f'{x/1e9:.1f}B'
    elif x >= 1e6:
        return f'{x/1e6:.0f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    else:
        return f'{x:.0f}'


def save_figure(fig, filepath: Path, formats=['png', 'pdf']):
    """Save figure in multiple formats"""
    filepath = Path(filepath)
    for fmt in formats:
        output_path = filepath.with_suffix(f'.{fmt}')
        fig.savefig(output_path, format=fmt, bbox_inches='tight', dpi=300)
        print(f"✓ Saved: {output_path}")


# ============================================================================
# SPAGHETTI PLOT - ALL SCENARIOS WITH HIGHLIGHTS
# ============================================================================

def create_spaghetti_plot(
    results_df: pd.DataFrame,
    material: str,
    highlight_scenarios: List[str] = None,
    output_path: Optional[Path] = None,
    show_average: bool = True,
    figsize: Tuple[float, float] = (10, 6)
) -> plt.Figure:
    """
    Create spaghetti plot showing all scenarios with highlighted key scenarios.
    
    Design:
    - All scenarios in light gray (no legend)
    - 4 key scenarios highlighted in color
    - Optional mean line across all scenarios
    - No legend (explained in caption)
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from simulation with columns: scenario, year, material, p50
    material : str
        Material to plot
    highlight_scenarios : List[str], optional
        Scenarios to highlight (default: baseline, 100by2035, no-IRA, max-demand)
    output_path : Path, optional
        Where to save figure
    show_average : bool
        Whether to show mean across all scenarios
    
    Returns
    -------
    fig : matplotlib.Figure
    """
    if highlight_scenarios is None:
        highlight_scenarios = list(HIGHLIGHT_COLORS.keys())
    
    # Filter data
    data = results_df[results_df['material'] == material].copy()
    
    if len(data) == 0:
        raise ValueError(f"No data for material: {material}")
    
    # Create figure
    fig, ax = setup_figure(figsize=figsize)
    
    years = sorted(data['year'].unique())
    scenarios = data['scenario'].unique()
    
    # Plot background scenarios (gray)
    for scenario in scenarios:
        if scenario not in highlight_scenarios:
            scenario_data = data[data['scenario'] == scenario].sort_values('year')
            ax.plot(scenario_data['year'], scenario_data['p50'], 
                   color=BACKGROUND_GRAY, alpha=BACKGROUND_GRAY_ALPHA, 
                   linewidth=1.0, zorder=1)
    
    # Calculate and plot mean (if requested)
    if show_average:
        mean_by_year = data.groupby('year')['p50'].mean().reset_index()
        ax.plot(mean_by_year['year'], mean_by_year['p50'],
               color='black', linewidth=2.0, linestyle='-', 
               alpha=0.7, zorder=3, label='Mean')
    
    # Plot highlighted scenarios (color, on top)
    for scenario in highlight_scenarios:
        if scenario in scenarios:
            scenario_data = data[data['scenario'] == scenario].sort_values('year')
            color = HIGHLIGHT_COLORS.get(scenario, 'black')
            
            # Create readable label
            label = scenario.replace('_', ' ')
            # Add line breaks for long labels to improve readability
            if len(label) > 25:
                words = label.split()
                mid = len(words) // 2
                label = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
            
            ax.plot(scenario_data['year'], scenario_data['p50'],
                   color=color, linewidth=2.5, zorder=4,
                   label=label)
    
    # Formatting
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{material} Demand (mt)', fontsize=12, fontweight='bold')
    ax.set_title(f'{material} Demand Across Scenarios', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Format y-axis with large number formatting
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(format_large_number))
    
    # Add clear legend showing highlighted scenarios
    # Place in upper left by default, but will auto-adjust if needed
    legend = ax.legend(loc='upper left', frameon=True, fontsize=10, 
                      fancybox=False, shadow=False, framealpha=0.9,
                      edgecolor='gray', facecolor='white')
    # Make legend text bold for better readability
    for text in legend.get_texts():
        text.set_fontweight('bold')
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


# ============================================================================
# STACKED TIME SERIES - TECHNOLOGY CONTRIBUTIONS
# ============================================================================

def create_technology_stacked_chart(
    simulation,
    results_df: pd.DataFrame,
    material: str,
    scenario: str,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 7)
) -> plt.Figure:
    """
    Create stacked time series showing technology contributions to material demand.
    
    Design:
    - Stacked areas for each technology
    - Black dotted lines for uncertainty (p2.5, p97.5 = 95% CI)
    - Black solid line for mean
    - Clean legend
    
    Parameters
    ----------
    simulation : MaterialsStockFlowSimulation
        Simulation object (needed to access stock-flow states)
    results_df : pd.DataFrame
        Results with scenario, year, material, mean, p2, p97
    material : str
        Material to plot
    scenario : str
        Scenario to plot
    
    Returns
    -------
    fig : matplotlib.Figure
    """
    # Get data for this scenario and material
    data = results_df[
        (results_df['scenario'] == scenario) & 
        (results_df['material'] == material)
    ].sort_values('year')
    
    if len(data) == 0:
        raise ValueError(f"No data for {scenario}, {material}")
    
    # Get technology breakdown (we need to calculate this from simulation)
    # This requires storing technology-level results - for now, we'll create
    # a simplified version
    
    fig, ax = setup_figure(figsize=figsize)
    
    years = data['year'].values
    
    # Plot uncertainty bands with black dotted lines (95% CI: p2.5 to p97.5)
    ax.plot(years, data['p97'], 'k--', linewidth=1.0, alpha=0.7, label='97.5th %ile')
    ax.plot(years, data['p2'], 'k--', linewidth=1.0, alpha=0.7, label='2.5th %ile')

    # Plot mean with black solid line
    ax.plot(years, data['mean'], 'k-', linewidth=2.0, label='Mean')

    # Fill between for visual clarity (optional)
    ax.fill_between(years, data['p2'], data['p97'],
                    color='black', alpha=0.1, linewidth=0)
    
    # Formatting
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{material} Demand (mt)', fontsize=12, fontweight='bold')
    ax.set_title(f'{material} Demand - {scenario.replace("_", " ")}', 
                fontsize=14, fontweight='bold', pad=15)
    
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(format_large_number))
    
    ax.legend(loc='best', frameon=False, fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


# ============================================================================
# MATERIAL COMPARISON - KEY MATERIALS
# ============================================================================

def create_material_comparison(
    results_df: pd.DataFrame,
    year: int,
    scenario: str,
    top_n: int = 10,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 6)
) -> plt.Figure:
    """
    Create bar chart comparing demand for top N materials in a given year/scenario.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results with scenario, year, material, mean, p2, p97
    year : int
        Year to plot
    scenario : str
        Scenario to plot
    top_n : int
        Number of top materials to show
    
    Returns
    -------
    fig : matplotlib.Figure
    """
    # Filter data
    data = results_df[
        (results_df['scenario'] == scenario) & 
        (results_df['year'] == year)
    ].copy()
    
    # Sort by mean and take top N
    data = data.nlargest(top_n, 'mean')
    
    fig, ax = setup_figure(figsize=figsize)
    
    # Create bar chart
    x_pos = np.arange(len(data))
    bars = ax.bar(x_pos, data['mean'], color='#2E86AB', alpha=0.8, edgecolor='black')
    
    # Add error bars (p2.5 to p97.5 range = 95% CI)
    # Use absolute values and clip to avoid negative/invalid error bars for skewed distributions
    yerr_lower = np.abs(data['mean'].values - data['p2'].values)
    yerr_upper = np.abs(data['p97'].values - data['mean'].values)
    
    # Replace any NaN or Inf with 0 (no error bar for that point)
    yerr_lower = np.nan_to_num(yerr_lower, nan=0.0, posinf=0.0, neginf=0.0)
    yerr_upper = np.nan_to_num(yerr_upper, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Only add error bars if we have valid values
    if np.any(yerr_lower > 0) or np.any(yerr_upper > 0):
        ax.errorbar(x_pos, data['mean'], 
                   yerr=[yerr_lower, yerr_upper],
                   fmt='none', ecolor='black', capsize=4, linewidth=1.5, alpha=0.7)
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(data['material'], rotation=45, ha='right')
    ax.set_ylabel('Demand (mt)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Materials - {scenario.replace("_", " ")} ({year})',
                fontsize=14, fontweight='bold', pad=15)
    
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(format_large_number))
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


# ============================================================================
# SCENARIO COMPARISON - SPECIFIC MATERIAL
# ============================================================================

def create_scenario_comparison(
    results_df: pd.DataFrame,
    material: str,
    year: int,
    scenarios: List[str] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 6)
) -> plt.Figure:
    """
    Create bar chart comparing scenarios for a specific material and year.
    
    Useful for comparing:
    - Mid_Case vs Mid_Case_No_IRA (IRA impact)
    - Mid_Case vs Mid_Case_100by2035 (Net-zero impact)
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe
    material : str
        Material to compare
    year : int
        Year to compare
    scenarios : List[str], optional
        Scenarios to include (default: 4 highlighted scenarios)
    
    Returns
    -------
    fig : matplotlib.Figure
    """
    if scenarios is None:
        scenarios = list(HIGHLIGHT_COLORS.keys())
    
    # Filter data
    data = results_df[
        (results_df['material'] == material) & 
        (results_df['year'] == year) &
        (results_df['scenario'].isin(scenarios))
    ].copy()
    
    fig, ax = setup_figure(figsize=figsize)
    
    # Sort by mean demand
    data = data.sort_values('mean', ascending=True)
    
    # Create horizontal bar chart (easier to read scenario names)
    y_pos = np.arange(len(data))
    colors = [HIGHLIGHT_COLORS.get(s, '#2E86AB') for s in data['scenario']]
    
    bars = ax.barh(y_pos, data['mean'], color=colors, alpha=0.8, edgecolor='black')
    
    # Add error bars (p2.5 to p97.5 range = 95% CI)
    # Use absolute values and clip to avoid negative/invalid error bars for skewed distributions
    xerr_lower = np.abs(data['mean'].values - data['p2'].values)
    xerr_upper = np.abs(data['p97'].values - data['mean'].values)
    
    # Replace any NaN or Inf with 0 (no error bar for that point)
    xerr_lower = np.nan_to_num(xerr_lower, nan=0.0, posinf=0.0, neginf=0.0)
    xerr_upper = np.nan_to_num(xerr_upper, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Only add error bars if we have valid values
    if np.any(xerr_lower > 0) or np.any(xerr_upper > 0):
        ax.errorbar(data['mean'], y_pos,
                   xerr=[xerr_lower, xerr_upper],
                   fmt='none', ecolor='black', capsize=4, linewidth=1.5, alpha=0.7)
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s.replace('_', ' ') for s in data['scenario']])
    ax.set_xlabel('Demand (mt)', fontsize=12, fontweight='bold')
    ax.set_title(f'{material} Demand Comparison ({year})',
                fontsize=14, fontweight='bold', pad=15)
    
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(format_large_number))
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


# ============================================================================
# MAIN VISUALIZATION SUITE
# ============================================================================

class MaterialsDemandVisualizer:
    """
    Complete visualization suite for materials demand analysis.
    
    Creates all publication-ready figures from simulation results.
    """
    
    def __init__(
        self,
        results_detailed: pd.DataFrame,
        results_summary: Optional[pd.DataFrame] = None,
        output_dir: Union[str, Path] = 'figures'
    ):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        results_detailed : pd.DataFrame
            Detailed results from simulation (by scenario)
        results_summary : pd.DataFrame, optional
            Summary results (aggregated across scenarios)
        output_dir : str or Path
            Directory for saving figures
        """
        self.results = results_detailed
        self.summary = results_summary
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Visualizer initialized")
        print(f"  Results: {len(self.results):,} rows")
        print(f"  Output: {self.output_dir}")
    
    def create_all_spaghetti_plots(
        self,
        materials: Optional[List[str]] = None,
        highlight_scenarios: List[str] = None
    ):
        """Create spaghetti plots for all key materials"""
        if materials is None:
            # Get top 10 materials by median demand
            top_materials = self.results.groupby('material')['p50'].median().nlargest(10).index.tolist()
            materials = top_materials
        
        if highlight_scenarios is None:
            highlight_scenarios = list(HIGHLIGHT_COLORS.keys())
        
        print(f"\nCreating spaghetti plots for {len(materials)} materials...")
        
        for material in materials:
            try:
                output_path = self.output_dir / f'spaghetti_{material.lower().replace(" ", "_")}'
                fig = create_spaghetti_plot(
                    self.results,
                    material=material,
                    highlight_scenarios=highlight_scenarios,
                    output_path=output_path
                )
                plt.close(fig)
            except Exception as e:
                print(f"  ✗ Failed for {material}: {e}")
    
    def create_scenario_comparisons(
        self,
        materials: List[str],
        year: int = 2035
    ):
        """Create scenario comparison charts for key materials"""
        print(f"\nCreating scenario comparisons for {len(materials)} materials ({year})...")
        
        for material in materials:
            try:
                output_path = self.output_dir / f'scenario_comp_{material.lower().replace(" ", "_")}_{year}'
                fig = create_scenario_comparison(
                    self.results,
                    material=material,
                    year=year,
                    output_path=output_path
                )
                plt.close(fig)
            except Exception as e:
                print(f"  ✗ Failed for {material}: {e}")
    
    def create_material_rankings(
        self,
        scenario: str = 'Mid_Case',
        year: int = 2035,
        top_n: int = 15
    ):
        """Create material ranking chart"""
        print(f"\nCreating material ranking chart...")
        
        output_path = self.output_dir / f'material_ranking_{scenario}_{year}'
        fig = create_material_comparison(
            self.results,
            year=year,
            scenario=scenario,
            top_n=top_n,
            output_path=output_path
        )
        plt.close(fig)
    
    def generate_figure_suite(
        self,
        key_materials: List[str] = ['Copper', 'Aluminum', 'Steel', 'Lithium', 'Silicon'],
        key_year: int = 2035
    ):
        """
        Generate complete suite of publication figures.
        
        Creates:
        1. Spaghetti plots for key materials
        2. Scenario comparisons for key materials
        3. Material rankings
        
        Parameters
        ----------
        key_materials : List[str]
            Materials to highlight in detailed analysis
        key_year : int
            Year to use for snapshot comparisons
        """
        print("="*80)
        print("GENERATING PUBLICATION FIGURE SUITE")
        print("="*80)
        
        # 1. Spaghetti plots
        self.create_all_spaghetti_plots(materials=key_materials)
        
        # 2. Scenario comparisons
        self.create_scenario_comparisons(materials=key_materials, year=key_year)
        
        # 3. Material rankings
        self.create_material_rankings(scenario='Mid_Case', year=key_year, top_n=15)
        self.create_material_rankings(scenario='Mid_Case_100by2035', year=key_year, top_n=15)
        
        print("\n" + "="*80)
        print("FIGURE GENERATION COMPLETE")
        print("="*80)
        print(f"All figures saved to: {self.output_dir}")
        print("\nFigure types created:")
        print("  - Spaghetti plots (all scenarios with highlights)")
        print("  - Scenario comparisons (bar charts)")
        print("  - Material rankings (top materials)")
        print("\nFormats: PNG and PDF")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    print("This is a module for creating publication-ready visualizations.")
    print("\nExample usage:")
    print("""
    from materials_visualizations import MaterialsDemandVisualizer
    import pandas as pd
    
    # Load results
    results = pd.read_csv('material_demand_by_scenario.csv')
    
    # Create visualizer
    viz = MaterialsDemandVisualizer(results, output_dir='figures')
    
    # Generate complete figure suite
    viz.generate_figure_suite(
        key_materials=['Copper', 'Aluminum', 'Steel', 'Lithium'],
        key_year=2035
    )
    """)
