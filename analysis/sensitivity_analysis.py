"""
Sensitivity Analysis for Materials Demand Model
================================================

This module implements sensitivity analysis to understand:
1. Variance decomposition: How much uncertainty comes from material intensity
   (MC simulation) vs. scenario assumptions?
2. Intensity elasticity: Which material-technology intensity parameters have
   the largest impact on demand?
3. Spearman correlation analysis: Rank correlation between sampled intensity
   values and resulting demand across Monte Carlo iterations.

Author: Materials Demand Pipeline
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# =============================================================================
# VARIANCE DECOMPOSITION
# =============================================================================

def compute_variance_decomposition(demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose total demand variance into within-scenario (MC intensity uncertainty)
    and between-scenario (scenario assumption) components.

    Uses the law of total variance:
        Var(Y) = E[Var(Y|X)] + Var(E[Y|X])

    Where:
        - Y = demand
        - X = scenario
        - E[Var(Y|X)] = mean of within-scenario variances (intensity uncertainty)
        - Var(E[Y|X]) = variance of scenario means (scenario uncertainty)

    Parameters
    ----------
    demand_df : pd.DataFrame
        Demand data with columns: scenario, year, material, mean, std

    Returns
    -------
    pd.DataFrame
        Variance decomposition by material with columns:
        - material
        - within_var: Mean within-scenario variance (intensity uncertainty)
        - between_var: Between-scenario variance (scenario uncertainty)
        - total_var: Total variance
        - within_pct: Percentage from intensity uncertainty
        - between_pct: Percentage from scenario uncertainty
    """
    results = []

    for material in demand_df['material'].unique():
        mat_data = demand_df[demand_df['material'] == material]

        # Skip materials with no demand
        if mat_data['mean'].sum() == 0:
            continue

        # Within-scenario variance: mean of MC variances
        # std² gives variance for each scenario-year combination
        within_vars = mat_data['std'] ** 2
        mean_within_var = within_vars.mean()

        # Between-scenario variance: variance of scenario means
        # Group by scenario, sum across years, then compute variance across scenarios
        scenario_totals = mat_data.groupby('scenario')['mean'].sum()
        between_var = scenario_totals.var()

        # For annual comparison (not cumulative)
        scenario_annual_means = mat_data.groupby('scenario')['mean'].mean()
        between_var_annual = scenario_annual_means.var()

        # Total variance (using law of total variance approximation)
        # Note: This is approximate because we're aggregating differently
        total_var = mean_within_var + between_var_annual

        # Percentages
        if total_var > 0:
            within_pct = 100 * mean_within_var / total_var
            between_pct = 100 * between_var_annual / total_var
        else:
            within_pct = 0
            between_pct = 0

        results.append({
            'material': material,
            'within_var': mean_within_var,
            'between_var': between_var_annual,
            'total_var': total_var,
            'within_pct': within_pct,
            'between_pct': between_pct,
            'cumulative_between_var': between_var,
            'mean_demand': mat_data['mean'].mean(),
        })

    df = pd.DataFrame(results)
    df = df.sort_values('mean_demand', ascending=False).reset_index(drop=True)

    return df


def compute_variance_decomposition_by_year(demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute variance decomposition for each material-year combination.

    Returns DataFrame with columns:
        material, year, within_var, between_var, total_var, within_pct, between_pct
    """
    results = []

    for (material, year), group in demand_df.groupby(['material', 'year']):
        if group['mean'].sum() == 0:
            continue

        # Within-scenario variance: mean of MC variances for this year
        within_var = (group['std'] ** 2).mean()

        # Between-scenario variance: variance across scenarios for this year
        between_var = group['mean'].var()

        total_var = within_var + between_var

        if total_var > 0:
            within_pct = 100 * within_var / total_var
            between_pct = 100 * between_var / total_var
        else:
            within_pct = 0
            between_pct = 0

        results.append({
            'material': material,
            'year': year,
            'within_var': within_var,
            'between_var': between_var,
            'total_var': total_var,
            'within_pct': within_pct,
            'between_pct': between_pct,
            'mean_demand': group['mean'].mean(),
        })

    return pd.DataFrame(results)


# =============================================================================
# INTENSITY ELASTICITY ANALYSIS
# =============================================================================

def compute_intensity_elasticity(
    tech_contributions: pd.DataFrame,
    demand_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute elasticity of total demand with respect to each material-technology
    intensity parameter.

    Elasticity ε = (∂D/D) / (∂I/I) = contribution_share

    Since Demand = Σ(Capacity_tech × Intensity_tech × Weight_tech),
    the elasticity of total demand with respect to intensity_i is equal to
    the normalized share of demand coming from that technology.

    Parameters
    ----------
    tech_contributions : pd.DataFrame
        Technology contributions with columns: material, technology,
        expected_demand_tonnes, pct_contribution
    demand_df : pd.DataFrame
        Demand data for computing total demand

    Returns
    -------
    pd.DataFrame
        Elasticity for each material-technology pair
    """
    # Use pct_contribution directly but normalize to ensure each material sums to 1
    # (handles any double-counting in the original data)

    results = []

    # Normalize pct_contribution per material
    for material in tech_contributions['material'].unique():
        mat_data = tech_contributions[tech_contributions['material'] == material].copy()

        # Sum of contributions for this material
        total_pct = mat_data['pct_contribution'].sum()

        for _, row in mat_data.iterrows():
            # Normalized elasticity (contribution share, sums to 1 per material)
            if total_pct > 0:
                elasticity = row['pct_contribution'] / total_pct
            else:
                elasticity = 0

            results.append({
                'material': material,
                'technology': row['technology'],
                'intensity_t_per_mw': row['intensity_t_per_mw'],
                'expected_demand_tonnes': row['expected_demand_tonnes'],
                'pct_contribution_raw': row['pct_contribution'],
                'elasticity': elasticity,
            })

    df = pd.DataFrame(results)
    df = df.sort_values('elasticity', ascending=False).reset_index(drop=True)

    return df


def compute_demand_sensitivity_to_intensity(
    demand_df: pd.DataFrame,
    tech_contributions: pd.DataFrame,
    perturbation_pct: float = 10.0
) -> pd.DataFrame:
    """
    Compute how total demand changes with ±X% perturbation to each intensity.

    Parameters
    ----------
    demand_df : pd.DataFrame
        Demand data
    tech_contributions : pd.DataFrame
        Technology contributions
    perturbation_pct : float
        Percentage perturbation (default 10%)

    Returns
    -------
    pd.DataFrame
        Sensitivity metrics for each material-technology pair
    """
    # Compute elasticities first
    elasticities = compute_intensity_elasticity(tech_contributions, demand_df)

    # Add sensitivity columns
    # A 10% increase in intensity → (elasticity × 10)% increase in demand
    elasticities['demand_change_pct'] = elasticities['elasticity'] * perturbation_pct

    return elasticities


# =============================================================================
# SPEARMAN CORRELATION SENSITIVITY ANALYSIS
# =============================================================================

def compute_spearman_sensitivity(
    fitted_distributions: Dict[Tuple[str, str], Any],
    capacity_data: pd.DataFrame,
    technology_mapping_func: callable,
    n_iterations: int = 1000,
    years: Optional[List[int]] = None,
    scenarios: Optional[List[str]] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Spearman rank correlation between sampled intensity parameters
    and resulting material demand across Monte Carlo iterations.

    This is a rigorous sensitivity measure that captures:
    - Non-linear relationships between inputs and outputs
    - Rank-based importance (robust to outliers)
    - Direction of influence (positive/negative correlation)

    Parameters
    ----------
    fitted_distributions : dict
        {(technology, material): MaterialIntensityDistribution}
        Fitted distributions from the main simulation
    capacity_data : pd.DataFrame
        Capacity projections with columns: scenario, year, technology, capacity_MW
    technology_mapping_func : callable
        Function that maps capacity technology to intensity technologies
        (e.g., get_intensity_technologies)
    n_iterations : int
        Number of Monte Carlo samples for sensitivity analysis (default 1000)
    years : list, optional
        Years to include (default: all)
    scenarios : list, optional
        Scenarios to include (default: all)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    spearman_results : pd.DataFrame
        Spearman correlations for each (technology, material, target_material) with:
        - spearman_rho: Spearman rank correlation coefficient
        - p_value: Statistical significance
        - abs_rho: Absolute value of rho (for ranking)
        - significant: Whether p < 0.05
        - direction: 'positive' or 'negative'

    iteration_data : pd.DataFrame
        Raw iteration data with sampled intensities and resulting demands
        (for additional analysis/debugging)

    Notes
    -----
    Interpretation of Spearman's rho:
    - |rho| > 0.7: Strong correlation (highly influential parameter)
    - |rho| 0.4-0.7: Moderate correlation (moderately influential)
    - |rho| < 0.4: Weak correlation (less influential)

    A positive rho means higher intensity → higher demand (expected).
    All intensity-demand correlations should be positive unless there's
    an error in the model.
    """
    np.random.seed(random_state)

    # Get unique parameters
    param_keys = list(fitted_distributions.keys())  # (tech, material) pairs
    materials = sorted(set(mat for _, mat in param_keys))

    # Filter capacity data
    if scenarios is not None:
        capacity_data = capacity_data[capacity_data['scenario'].isin(scenarios)]
    if years is not None:
        capacity_data = capacity_data[capacity_data['year'].isin(years)]

    # Get unique scenarios and years
    all_scenarios = capacity_data['scenario'].unique().tolist() if scenarios is None else scenarios
    all_years = sorted(capacity_data['year'].unique().tolist()) if years is None else years

    # Pre-compute capacity additions by (scenario, year, cap_technology)
    # Group by scenario, year, technology and get capacity
    capacity_additions = {}
    for _, row in capacity_data.iterrows():
        key = (row['scenario'], row['year'], row['technology'])
        # Assuming we have year-over-year additions or can compute from cumulative
        # For now, we'll use capacity values directly (will need adjustment based on actual data structure)
        capacity_additions[key] = row.get('additions_MW', row.get('capacity_MW', 0))

    logger.info(f"Running Spearman sensitivity analysis with {n_iterations} iterations")
    logger.info(f"  Parameters: {len(param_keys)}")
    logger.info(f"  Scenarios: {len(all_scenarios)}")
    logger.info(f"  Years: {len(all_years)}")

    # Storage for iteration data
    # Columns: iteration, (tech, material) intensity values, material demand values
    intensity_cols = [f"I_{tech}_{mat}" for tech, mat in param_keys]
    demand_cols = [f"D_{mat}" for mat in materials]

    iteration_records = []

    for iteration in range(n_iterations):
        if iteration % 200 == 0:
            logger.info(f"  Iteration {iteration}/{n_iterations}")

        # Sample all intensities for this iteration
        sampled_intensities = {}
        record = {'iteration': iteration}

        for (tech, mat), dist_info in fitted_distributions.items():
            intensity = dist_info.sample(n=1, random_state=None)[0]
            sampled_intensities[(tech, mat)] = intensity
            record[f"I_{tech}_{mat}"] = intensity

        # Calculate total demand per material (aggregated across scenarios, years, technologies)
        material_demand = {mat: 0.0 for mat in materials}

        for scenario in all_scenarios:
            for year in all_years:
                # Get unique capacity technologies from capacity data
                cap_techs = capacity_data[
                    (capacity_data['scenario'] == scenario) &
                    (capacity_data['year'] == year)
                ]['technology'].unique()

                for cap_tech in cap_techs:
                    # Get capacity for this (scenario, year, tech)
                    key = (scenario, year, cap_tech)
                    capacity = capacity_additions.get(key, 0)

                    if capacity <= 0:
                        continue

                    # Map to intensity technologies
                    try:
                        intensity_mappings = technology_mapping_func(cap_tech)
                    except:
                        continue

                    for intensity_tech, weight in intensity_mappings.items():
                        for material in materials:
                            if (intensity_tech, material) in sampled_intensities:
                                intensity = sampled_intensities[(intensity_tech, material)]
                                demand = capacity * weight * intensity
                                material_demand[material] += demand

        # Store demand values
        for mat in materials:
            record[f"D_{mat}"] = material_demand[mat]

        iteration_records.append(record)

    # Convert to DataFrame
    iteration_df = pd.DataFrame(iteration_records)

    logger.info("Computing Spearman correlations...")

    # Compute Spearman correlations: each intensity parameter vs each material demand
    spearman_results = []

    for (tech, mat) in param_keys:
        intensity_col = f"I_{tech}_{mat}"

        # Correlate with the same material's demand (primary)
        demand_col = f"D_{mat}"

        if intensity_col in iteration_df.columns and demand_col in iteration_df.columns:
            intensity_vals = iteration_df[intensity_col].values
            demand_vals = iteration_df[demand_col].values

            # Check for zero variance
            if np.std(intensity_vals) > 0 and np.std(demand_vals) > 0:
                rho, p_value = stats.spearmanr(intensity_vals, demand_vals)
            else:
                rho, p_value = 0.0, 1.0

            spearman_results.append({
                'technology': tech,
                'material': mat,
                'target_material': mat,  # Same material
                'spearman_rho': rho,
                'p_value': p_value,
                'abs_rho': abs(rho),
                'significant': p_value < 0.05,
                'direction': 'positive' if rho > 0 else 'negative' if rho < 0 else 'none',
                'interpretation': _interpret_spearman(rho, p_value),
            })

    spearman_df = pd.DataFrame(spearman_results)
    spearman_df = spearman_df.sort_values('abs_rho', ascending=False).reset_index(drop=True)

    logger.info(f"Spearman analysis complete: {len(spearman_df)} parameter correlations computed")

    return spearman_df, iteration_df


def _interpret_spearman(rho: float, p_value: float) -> str:
    """Interpret Spearman correlation coefficient."""
    if p_value >= 0.05:
        return "not significant"

    abs_rho = abs(rho)
    if abs_rho >= 0.7:
        strength = "strong"
    elif abs_rho >= 0.4:
        strength = "moderate"
    elif abs_rho >= 0.2:
        strength = "weak"
    else:
        return "negligible"

    direction = "positive" if rho > 0 else "negative"
    return f"{strength} {direction}"


def compute_spearman_sensitivity_from_simulation(
    simulation,
    n_iterations: int = 1000,
    scenarios_to_run: Optional[List[str]] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to run Spearman sensitivity analysis using an existing
    MaterialsStockFlowSimulation object.

    This is the recommended way to run Spearman analysis as it uses the exact
    same distributions and capacity data as the main simulation.

    Parameters
    ----------
    simulation : MaterialsStockFlowSimulation
        Configured simulation object with fitted distributions
    n_iterations : int
        Number of MC iterations for sensitivity analysis
    scenarios_to_run : list, optional
        Subset of scenarios to analyze
    random_state : int
        Random seed

    Returns
    -------
    spearman_df : pd.DataFrame
        Spearman correlation results
    iteration_df : pd.DataFrame
        Raw iteration data
    """
    # Import from parent directory - handle different execution contexts
    base_dir = Path(__file__).resolve().parent.parent
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    try:
        from src.technology_mapping import get_intensity_technologies
    except ImportError:
        sys.path.insert(0, str(base_dir / "src"))
        from technology_mapping import get_intensity_technologies

    # Get capacity data from simulation
    capacity_df = simulation.capacity_data

    # Compute additions from cumulative capacity
    # Group by scenario, technology and compute year-over-year differences
    capacity_df = capacity_df.sort_values(['scenario', 'technology', 'year'])
    capacity_df['additions_MW'] = capacity_df.groupby(['scenario', 'technology'])['capacity_MW'].diff().fillna(0)
    capacity_df['additions_MW'] = capacity_df['additions_MW'].clip(lower=0)  # Only positive additions

    return compute_spearman_sensitivity(
        fitted_distributions=simulation.fitted_distributions,
        capacity_data=capacity_df,
        technology_mapping_func=get_intensity_technologies,
        n_iterations=n_iterations,
        years=simulation.years,
        scenarios=scenarios_to_run,
        random_state=random_state
    )


def run_spearman_from_files(
    intensity_path: str,
    capacity_path: str,
    output_dir: str,
    n_iterations: int = 1000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Spearman sensitivity analysis by loading data from files.

    This function replicates the distribution fitting and capacity loading
    from the main simulation to compute Spearman correlations.

    Parameters
    ----------
    intensity_path : str
        Path to intensity_data.csv
    capacity_path : str
        Path to StdScen24_annual_national.csv
    output_dir : str
        Directory to save results
    n_iterations : int
        Number of MC iterations
    random_state : int
        Random seed

    Returns
    -------
    spearman_df : pd.DataFrame
        Spearman correlation results
    iteration_df : pd.DataFrame
        Raw iteration data
    """
    # Import from parent directory - handle different execution contexts
    base_dir = Path(__file__).resolve().parent.parent
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    try:
        from src.data_ingestion import MaterialIntensityLoader, CapacityProjectionLoader
        from src.distribution_fitting import DistributionFitter
        from src.technology_mapping import get_intensity_technologies
    except ImportError:
        # Try alternative import path
        sys.path.insert(0, str(base_dir / "src"))
        from data_ingestion import MaterialIntensityLoader, CapacityProjectionLoader
        from distribution_fitting import DistributionFitter
        from technology_mapping import get_intensity_technologies

    logger.info("Loading data for Spearman sensitivity analysis...")

    # Load intensity data and fit distributions
    intensity_loader = MaterialIntensityLoader()
    intensity_df = intensity_loader.load(intensity_path)

    # Set random seed for reproducibility
    np.random.seed(random_state)

    fitter = DistributionFitter()
    fitted_distributions = fitter.fit_all(intensity_df)

    # Load capacity data (wide format with columns like upv_MW, wind_onshore_MW, etc.)
    capacity_loader = CapacityProjectionLoader()
    capacity_wide = capacity_loader.load(capacity_path)

    # Melt to long format: scenario, year, technology, capacity_MW
    tech_cols = [c for c in capacity_wide.columns if c.endswith('_MW')]
    capacity_df = capacity_wide.melt(
        id_vars=['scenario', 'year'],
        value_vars=tech_cols,
        var_name='technology',
        value_name='capacity_MW'
    )
    # Remove _MW suffix from technology names
    capacity_df['technology'] = capacity_df['technology'].str.replace('_MW', '', regex=False)

    # Compute additions (year-over-year differences)
    capacity_df = capacity_df.sort_values(['scenario', 'technology', 'year'])
    capacity_df['additions_MW'] = capacity_df.groupby(['scenario', 'technology'])['capacity_MW'].diff().fillna(0)
    capacity_df['additions_MW'] = capacity_df['additions_MW'].clip(lower=0)

    # Get years
    years = sorted(capacity_df['year'].unique().tolist())

    logger.info(f"Fitted {len(fitted_distributions)} distributions")
    logger.info(f"Capacity data: {len(capacity_df)} rows, {len(capacity_df['scenario'].unique())} scenarios")
    logger.info(f"Technologies: {capacity_df['technology'].nunique()}")

    # Run Spearman analysis
    spearman_df, iteration_df = compute_spearman_sensitivity(
        fitted_distributions=fitted_distributions,
        capacity_data=capacity_df,
        technology_mapping_func=get_intensity_technologies,
        n_iterations=n_iterations,
        years=years,
        random_state=random_state
    )

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    spearman_df.to_csv(output_path / "spearman_sensitivity.csv", index=False)
    iteration_df.to_csv(output_path / "spearman_iteration_data.csv", index=False)

    logger.info(f"Saved Spearman results to {output_path}")

    return spearman_df, iteration_df


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_variance_decomposition(
    var_decomp: pd.DataFrame,
    output_path: Optional[Path] = None,
    top_n: int = 15,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create stacked bar chart showing variance decomposition by material.

    Parameters
    ----------
    var_decomp : pd.DataFrame
        Output from compute_variance_decomposition()
    output_path : Path, optional
        Save figure to this path
    top_n : int
        Number of top materials to show
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
    """
    # Select top N materials by total demand
    data = var_decomp.head(top_n).copy()

    fig, ax = plt.subplots(figsize=figsize)

    # Create stacked horizontal bar chart
    materials = data['material']
    y_pos = np.arange(len(materials))

    # Plot within (intensity) and between (scenario) variance percentages
    bars1 = ax.barh(y_pos, data['within_pct'], label='Intensity Uncertainty (MC)',
                    color='#2E86AB', alpha=0.8)
    bars2 = ax.barh(y_pos, data['between_pct'], left=data['within_pct'],
                    label='Scenario Uncertainty', color='#A23B72', alpha=0.8)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(materials)
    ax.set_xlabel('Percentage of Total Variance (%)', fontsize=12)
    ax.set_title('Variance Decomposition: Intensity vs. Scenario Uncertainty\n'
                 '(Top {} Materials by Demand)'.format(top_n), fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, 100)

    # Add percentage labels
    for i, (w, b) in enumerate(zip(data['within_pct'], data['between_pct'])):
        if w > 5:
            ax.text(w/2, i, f'{w:.0f}%', ha='center', va='center',
                   fontsize=9, color='white', fontweight='bold')
        if b > 5:
            ax.text(w + b/2, i, f'{b:.0f}%', ha='center', va='center',
                   fontsize=9, color='white', fontweight='bold')

    # Add vertical line at 50%
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

    return fig


def plot_tornado_diagram(
    elasticities: pd.DataFrame,
    output_path: Optional[Path] = None,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Create tornado diagram showing sensitivity of demand to intensity parameters.

    Parameters
    ----------
    elasticities : pd.DataFrame
        Output from compute_intensity_elasticity()
    output_path : Path, optional
        Save figure to this path
    top_n : int
        Number of top parameters to show
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
    """
    # Select top N by elasticity
    data = elasticities.head(top_n).copy()

    fig, ax = plt.subplots(figsize=figsize)

    # Create labels: "Material - Technology"
    labels = [f"{row['material']} - {row['technology']}"
              for _, row in data.iterrows()]

    y_pos = np.arange(len(labels))

    # Plot horizontal bars
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(data)))
    bars = ax.barh(y_pos, data['elasticity'] * 100, color=colors, alpha=0.8)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Elasticity (% demand change per 1% intensity change)', fontsize=12)
    ax.set_title('Tornado Diagram: Demand Sensitivity to Material Intensity\n'
                 '(Top {} Most Influential Parameters)'.format(top_n),
                 fontsize=14, fontweight='bold')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, data['elasticity'])):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
               f'{val*100:.1f}%', ha='left', va='center', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(data['elasticity'] * 100) * 1.15)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

    return fig


def plot_variance_by_year(
    var_by_year: pd.DataFrame,
    materials: List[str],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Create line plot showing variance decomposition over time for selected materials.
    """
    n_materials = len(materials)
    n_cols = 3
    n_rows = (n_materials + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, material in enumerate(materials):
        ax = axes[idx]
        mat_data = var_by_year[var_by_year['material'] == material].sort_values('year')

        if len(mat_data) == 0:
            ax.text(0.5, 0.5, f'No data for {material}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(material, fontsize=11, fontweight='bold')
            continue

        years = mat_data['year']

        # Stacked area plot
        ax.fill_between(years, 0, mat_data['within_pct'],
                       label='Intensity', color='#2E86AB', alpha=0.7)
        ax.fill_between(years, mat_data['within_pct'], 100,
                       label='Scenario', color='#A23B72', alpha=0.7)

        ax.set_ylim(0, 100)
        ax.set_xlim(years.min(), years.max())
        ax.set_title(material, fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel('% of Variance', fontsize=9)

        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Hide unused subplots
    for idx in range(n_materials, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Variance Decomposition Over Time\n'
                 '(Blue = Intensity Uncertainty, Pink = Scenario Uncertainty)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

    return fig


def plot_spearman_tornado(
    spearman_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    top_n: int = 25,
    figsize: Tuple[int, int] = (12, 12)
) -> plt.Figure:
    """
    Create tornado diagram showing Spearman correlation coefficients.

    Parameters
    ----------
    spearman_df : pd.DataFrame
        Output from compute_spearman_sensitivity()
    output_path : Path, optional
        Save figure to this path
    top_n : int
        Number of top parameters to show
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
    """
    # Filter to significant results and select top N by absolute rho
    data = spearman_df[spearman_df['significant']].head(top_n).copy()

    if len(data) == 0:
        # If no significant results, show top N regardless
        data = spearman_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=figsize)

    # Create labels: "Material - Technology"
    labels = [f"{row['material']} ({row['technology']})" for _, row in data.iterrows()]

    y_pos = np.arange(len(labels))

    # Color by significance and direction
    colors = []
    for _, row in data.iterrows():
        if not row['significant']:
            colors.append('#808080')  # Gray for non-significant
        elif row['spearman_rho'] > 0:
            # Positive correlation: shades of blue based on strength
            intensity = min(1.0, row['abs_rho'] / 0.7)
            colors.append(plt.cm.Blues(0.4 + 0.5 * intensity))
        else:
            # Negative correlation: shades of red
            intensity = min(1.0, row['abs_rho'] / 0.7)
            colors.append(plt.cm.Reds(0.4 + 0.5 * intensity))

    bars = ax.barh(y_pos, data['spearman_rho'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Spearman's ρ (rank correlation)", fontsize=12)
    ax.set_title('Spearman Sensitivity Analysis: Intensity → Demand Correlation\n'
                 f'(Top {len(data)} Parameters by |ρ|)',
                 fontsize=14, fontweight='bold')

    # Add vertical line at 0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

    # Add reference lines for correlation strength
    for threshold, label in [(0.7, 'Strong'), (0.4, 'Moderate'), (-0.4, ''), (-0.7, '')]:
        ax.axvline(x=threshold, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        if label:
            ax.text(threshold, len(data) - 0.5, f'|ρ|={abs(threshold)}',
                   fontsize=8, color='gray', ha='center')

    # Add value labels
    for i, (bar, row) in enumerate(zip(bars, data.itertuples())):
        width = bar.get_width()
        offset = 0.02 if width >= 0 else -0.02
        ha = 'left' if width >= 0 else 'right'
        significance = '*' if row.significant else ''
        ax.text(width + offset, bar.get_y() + bar.get_height()/2,
               f'{row.spearman_rho:.3f}{significance}', ha=ha, va='center', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set symmetric x-axis
    max_abs = max(abs(data['spearman_rho'].min()), abs(data['spearman_rho'].max())) * 1.2
    ax.set_xlim(-max_abs, max_abs)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=plt.cm.Blues(0.7), edgecolor='black', label='Positive (significant)'),
        Patch(facecolor=plt.cm.Reds(0.7), edgecolor='black', label='Negative (significant)'),
        Patch(facecolor='#808080', edgecolor='black', label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

    return fig


def plot_spearman_heatmap(
    spearman_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Create heatmap of Spearman correlations by technology and material.

    Parameters
    ----------
    spearman_df : pd.DataFrame
        Output from compute_spearman_sensitivity()
    output_path : Path, optional
        Save figure to this path
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
    """
    # Pivot to create heatmap matrix
    pivot_df = spearman_df.pivot_table(
        index='material',
        columns='technology',
        values='spearman_rho',
        aggfunc='first'
    )

    # Sort materials by mean absolute correlation
    material_importance = spearman_df.groupby('material')['abs_rho'].mean().sort_values(ascending=False)
    pivot_df = pivot_df.reindex(material_importance.index)

    # Sort technologies by mean absolute correlation
    tech_importance = spearman_df.groupby('technology')['abs_rho'].mean().sort_values(ascending=False)
    pivot_df = pivot_df[tech_importance.index.intersection(pivot_df.columns)]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(pivot_df.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman's ρ", fontsize=11)

    # Set ticks
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(pivot_df.index, fontsize=9)

    # Add text annotations
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.iloc[i, j]
            if pd.notna(val):
                text_color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=7, color=text_color)

    ax.set_xlabel('Technology', fontsize=12)
    ax.set_ylabel('Material', fontsize=12)
    ax.set_title('Spearman Correlation Heatmap: Intensity → Demand\n'
                 '(Ranked by importance)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

    return fig


def plot_spearman_scatter(
    iteration_df: pd.DataFrame,
    technology: str,
    material: str,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Create scatter plot showing relationship between sampled intensity
    and resulting demand for a specific technology-material pair.

    Parameters
    ----------
    iteration_df : pd.DataFrame
        Iteration data from compute_spearman_sensitivity()
    technology : str
        Technology name
    material : str
        Material name
    output_path : Path, optional
        Save figure to this path
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
    """
    intensity_col = f"I_{technology}_{material}"
    demand_col = f"D_{material}"

    if intensity_col not in iteration_df.columns:
        raise ValueError(f"Intensity column {intensity_col} not found")
    if demand_col not in iteration_df.columns:
        raise ValueError(f"Demand column {demand_col} not found")

    x = iteration_df[intensity_col].values
    y = iteration_df[demand_col].values

    # Compute correlation
    rho, p_value = stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(x, y, alpha=0.3, s=10, c='steelblue')

    # Add trend line (linear regression for visual)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Linear trend')

    ax.set_xlabel(f'{material} Intensity ({technology}) [t/MW]', fontsize=11)
    ax.set_ylabel(f'{material} Demand [tonnes]', fontsize=11)
    ax.set_title(f'Spearman Sensitivity: {material} ({technology})\n'
                 f'ρ = {rho:.3f}, p = {p_value:.2e}',
                 fontsize=12, fontweight='bold')

    ax.legend(loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

    return fig


def plot_spearman_summary(
    spearman_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create multi-panel summary figure for Spearman sensitivity analysis.

    Includes:
    1. Top 15 parameters by |ρ|
    2. Distribution of correlation strengths
    3. Material-level summary (mean |ρ| by material)
    4. Technology-level summary (mean |ρ| by technology)

    Parameters
    ----------
    spearman_df : pd.DataFrame
        Output from compute_spearman_sensitivity()
    output_path : Path, optional
        Save figure to this path
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: Top 15 parameters (tornado)
    ax1 = axes[0, 0]
    top_data = spearman_df.head(15)
    labels = [f"{row['material']}\n({row['technology'][:8]})" for _, row in top_data.iterrows()]
    y_pos = np.arange(len(labels))
    colors = ['#2E86AB' if r > 0 else '#A23B72' for r in top_data['spearman_rho']]
    ax1.barh(y_pos, top_data['spearman_rho'], color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Spearman's ρ", fontsize=10)
    ax1.set_title('Top 15 Parameters by |ρ|', fontsize=11, fontweight='bold')
    ax1.axvline(x=0, color='black', linewidth=0.5)

    # Panel 2: Distribution of |ρ|
    ax2 = axes[0, 1]
    ax2.hist(spearman_df['abs_rho'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0.4, color='orange', linestyle='--', label='Moderate threshold')
    ax2.axvline(x=0.7, color='red', linestyle='--', label='Strong threshold')
    ax2.set_xlabel("|Spearman's ρ|", fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_title('Distribution of Correlation Strengths', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)

    # Panel 3: Material-level summary
    ax3 = axes[1, 0]
    mat_summary = spearman_df.groupby('material')['abs_rho'].agg(['mean', 'max', 'count']).reset_index()
    mat_summary = mat_summary.sort_values('mean', ascending=True).tail(15)
    y_pos = np.arange(len(mat_summary))
    ax3.barh(y_pos, mat_summary['mean'], color='#2E86AB', alpha=0.8, label='Mean |ρ|')
    ax3.scatter(mat_summary['max'], y_pos, color='red', marker='|', s=100, label='Max |ρ|')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(mat_summary['material'], fontsize=9)
    ax3.set_xlabel("Mean |Spearman's ρ|", fontsize=10)
    ax3.set_title('Sensitivity by Material (Top 15)', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8, loc='lower right')

    # Panel 4: Technology-level summary
    ax4 = axes[1, 1]
    tech_summary = spearman_df.groupby('technology')['abs_rho'].agg(['mean', 'max', 'count']).reset_index()
    tech_summary = tech_summary.sort_values('mean', ascending=True)
    y_pos = np.arange(len(tech_summary))
    ax4.barh(y_pos, tech_summary['mean'], color='#A23B72', alpha=0.8, label='Mean |ρ|')
    ax4.scatter(tech_summary['max'], y_pos, color='red', marker='|', s=100, label='Max |ρ|')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(tech_summary['technology'], fontsize=9)
    ax4.set_xlabel("Mean |Spearman's ρ|", fontsize=10)
    ax4.set_title('Sensitivity by Technology', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8, loc='lower right')

    # Overall title
    n_significant = spearman_df['significant'].sum()
    n_total = len(spearman_df)
    fig.suptitle(f'Spearman Sensitivity Analysis Summary\n'
                 f'{n_significant}/{n_total} parameters significant (p < 0.05)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

    return fig


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def generate_sensitivity_report(
    var_decomp: pd.DataFrame,
    elasticities: pd.DataFrame,
    spearman_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None
) -> str:
    """
    Generate a text summary of sensitivity analysis results.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("SENSITIVITY ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Overall variance decomposition summary
    lines.append("1. VARIANCE DECOMPOSITION SUMMARY")
    lines.append("-" * 80)

    # Weighted average (by total demand)
    total_demand = var_decomp['mean_demand'].sum()
    weighted_within = (var_decomp['within_pct'] * var_decomp['mean_demand']).sum() / total_demand
    weighted_between = (var_decomp['between_pct'] * var_decomp['mean_demand']).sum() / total_demand

    lines.append(f"Demand-weighted average variance decomposition:")
    lines.append(f"  - Intensity uncertainty (MC): {weighted_within:.1f}%")
    lines.append(f"  - Scenario uncertainty:       {weighted_between:.1f}%")
    lines.append("")

    # Materials dominated by intensity uncertainty
    intensity_dominated = var_decomp[var_decomp['within_pct'] > 50]['material'].tolist()
    scenario_dominated = var_decomp[var_decomp['between_pct'] > 50]['material'].tolist()

    lines.append(f"Materials dominated by intensity uncertainty (>50%):")
    lines.append(f"  {', '.join(intensity_dominated[:10])}")
    if len(intensity_dominated) > 10:
        lines.append(f"  ... and {len(intensity_dominated) - 10} more")
    lines.append("")

    lines.append(f"Materials dominated by scenario uncertainty (>50%):")
    lines.append(f"  {', '.join(scenario_dominated[:10])}")
    if len(scenario_dominated) > 10:
        lines.append(f"  ... and {len(scenario_dominated) - 10} more")
    lines.append("")

    # Top variance decomposition
    lines.append("Top 10 materials by demand - variance breakdown:")
    lines.append(f"{'Material':<15} {'Intensity %':>12} {'Scenario %':>12}")
    lines.append("-" * 40)
    for _, row in var_decomp.head(10).iterrows():
        lines.append(f"{row['material']:<15} {row['within_pct']:>12.1f} {row['between_pct']:>12.1f}")
    lines.append("")

    # Elasticity summary
    lines.append("")
    lines.append("2. INTENSITY ELASTICITY SUMMARY")
    lines.append("-" * 80)
    lines.append("Top 20 most influential intensity parameters:")
    lines.append(f"{'Material':<15} {'Technology':<25} {'Elasticity':>12}")
    lines.append("-" * 55)
    for _, row in elasticities.head(20).iterrows():
        lines.append(f"{row['material']:<15} {row['technology']:<25} {row['elasticity']*100:>11.2f}%")
    lines.append("")

    lines.append("Interpretation: A 1% increase in [Technology] intensity for [Material]")
    lines.append("                results in [Elasticity]% increase in total demand for that material.")
    lines.append("")

    # Spearman correlation summary
    if spearman_df is not None and len(spearman_df) > 0:
        lines.append("")
        lines.append("3. SPEARMAN CORRELATION SENSITIVITY ANALYSIS")
        lines.append("-" * 80)

        n_total = len(spearman_df)
        n_significant = spearman_df['significant'].sum()
        n_strong = (spearman_df['abs_rho'] >= 0.7).sum()
        n_moderate = ((spearman_df['abs_rho'] >= 0.4) & (spearman_df['abs_rho'] < 0.7)).sum()

        lines.append(f"Total parameters analyzed: {n_total}")
        lines.append(f"Significant correlations (p < 0.05): {n_significant} ({100*n_significant/n_total:.1f}%)")
        lines.append(f"Strong correlations (|ρ| ≥ 0.7): {n_strong}")
        lines.append(f"Moderate correlations (0.4 ≤ |ρ| < 0.7): {n_moderate}")
        lines.append("")

        lines.append("Top 15 parameters by Spearman correlation:")
        lines.append(f"{'Material':<15} {'Technology':<20} {'ρ':>8} {'p-value':>12} {'Interpretation':<20}")
        lines.append("-" * 78)
        for _, row in spearman_df.head(15).iterrows():
            sig = '*' if row['significant'] else ' '
            lines.append(f"{row['material']:<15} {row['technology'][:18]:<20} {row['spearman_rho']:>7.3f}{sig} "
                        f"{row['p_value']:>11.2e} {row['interpretation']:<20}")
        lines.append("")

        lines.append("Interpretation guide:")
        lines.append("  - ρ > 0: Higher intensity → higher demand (expected relationship)")
        lines.append("  - |ρ| ≥ 0.7: Strong influence - prioritize data quality improvement")
        lines.append("  - |ρ| 0.4-0.7: Moderate influence - important but not critical")
        lines.append("  - |ρ| < 0.4: Weak influence - refinement has limited impact")
        lines.append("  - * indicates p < 0.05 (statistically significant)")
        lines.append("")

        # Key findings from Spearman
        lines.append("")
        lines.append("4. KEY FINDINGS")
    else:
        # Key findings without Spearman
        lines.append("")
        lines.append("3. KEY FINDINGS")
    lines.append("-" * 80)

    if weighted_between > weighted_within:
        lines.append(f"* Scenario uncertainty ({weighted_between:.1f}%) dominates intensity uncertainty ({weighted_within:.1f}%)")
        lines.append("  → Scenario assumptions are the primary driver of demand variation")
        lines.append("  → Improving intensity data would have limited impact on overall uncertainty")
    else:
        lines.append(f"* Intensity uncertainty ({weighted_within:.1f}%) dominates scenario uncertainty ({weighted_between:.1f}%)")
        lines.append("  → Material intensity data quality is the primary uncertainty driver")
        lines.append("  → Improving intensity estimates would significantly reduce uncertainty")

    lines.append("")
    top_elastic = elasticities.head(3)
    lines.append("* Most influential intensity parameters (by elasticity):")
    for _, row in top_elastic.iterrows():
        lines.append(f"  - {row['material']}/{row['technology']}: {row['elasticity']*100:.1f}% elasticity")

    if spearman_df is not None and len(spearman_df) > 0:
        lines.append("")
        top_spearman = spearman_df[spearman_df['significant']].head(3)
        if len(top_spearman) > 0:
            lines.append("* Most influential parameters (by Spearman ρ):")
            for _, row in top_spearman.iterrows():
                lines.append(f"  - {row['material']}/{row['technology']}: ρ = {row['spearman_rho']:.3f}")

        # Agreement between methods
        top_elastic_set = set(
            f"{r['material']}/{r['technology']}" for _, r in elasticities.head(10).iterrows()
        )
        top_spearman_set = set(
            f"{r['material']}/{r['technology']}" for _, r in spearman_df.head(10).iterrows()
        )
        overlap = top_elastic_set.intersection(top_spearman_set)
        if len(overlap) > 0:
            lines.append("")
            lines.append(f"* Methods agreement: {len(overlap)}/10 top parameters appear in both elasticity and Spearman rankings")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Saved: {output_path}")

    return report


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run complete sensitivity analysis."""
    print("=" * 80)
    print("SENSITIVITY ANALYSIS FOR MATERIALS DEMAND MODEL")
    print("=" * 80)

    # Setup paths
    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / "outputs" / "data" / "sensitivity"
    figures_dir = base_dir / "outputs" / "figures" / "sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    demand_df = pd.read_csv(base_dir / "outputs" / "data" / "material_demand_by_scenario.csv")
    tech_contrib = pd.read_csv(base_dir / "outputs" / "data" / "sensitivity" / "technology_contributions.csv")

    print(f"  Demand data: {len(demand_df)} rows")
    print(f"  Technology contributions: {len(tech_contrib)} rows")

    # 1. Variance Decomposition
    print("\n" + "-" * 50)
    print("Computing variance decomposition...")
    print("-" * 50)

    var_decomp = compute_variance_decomposition(demand_df)
    var_decomp.to_csv(output_dir / "variance_decomposition.csv", index=False)
    print(f"  Saved: variance_decomposition.csv")

    var_by_year = compute_variance_decomposition_by_year(demand_df)
    var_by_year.to_csv(output_dir / "variance_decomposition_by_year.csv", index=False)
    print(f"  Saved: variance_decomposition_by_year.csv")

    # 2. Intensity Elasticity
    print("\n" + "-" * 50)
    print("Computing intensity elasticity...")
    print("-" * 50)

    elasticities = compute_intensity_elasticity(tech_contrib, demand_df)
    elasticities.to_csv(output_dir / "intensity_elasticity.csv", index=False)
    print(f"  Saved: intensity_elasticity.csv")

    # 3. Spearman Correlation Sensitivity Analysis
    print("\n" + "-" * 50)
    print("Computing Spearman correlation sensitivity...")
    print("-" * 50)

    spearman_df = None
    iteration_df = None

    try:
        # Try to run Spearman analysis (requires access to fitted distributions)
        intensity_path = base_dir / "data" / "intensity_data.csv"
        capacity_path = base_dir / "data" / "StdScen24_annual_national.csv"

        if intensity_path.exists() and capacity_path.exists():
            spearman_df, iteration_df = run_spearman_from_files(
                intensity_path=str(intensity_path),
                capacity_path=str(capacity_path),
                output_dir=str(output_dir),
                n_iterations=1000,  # Use 1000 iterations for faster analysis
                random_state=42
            )
            print(f"  Saved: spearman_sensitivity.csv")
            print(f"  Saved: spearman_iteration_data.csv")
            print(f"  Analyzed {len(spearman_df)} parameters")
            print(f"  Significant correlations: {spearman_df['significant'].sum()}")
        else:
            print(f"  Warning: Data files not found for Spearman analysis")
            print(f"    Intensity: {intensity_path.exists()}")
            print(f"    Capacity: {capacity_path.exists()}")
    except Exception as e:
        print(f"  Warning: Spearman analysis failed: {e}")
        print(f"  Continuing with other analyses...")

    # 4. Generate Visualizations
    print("\n" + "-" * 50)
    print("Generating visualizations...")
    print("-" * 50)

    plot_variance_decomposition(
        var_decomp,
        output_path=figures_dir / "variance_decomposition.png",
        top_n=15
    )

    plot_tornado_diagram(
        elasticities,
        output_path=figures_dir / "tornado_elasticity.png",
        top_n=20
    )

    # Variance over time for key materials
    key_materials = ['Copper', 'Steel', 'Aluminum', 'Silicon', 'Neodymium',
                     'Cement', 'Nickel', 'Chromium', 'Dysprosium']
    plot_variance_by_year(
        var_by_year,
        materials=key_materials,
        output_path=figures_dir / "variance_by_year.png"
    )

    # Spearman visualizations (if analysis succeeded)
    if spearman_df is not None:
        plot_spearman_tornado(
            spearman_df,
            output_path=figures_dir / "spearman_tornado.png",
            top_n=25
        )

        plot_spearman_heatmap(
            spearman_df,
            output_path=figures_dir / "spearman_heatmap.png"
        )

        plot_spearman_summary(
            spearman_df,
            output_path=figures_dir / "spearman_summary.png"
        )

    # 5. Generate Report
    print("\n" + "-" * 50)
    print("Generating summary report...")
    print("-" * 50)

    report = generate_sensitivity_report(
        var_decomp,
        elasticities,
        spearman_df=spearman_df,
        output_path=base_dir / "outputs" / "reports" / "sensitivity_analysis_report.txt"
    )
    print("\n" + report)

    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to:")
    print(f"  Data: {output_dir}")
    print(f"  Figures: {figures_dir}")
    print(f"  Report: outputs/reports/sensitivity_analysis_report.txt")


if __name__ == "__main__":
    main()
