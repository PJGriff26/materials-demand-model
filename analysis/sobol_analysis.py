"""
Sobol Sensitivity Analysis for Materials Demand Model
======================================================

Variance-based global sensitivity analysis using Sobol indices to decompose
output variance into contributions from individual input parameters.

Three analysis levels:
1. **Per-material individual Sobol**: For each output material, decompose
   demand variance across the contributing intensity parameters (technologies).
2. **Per-material grouped Sobol**: Group intensity parameters by technology
   sector (Solar, Wind, Nuclear, etc.) and compute sector-level indices.
3. **Global grouped Sobol**: Aggregate demand across critical materials and
   decompose variance by technology sector.

Key model property: Demand(m) = sum_i [a_i * I_i] — the model is linear in
intensity parameters.  Consequently S1 ~ ST (negligible interactions), which
validates the simpler Spearman and elasticity methods used elsewhere.

Uses SALib (Sensitivity Analysis Library) for Saltelli sampling and Sobol
index computation.

Author: Materials Demand Pipeline
Date: March 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys
import logging
import warnings

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# Check SALib availability
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    logger.warning(
        "SALib not installed. Install with: pip install SALib>=1.4.0  "
        "or: pip install -e .[sensitivity]"
    )


# =============================================================================
# TECHNOLOGY SECTOR GROUPING
# =============================================================================

# Maps intensity technology names to sector labels for grouped Sobol analysis
INTENSITY_TECH_TO_SECTOR = {
    # Solar
    'utility-scale solar pv': 'Solar',
    'Solar Distributed': 'Solar',
    'CSP': 'Solar',
    'CdTe': 'Solar',
    'CIGS': 'Solar',
    'a-Si': 'Solar',
    # Wind
    'onshore wind': 'Wind',
    'offshore wind': 'Wind',
    # Nuclear
    'Nuclear New': 'Nuclear',
    # Hydro
    'Hydro': 'Hydro',
    # Fossil / Gas
    'NGCC': 'Fossil',
    'NGGT': 'Fossil',
    'Gas CCS': 'Fossil',
    'Coal': 'Fossil',
    'Coal CCS': 'Fossil',
    # Biomass
    'Biomass': 'Biomass',
    'Bio CCS': 'Biomass',
    # Geothermal
    'Geothermal': 'Geothermal',
}


# =============================================================================
# CORE HELPER FUNCTIONS
# =============================================================================

def identify_contributing_parameters(
    fitted_distributions: Dict[Tuple[str, str], Any],
    target_material: str
) -> List[Tuple[str, str]]:
    """
    For a given target material, identify which (technology, material) intensity
    parameters contribute to its demand.

    Parameters
    ----------
    fitted_distributions : dict
        {(technology, material): MaterialIntensityDistribution}
    target_material : str
        Material whose demand we are analyzing

    Returns
    -------
    list of (technology, material) tuples
        Contributing parameters, sorted by technology name
    """
    params = [
        (tech, mat) for (tech, mat) in fitted_distributions
        if mat == target_material
    ]
    return sorted(params, key=lambda x: x[0])


def get_parameter_bounds(
    dist_info,
    coverage: float = 0.998
) -> Tuple[float, float]:
    """
    Get bounds for a single intensity parameter using the fitted distribution.

    Uses the distribution's percent-point function (ppf) to obtain bounds
    covering the specified probability mass.  Lower bound is clipped at 0
    (intensities are non-negative).

    Parameters
    ----------
    dist_info : MaterialIntensityDistribution
        Fitted distribution object
    coverage : float
        Probability coverage (default 0.998 → p0.1 to p99.9)

    Returns
    -------
    (lower, upper) : tuple of float
    """
    alpha = (1 - coverage) / 2  # 0.001 for coverage=0.998
    frozen_dist = dist_info.get_frozen_distribution()

    if frozen_dist is not None:
        try:
            lower = max(0.0, float(frozen_dist.ppf(alpha)))
            upper = float(frozen_dist.ppf(1 - alpha))
            if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
                return (lower, upper)
        except Exception:
            pass

    # Fallback: mean +/- 3*std, clipped at 0
    mean = float(dist_info.mean)
    std = float(dist_info.std)
    lower = max(0.0, mean - 3 * std)
    upper = mean + 3 * std
    if upper <= lower:
        upper = lower + 1e-10
    return (lower, upper)


def build_salib_problem(
    fitted_distributions: Dict[Tuple[str, str], Any],
    target_material: str,
    coverage: float = 0.998
) -> Tuple[dict, List[Tuple[str, str]]]:
    """
    Build a SALib problem dict for a single target material.

    Parameters
    ----------
    fitted_distributions : dict
        All fitted distributions
    target_material : str
        Material to analyze
    coverage : float
        Probability coverage for bounds

    Returns
    -------
    problem : dict
        SALib problem dict with 'num_vars', 'names', 'bounds'
    param_keys : list of (technology, material)
        Ordered parameter keys matching SALib variable indices
    """
    param_keys = identify_contributing_parameters(
        fitted_distributions, target_material
    )

    names = [tech for tech, _ in param_keys]
    bounds = [
        list(get_parameter_bounds(fitted_distributions[(tech, mat)], coverage))
        for tech, mat in param_keys
    ]

    problem = {
        'num_vars': len(param_keys),
        'names': names,
        'bounds': bounds,
    }
    return problem, param_keys


def build_salib_problem_grouped(
    fitted_distributions: Dict[Tuple[str, str], Any],
    target_material: str,
    coverage: float = 0.998
) -> Tuple[dict, List[Tuple[str, str]]]:
    """
    Build a SALib problem dict with technology-sector grouping.

    Parameters
    ----------
    fitted_distributions : dict
    target_material : str
    coverage : float

    Returns
    -------
    problem : dict
        SALib problem dict with 'num_vars', 'names', 'bounds', 'groups'
    param_keys : list of (technology, material)
    """
    problem, param_keys = build_salib_problem(
        fitted_distributions, target_material, coverage
    )

    groups = [
        INTENSITY_TECH_TO_SECTOR.get(tech, 'Other')
        for tech, _ in param_keys
    ]
    problem['groups'] = groups

    return problem, param_keys


def build_global_salib_problem(
    fitted_distributions: Dict[Tuple[str, str], Any],
    target_materials: List[str],
    coverage: float = 0.998
) -> Tuple[dict, List[Tuple[str, str]]]:
    """
    Build a SALib problem dict for global analysis across multiple materials.

    Includes ALL intensity parameters that contribute to any of the target
    materials.  Parameters are grouped by technology sector.

    Parameters
    ----------
    fitted_distributions : dict
    target_materials : list of str
        Materials to include in the aggregate output
    coverage : float

    Returns
    -------
    problem : dict
        SALib problem dict with groups
    param_keys : list of (technology, material)
    """
    param_keys = sorted(
        [(tech, mat) for (tech, mat) in fitted_distributions
         if mat in target_materials],
        key=lambda x: (x[1], x[0])
    )

    names = [f"{tech}|{mat}" for tech, mat in param_keys]
    bounds = [
        list(get_parameter_bounds(fitted_distributions[(tech, mat)], coverage))
        for tech, mat in param_keys
    ]
    groups = [
        INTENSITY_TECH_TO_SECTOR.get(tech, 'Other')
        for tech, _ in param_keys
    ]

    problem = {
        'num_vars': len(param_keys),
        'names': names,
        'bounds': bounds,
        'groups': groups,
    }
    return problem, param_keys


# =============================================================================
# COEFFICIENT PRECOMPUTATION
# =============================================================================

def precompute_demand_coefficients(
    param_keys: List[Tuple[str, str]],
    stock_flow_states: Dict,
    technology_mapping: Dict,
    scenarios: Optional[List[str]] = None,
    years: Optional[List[int]] = None
) -> np.ndarray:
    """
    Precompute the linear coefficient for each intensity parameter.

    For the linear model Demand(m) = sum_i [a_i * I_i], computes the
    coefficient a_i = sum_{scenario, year, cap_tech} [additions * weight]
    for each parameter i.

    Parameters
    ----------
    param_keys : list of (intensity_tech, material)
    stock_flow_states : dict
        {(scenario, cap_tech): StockFlowState}
    technology_mapping : dict
        TECHNOLOGY_MAPPING from technology_mapping.py
    scenarios : list, optional
        Filter to these scenarios (default: all)
    years : list, optional
        Filter to these years (default: all)

    Returns
    -------
    coefficients : np.ndarray of shape (D,)
        coefficient[i] corresponds to param_keys[i]
    """
    # Build a lookup from intensity_tech to param index
    param_index = {}
    for i, (tech, mat) in enumerate(param_keys):
        param_index[(tech, mat)] = i

    coefficients = np.zeros(len(param_keys))

    for (scenario, cap_tech), state in stock_flow_states.items():
        if scenarios is not None and scenario not in scenarios:
            continue

        # Get intensity mappings for this capacity technology
        intensity_mappings = technology_mapping.get(cap_tech, {})

        for intensity_tech, weight in intensity_mappings.items():
            for year, addition_MW in state.additions.items():
                if years is not None and year not in years:
                    continue
                if addition_MW <= 0:
                    continue

                # For each param_key that matches this intensity tech
                for (pk_tech, pk_mat), idx in param_index.items():
                    if pk_tech == intensity_tech:
                        coefficients[idx] += addition_MW * weight

    return coefficients


# =============================================================================
# PER-MATERIAL SOBOL ANALYSIS
# =============================================================================

def run_sobol_single_material(
    target_material: str,
    fitted_distributions: Dict[Tuple[str, str], Any],
    stock_flow_states: Dict,
    technology_mapping: Dict,
    N: int = 1024,
    scenarios: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    calc_second_order: bool = False,
    random_state: int = 42
) -> Optional[Dict]:
    """
    Run Sobol analysis for a single target material.

    Parameters
    ----------
    target_material : str
    fitted_distributions : dict
    stock_flow_states : dict
    technology_mapping : dict
    N : int
        Base Saltelli sample size (default 1024).
    scenarios, years : optional filters
    calc_second_order : bool
        Compute second-order indices (default False).
    random_state : int

    Returns
    -------
    dict or None
        Results dict with keys: material, n_params, n_evaluations,
        param_keys, S1, S1_conf, ST, ST_conf, problem, coefficients.
        Returns None if D < 2 (sets S1=1.0 analytically).
    """
    if not SALIB_AVAILABLE:
        raise ImportError("SALib is required. Install with: pip install SALib>=1.4.0")

    param_keys = identify_contributing_parameters(
        fitted_distributions, target_material
    )

    if len(param_keys) == 0:
        return None

    # D=1: Sobol needs >= 2 dimensions; set analytically
    if len(param_keys) == 1:
        tech, mat = param_keys[0]
        coeff = precompute_demand_coefficients(
            param_keys, stock_flow_states, technology_mapping,
            scenarios, years
        )
        return {
            'material': target_material,
            'n_params': 1,
            'n_evaluations': 0,
            'param_keys': param_keys,
            'S1': np.array([1.0]),
            'S1_conf': np.array([0.0]),
            'ST': np.array([1.0]),
            'ST_conf': np.array([0.0]),
            'problem': {'num_vars': 1, 'names': [tech], 'bounds': []},
            'coefficients': coeff,
            'analytical': True,
        }

    # Build problem definition
    problem, param_keys = build_salib_problem(
        fitted_distributions, target_material
    )

    # Precompute coefficients for vectorized evaluation
    coefficients = precompute_demand_coefficients(
        param_keys, stock_flow_states, technology_mapping,
        scenarios, years
    )

    # Skip if all coefficients are zero (no capacity additions for this material)
    if np.all(coefficients == 0):
        logger.warning(f"  {target_material}: All coefficients are zero, skipping")
        return None

    # Generate Saltelli samples
    np.random.seed(random_state)
    X = saltelli.sample(problem, N, calc_second_order=calc_second_order)
    n_evaluations = X.shape[0]

    # Vectorized model evaluation: Y = X @ coefficients
    Y = X @ coefficients

    # Compute Sobol indices
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Si = sobol.analyze(
            problem, Y, calc_second_order=calc_second_order,
            print_to_console=False
        )

    return {
        'material': target_material,
        'n_params': len(param_keys),
        'n_evaluations': n_evaluations,
        'param_keys': param_keys,
        'S1': np.clip(Si['S1'], 0, None),
        'S1_conf': Si['S1_conf'],
        'ST': np.clip(Si['ST'], 0, None),
        'ST_conf': Si['ST_conf'],
        'problem': problem,
        'coefficients': coefficients,
        'analytical': False,
    }


def run_sobol_all_materials(
    fitted_distributions: Dict[Tuple[str, str], Any],
    stock_flow_states: Dict,
    technology_mapping: Dict,
    materials: Optional[List[str]] = None,
    N: int = 1024,
    scenarios: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run per-material Sobol analysis for all (or specified) materials.

    Returns
    -------
    pd.DataFrame
        Columns: material, technology, S1, S1_conf, ST, ST_conf,
                 coefficient, n_params, n_evaluations, analytical
    """
    if materials is None:
        materials = sorted(set(mat for _, mat in fitted_distributions))

    rows = []
    for i, material in enumerate(materials, 1):
        logger.info(f"  [{i}/{len(materials)}] {material}")

        result = run_sobol_single_material(
            target_material=material,
            fitted_distributions=fitted_distributions,
            stock_flow_states=stock_flow_states,
            technology_mapping=technology_mapping,
            N=N,
            scenarios=scenarios,
            years=years,
            random_state=random_state,
        )

        if result is None:
            continue

        for j, (tech, mat) in enumerate(result['param_keys']):
            rows.append({
                'material': material,
                'technology': tech,
                'S1': float(result['S1'][j]),
                'S1_conf': float(result['S1_conf'][j]),
                'ST': float(result['ST'][j]),
                'ST_conf': float(result['ST_conf'][j]),
                'coefficient': float(result['coefficients'][j]),
                'n_params': result['n_params'],
                'n_evaluations': result['n_evaluations'],
                'analytical': result.get('analytical', False),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['material', 'S1'], ascending=[True, False])
        df = df.reset_index(drop=True)
    return df


# =============================================================================
# GROUPED SOBOL ANALYSIS (BY TECHNOLOGY SECTOR)
# =============================================================================

def run_grouped_sobol_single_material(
    target_material: str,
    fitted_distributions: Dict[Tuple[str, str], Any],
    stock_flow_states: Dict,
    technology_mapping: Dict,
    N: int = 1024,
    scenarios: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    random_state: int = 42
) -> Optional[Dict]:
    """
    Run grouped Sobol analysis for a single material, grouping intensity
    parameters by technology sector (Solar, Wind, Nuclear, etc.).

    Returns
    -------
    dict or None
        Results with sector-level S1 and ST indices.
    """
    if not SALIB_AVAILABLE:
        raise ImportError("SALib is required.")

    problem, param_keys = build_salib_problem_grouped(
        fitted_distributions, target_material
    )

    if problem['num_vars'] < 2:
        return None

    # Check we have at least 2 distinct groups
    unique_groups = sorted(set(problem['groups']))
    if len(unique_groups) < 2:
        return None

    coefficients = precompute_demand_coefficients(
        param_keys, stock_flow_states, technology_mapping,
        scenarios, years
    )

    if np.all(coefficients == 0):
        return None

    np.random.seed(random_state)
    X = saltelli.sample(problem, N, calc_second_order=False)
    Y = X @ coefficients

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Si = sobol.analyze(
            problem, Y, calc_second_order=False,
            print_to_console=False
        )

    return {
        'material': target_material,
        'groups': unique_groups,
        'n_params': problem['num_vars'],
        'n_evaluations': X.shape[0],
        'S1': np.clip(Si['S1'], 0, None),
        'S1_conf': Si['S1_conf'],
        'ST': np.clip(Si['ST'], 0, None),
        'ST_conf': Si['ST_conf'],
    }


def run_grouped_sobol_all_materials(
    fitted_distributions: Dict[Tuple[str, str], Any],
    stock_flow_states: Dict,
    technology_mapping: Dict,
    materials: Optional[List[str]] = None,
    N: int = 1024,
    scenarios: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run grouped Sobol for all materials.

    Returns
    -------
    pd.DataFrame
        Columns: material, sector, S1, S1_conf, ST, ST_conf
    """
    if materials is None:
        materials = sorted(set(mat for _, mat in fitted_distributions))

    rows = []
    for i, material in enumerate(materials, 1):
        logger.info(f"  [{i}/{len(materials)}] {material} (grouped)")

        result = run_grouped_sobol_single_material(
            target_material=material,
            fitted_distributions=fitted_distributions,
            stock_flow_states=stock_flow_states,
            technology_mapping=technology_mapping,
            N=N,
            scenarios=scenarios,
            years=years,
            random_state=random_state,
        )

        if result is None:
            continue

        for j, sector in enumerate(result['groups']):
            rows.append({
                'material': material,
                'sector': sector,
                'S1': float(result['S1'][j]),
                'S1_conf': float(result['S1_conf'][j]),
                'ST': float(result['ST'][j]),
                'ST_conf': float(result['ST_conf'][j]),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['material', 'S1'], ascending=[True, False])
        df = df.reset_index(drop=True)
    return df


# =============================================================================
# GLOBAL SOBOL ANALYSIS (CRITICAL MATERIALS AGGREGATE)
# =============================================================================

# Materials commonly identified as critical for energy transition
CRITICAL_MATERIALS = [
    'Copper', 'Neodymium', 'Dysprosium', 'Lithium', 'Cobalt', 'Nickel',
    'Tellurium', 'Indium', 'Gallium', 'Germanium', 'Selenium',
    'Cadmium', 'Silicon', 'Silver', 'Manganese', 'Chromium', 'Molybdenum',
]


def run_global_sobol(
    fitted_distributions: Dict[Tuple[str, str], Any],
    stock_flow_states: Dict,
    technology_mapping: Dict,
    target_materials: Optional[List[str]] = None,
    N: int = 1024,
    scenarios: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run global Sobol analysis for aggregate demand across multiple materials.

    All intensity parameters contributing to any target material are included.
    Parameters are grouped by technology sector.  The output is total demand
    (tonnes) summed across all target materials.

    Parameters
    ----------
    target_materials : list, optional
        Materials to include (default: CRITICAL_MATERIALS)

    Returns
    -------
    global_df : pd.DataFrame
        Sector-level S1 and ST for aggregate demand
    metadata : dict
        Analysis metadata (n_params, n_evaluations, etc.)
    """
    if not SALIB_AVAILABLE:
        raise ImportError("SALib is required.")

    if target_materials is None:
        # Filter to materials that actually exist in the fitted distributions
        available = set(mat for _, mat in fitted_distributions)
        target_materials = [m for m in CRITICAL_MATERIALS if m in available]

    problem, param_keys = build_global_salib_problem(
        fitted_distributions, target_materials
    )

    if problem['num_vars'] < 2:
        logger.warning("Global Sobol: fewer than 2 parameters, skipping")
        return pd.DataFrame(), {}

    unique_groups = sorted(set(problem['groups']))
    if len(unique_groups) < 2:
        logger.warning("Global Sobol: fewer than 2 groups, skipping")
        return pd.DataFrame(), {}

    # Precompute coefficients — each parameter maps to ONE material,
    # so we sum demand across all target materials
    coefficients = precompute_demand_coefficients(
        param_keys, stock_flow_states, technology_mapping,
        scenarios, years
    )

    if np.all(coefficients == 0):
        logger.warning("Global Sobol: all coefficients zero, skipping")
        return pd.DataFrame(), {}

    np.random.seed(random_state)
    X = saltelli.sample(problem, N, calc_second_order=False)
    Y = X @ coefficients

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Si = sobol.analyze(
            problem, Y, calc_second_order=False,
            print_to_console=False
        )

    rows = []
    for j, sector in enumerate(unique_groups):
        rows.append({
            'sector': sector,
            'S1': float(np.clip(Si['S1'][j], 0, None)),
            'S1_conf': float(Si['S1_conf'][j]),
            'ST': float(np.clip(Si['ST'][j], 0, None)),
            'ST_conf': float(Si['ST_conf'][j]),
        })

    global_df = pd.DataFrame(rows).sort_values('S1', ascending=False).reset_index(drop=True)

    metadata = {
        'n_params': problem['num_vars'],
        'n_groups': len(unique_groups),
        'n_evaluations': X.shape[0],
        'target_materials': target_materials,
        'groups': unique_groups,
    }

    return global_df, metadata


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_sobol_by_material(
    sobol_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    top_n_materials: int = 12,
    figsize: Tuple[int, int] = (16, 14)
) -> plt.Figure:
    """
    Multi-panel horizontal bar chart of S1 indices per material.
    Each panel shows one material with technologies ranked by S1.
    """
    if sobol_df.empty:
        logger.warning("Empty DataFrame, skipping S1 bar chart")
        return None

    # Select top materials by total coefficient (proxy for demand magnitude)
    mat_totals = sobol_df.groupby('material')['coefficient'].sum().sort_values(ascending=False)
    top_materials = mat_totals.head(top_n_materials).index.tolist()

    n_mats = len(top_materials)
    if n_mats == 0:
        return None

    ncols = 3
    nrows = int(np.ceil(n_mats / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    cmap = plt.cm.YlOrRd

    for idx, material in enumerate(top_materials):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        mat_data = sobol_df[sobol_df['material'] == material].sort_values('S1', ascending=True)
        techs = mat_data['technology'].values
        s1_vals = mat_data['S1'].values
        s1_conf = mat_data['S1_conf'].values

        colors = [cmap(v / max(s1_vals.max(), 0.01)) for v in s1_vals]
        bars = ax.barh(range(len(techs)), s1_vals, xerr=s1_conf,
                       color=colors, edgecolor='grey', linewidth=0.5,
                       capsize=3, ecolor='#555555')

        ax.set_yticks(range(len(techs)))
        ax.set_yticklabels(techs, fontsize=8)
        ax.set_xlim(0, min(1.05, s1_vals.max() * 1.3 + 0.05))
        ax.set_xlabel('S1 (first-order index)', fontsize=8)
        ax.set_title(material, fontsize=10, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.5)

        # Annotate top bar
        if len(s1_vals) > 0 and s1_vals.max() > 0.01:
            top_idx = np.argmax(s1_vals)
            ax.annotate(f'{s1_vals[top_idx]:.2f}',
                        xy=(s1_vals[top_idx], top_idx),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=7, va='center')

    # Hide unused subplots
    for idx in range(n_mats, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle('Sobol First-Order Indices (S1) by Material',
                 fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        logger.info(f"Saved: {output_path}")

    plt.close(fig)
    return fig


def plot_sobol_heatmap(
    sobol_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Material x technology heatmap of S1 values.
    """
    if sobol_df.empty:
        return None

    pivot = sobol_df.pivot_table(
        index='material', columns='technology', values='S1', aggfunc='first'
    )

    # Sort by mean S1
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    col_order = pivot.mean(axis=0).sort_values(ascending=False).index
    pivot = pivot.loc[row_order, col_order]

    fig, ax = plt.subplots(figsize=figsize)
    mask = pivot.isna()

    sns.heatmap(
        pivot, mask=mask, cmap='YlOrRd', vmin=0, vmax=1,
        annot=True, fmt='.2f', linewidths=0.5, linecolor='white',
        ax=ax, cbar_kws={'label': 'Sobol S1', 'shrink': 0.8},
        annot_kws={'size': 7}
    )

    ax.set_xlabel('Intensity Technology', fontsize=11)
    ax.set_ylabel('Material', fontsize=11)
    ax.set_title('Sobol First-Order Sensitivity Indices (S1)\nMaterial × Technology',
                 fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        logger.info(f"Saved: {output_path}")

    plt.close(fig)
    return fig


def plot_sobol_s1_vs_st(
    sobol_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    S1 vs ST scatter plot to validate model linearity.
    Points on the y=x line confirm negligible interactions.
    """
    if sobol_df.empty:
        return None

    # Exclude analytical D=1 entries
    df = sobol_df[~sobol_df['analytical']].copy()
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(df['S1'], df['ST'], alpha=0.6, s=30, c='#2196F3',
               edgecolor='white', linewidth=0.5, zorder=3)

    # y=x reference line
    lim = max(df['S1'].max(), df['ST'].max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1, alpha=0.7, label='S1 = ST (linear model)')

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('First-Order Index (S1)', fontsize=11)
    ax.set_ylabel('Total-Order Index (ST)', fontsize=11)
    ax.set_title('Sobol S1 vs ST — Linearity Validation',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')

    # Compute R^2 as annotation
    if len(df) > 1:
        from scipy.stats import pearsonr
        r, _ = pearsonr(df['S1'], df['ST'])
        ax.annotate(f'R² = {r**2:.4f}\nn = {len(df)}',
                    xy=(0.05, 0.90), xycoords='axes fraction',
                    fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        logger.info(f"Saved: {output_path}")

    plt.close(fig)
    return fig


def plot_sobol_vs_spearman(
    sobol_df: pd.DataFrame,
    spearman_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> Optional[plt.Figure]:
    """
    Cross-validation scatter: Sobol S1 vs Spearman rho² for shared parameters.
    """
    if sobol_df.empty:
        return None

    if spearman_path is None or not Path(spearman_path).exists():
        logger.info("Spearman results not found, skipping cross-validation plot")
        return None

    spearman_df = pd.read_csv(spearman_path)

    # Match on (material, technology)
    # Spearman uses 'target_material' for the output material and 'material'
    # for the intensity parameter's material
    if 'target_material' in spearman_df.columns:
        spearman_df = spearman_df[
            spearman_df['material'] == spearman_df['target_material']
        ].copy()

    merged = sobol_df.merge(
        spearman_df[['technology', 'material', 'spearman_rho']],
        on=['technology', 'material'],
        how='inner'
    )

    if merged.empty:
        logger.info("No matching parameters between Sobol and Spearman")
        return None

    merged['rho_squared'] = merged['spearman_rho'] ** 2

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(merged['rho_squared'], merged['S1'], alpha=0.6, s=30,
               c='#FF5722', edgecolor='white', linewidth=0.5, zorder=3)

    lim = max(merged['rho_squared'].max(), merged['S1'].max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1, alpha=0.7,
            label='Perfect agreement')

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('Spearman ρ²', fontsize=11)
    ax.set_ylabel('Sobol S1', fontsize=11)
    ax.set_title('Cross-Method Validation: Sobol S1 vs Spearman ρ²',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')

    if len(merged) > 1:
        from scipy.stats import pearsonr
        r, _ = pearsonr(merged['rho_squared'], merged['S1'])
        ax.annotate(f'R² = {r**2:.4f}\nn = {len(merged)}',
                    xy=(0.05, 0.90), xycoords='axes fraction',
                    fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        logger.info(f"Saved: {output_path}")

    plt.close(fig)
    return fig


def plot_grouped_sobol(
    grouped_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    top_n_materials: int = 15,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Stacked bar chart of grouped Sobol S1 indices by technology sector.
    Each bar represents one material, segments show sector contributions.
    """
    if grouped_df.empty:
        return None

    # Get materials with most total S1 spread
    mat_order = (
        grouped_df.groupby('material')['S1'].max()
        .sort_values(ascending=False)
        .head(top_n_materials)
        .index.tolist()
    )
    df = grouped_df[grouped_df['material'].isin(mat_order)].copy()

    pivot = df.pivot_table(index='material', columns='sector', values='S1',
                           aggfunc='first', fill_value=0)
    pivot = pivot.loc[mat_order]

    # Color palette for sectors
    sector_colors = {
        'Solar': '#FFC107',
        'Wind': '#2196F3',
        'Nuclear': '#9C27B0',
        'Hydro': '#00BCD4',
        'Fossil': '#795548',
        'Biomass': '#4CAF50',
        'Geothermal': '#FF5722',
        'Other': '#9E9E9E',
    }

    fig, ax = plt.subplots(figsize=figsize)

    bottom = np.zeros(len(pivot))
    sectors_plotted = []

    for sector in pivot.columns:
        vals = pivot[sector].values
        color = sector_colors.get(sector, '#9E9E9E')
        ax.barh(range(len(pivot)), vals, left=bottom, color=color,
                edgecolor='white', linewidth=0.5, label=sector)
        bottom += vals
        sectors_plotted.append(sector)

    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel('Sobol S1 (grouped by technology sector)', fontsize=11)
    ax.set_title('Grouped Sobol Analysis: Technology Sector Contributions\nto Material Demand Variance',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, title='Sector')
    ax.set_xlim(0, min(1.05, bottom.max() * 1.1))
    ax.invert_yaxis()

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        logger.info(f"Saved: {output_path}")

    plt.close(fig)
    return fig


def plot_global_sobol(
    global_df: pd.DataFrame,
    metadata: Dict,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Bar chart of global Sobol S1 indices by technology sector for
    aggregate critical material demand.
    """
    if global_df.empty:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    df = global_df.sort_values('S1', ascending=True)

    sector_colors = {
        'Solar': '#FFC107', 'Wind': '#2196F3', 'Nuclear': '#9C27B0',
        'Hydro': '#00BCD4', 'Fossil': '#795548', 'Biomass': '#4CAF50',
        'Geothermal': '#FF5722', 'Other': '#9E9E9E',
    }

    colors = [sector_colors.get(s, '#9E9E9E') for s in df['sector']]

    bars = ax.barh(range(len(df)), df['S1'].values, xerr=df['S1_conf'].values,
                   color=colors, edgecolor='grey', linewidth=0.5,
                   capsize=4, ecolor='#555555')

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['sector'].values, fontsize=11)
    ax.set_xlabel('Sobol S1 (first-order index)', fontsize=11)
    ax.set_title('Global Sobol Analysis: Technology Sector Contributions\n'
                 'to Aggregate Critical Material Demand Variance',
                 fontsize=13, fontweight='bold')

    # Annotate values
    for i, (_, row) in enumerate(df.iterrows()):
        ax.annotate(f'{row["S1"]:.3f}', xy=(row['S1'], i),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=9, va='center')

    n_mats = len(metadata.get('target_materials', []))
    ax.annotate(f'Aggregate across {n_mats} critical materials\n'
                f'{metadata.get("n_params", "?")} parameters, '
                f'{metadata.get("n_evaluations", "?"):,} evaluations',
                xy=(0.95, 0.05), xycoords='axes fraction',
                fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

    ax.set_xlim(0, min(1.05, df['S1'].max() * 1.3 + 0.05))
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        logger.info(f"Saved: {output_path}")

    plt.close(fig)
    return fig


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_sobol_report(
    sobol_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
    global_df: pd.DataFrame,
    global_metadata: Dict,
    output_path: Optional[Path] = None
) -> str:
    """Generate a text summary report of Sobol analysis results."""
    lines = []
    lines.append("=" * 80)
    lines.append("SOBOL SENSITIVITY ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    # --- Per-material individual results ---
    lines.append("1. PER-MATERIAL SOBOL INDICES (Individual Technologies)")
    lines.append("-" * 60)

    if not sobol_df.empty:
        materials_analyzed = sobol_df['material'].nunique()
        total_params = len(sobol_df)
        analytical_count = sobol_df['analytical'].sum()
        lines.append(f"   Materials analyzed: {materials_analyzed}")
        lines.append(f"   Total (material, technology) pairs: {total_params}")
        lines.append(f"   Analytical (D=1, S1=1.0): {analytical_count}")
        lines.append("")

        # Top 20 most influential parameters
        lines.append("   Top 20 most influential intensity parameters:")
        lines.append(f"   {'Material':<15} {'Technology':<25} {'S1':>8} {'ST':>8} {'Coeff':>12}")
        lines.append("   " + "-" * 70)

        top = sobol_df.nlargest(20, 'S1')
        for _, row in top.iterrows():
            lines.append(
                f"   {row['material']:<15} {row['technology']:<25} "
                f"{row['S1']:>8.4f} {row['ST']:>8.4f} "
                f"{row['coefficient']:>12.1f}"
            )

        # Linearity validation
        non_analytical = sobol_df[~sobol_df['analytical']]
        if len(non_analytical) > 1:
            from scipy.stats import pearsonr
            r, _ = pearsonr(non_analytical['S1'], non_analytical['ST'])
            lines.append("")
            lines.append(f"   Linearity validation: R²(S1, ST) = {r**2:.6f}")
            if r ** 2 > 0.99:
                lines.append("   → Model is effectively linear (S1 ≈ ST), confirming")
                lines.append("     Spearman and elasticity methods are valid.")
            else:
                lines.append("   → Some interaction effects detected (S1 ≠ ST).")

        # Concentrated vs diversified sensitivity
        lines.append("")
        lines.append("   Sensitivity concentration (max S1 per material):")
        max_s1 = sobol_df.groupby('material')['S1'].max().sort_values(ascending=False)
        concentrated = max_s1[max_s1 > 0.8]
        diversified = max_s1[max_s1 < 0.4]
        lines.append(f"   Concentrated (max S1 > 0.8): {len(concentrated)} materials")
        for mat, val in concentrated.items():
            tech = sobol_df[(sobol_df['material'] == mat)].nlargest(1, 'S1')['technology'].iloc[0]
            lines.append(f"     {mat}: S1={val:.3f} ({tech})")
        lines.append(f"   Diversified (max S1 < 0.4): {len(diversified)} materials")
    else:
        lines.append("   No per-material results available.")

    lines.append("")

    # --- Grouped results ---
    lines.append("2. GROUPED SOBOL INDICES (Technology Sectors)")
    lines.append("-" * 60)

    if not grouped_df.empty:
        lines.append("   Sector contributions to material demand variance:")
        lines.append("")

        for material in grouped_df['material'].unique():
            mat_data = grouped_df[grouped_df['material'] == material].sort_values('S1', ascending=False)
            top_sector = mat_data.iloc[0]
            lines.append(
                f"   {material:<15} → {top_sector['sector']} "
                f"(S1={top_sector['S1']:.3f})"
            )
    else:
        lines.append("   No grouped results available.")

    lines.append("")

    # --- Global results ---
    lines.append("3. GLOBAL SOBOL ANALYSIS (Aggregate Critical Material Demand)")
    lines.append("-" * 60)

    if not global_df.empty:
        n_mats = len(global_metadata.get('target_materials', []))
        lines.append(f"   Aggregate across {n_mats} critical materials")
        lines.append(f"   Parameters: {global_metadata.get('n_params', '?')}")
        lines.append(f"   Evaluations: {global_metadata.get('n_evaluations', '?'):,}")
        lines.append("")
        lines.append(f"   {'Sector':<15} {'S1':>8} {'±':>3} {'conf':>8} {'ST':>8}")
        lines.append("   " + "-" * 45)

        for _, row in global_df.iterrows():
            lines.append(
                f"   {row['sector']:<15} {row['S1']:>8.4f} ± {row['S1_conf']:>8.4f} "
                f"{row['ST']:>8.4f}"
            )

        top = global_df.iloc[0]
        lines.append("")
        lines.append(
            f"   → {top['sector']} sector intensity uncertainty dominates "
            f"aggregate critical material demand variance (S1={top['S1']:.3f})"
        )
    else:
        lines.append("   No global results available.")

    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Saved: {output_path}")

    return report


# =============================================================================
# TOP-LEVEL RUNNER (FROM FILES)
# =============================================================================

def run_sobol_from_files(
    intensity_path: str,
    capacity_path: str,
    output_dir: str,
    figures_dir: str,
    N: int = 1024,
    scenario_filter: Optional[str] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run complete Sobol analysis from data files.

    Parameters
    ----------
    intensity_path : str
        Path to intensity_data.csv
    capacity_path : str
        Path to capacity projections CSV
    output_dir : str
        Directory for CSV outputs
    figures_dir : str
        Directory for figure outputs
    N : int
        Base Saltelli sample size
    scenario_filter : str, optional
        Single scenario name to analyze, or None for all
    random_state : int

    Returns
    -------
    sobol_df, grouped_df, global_df : pd.DataFrames
    """
    from src.data_ingestion import load_all_data
    from src.distribution_fitting import DistributionFitter
    from src.stock_flow_simulation import MaterialsStockFlowSimulation
    from src.technology_mapping import TECHNOLOGY_MAPPING

    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    print("  Loading data and fitting distributions...")
    all_data = load_all_data(
        str(intensity_path), str(capacity_path)
    )
    intensity_data = all_data['intensity']
    capacity_data = all_data['capacity_national']

    fitter = DistributionFitter()
    fitted_distributions = fitter.fit_all(intensity_data)

    # Build simulation (we only need the stock-flow states, not full MC)
    sim = MaterialsStockFlowSimulation(
        capacity_data=capacity_data,
        intensity_data=intensity_data,
        fitted_distributions=fitted_distributions,
        random_state=random_state
    )
    stock_flow_states = sim.build_stock_flow_model()

    # Apply scenario filter
    scenarios = None
    if scenario_filter:
        scenarios = [scenario_filter]
        stock_flow_states = {
            k: v for k, v in stock_flow_states.items()
            if k[0] == scenario_filter
        }
        print(f"  Filtered to scenario: {scenario_filter}")

    years = sim.years

    # --- 1. Per-material individual Sobol ---
    print("\n  Running per-material individual Sobol analysis...")
    sobol_df = run_sobol_all_materials(
        fitted_distributions=fitted_distributions,
        stock_flow_states=stock_flow_states,
        technology_mapping=TECHNOLOGY_MAPPING,
        N=N,
        scenarios=scenarios,
        years=years,
        random_state=random_state,
    )
    sobol_df.to_csv(output_dir / "sobol_indices.csv", index=False)
    print(f"  Saved: sobol_indices.csv ({len(sobol_df)} rows)")

    # --- 2. Per-material grouped Sobol ---
    print("\n  Running per-material grouped Sobol analysis...")
    grouped_df = run_grouped_sobol_all_materials(
        fitted_distributions=fitted_distributions,
        stock_flow_states=stock_flow_states,
        technology_mapping=TECHNOLOGY_MAPPING,
        N=N,
        scenarios=scenarios,
        years=years,
        random_state=random_state,
    )
    grouped_df.to_csv(output_dir / "sobol_grouped_indices.csv", index=False)
    print(f"  Saved: sobol_grouped_indices.csv ({len(grouped_df)} rows)")

    # --- 3. Global Sobol ---
    print("\n  Running global Sobol analysis (critical materials aggregate)...")
    global_df, global_metadata = run_global_sobol(
        fitted_distributions=fitted_distributions,
        stock_flow_states=stock_flow_states,
        technology_mapping=TECHNOLOGY_MAPPING,
        N=N,
        scenarios=scenarios,
        years=years,
        random_state=random_state,
    )
    global_df.to_csv(output_dir / "sobol_global_indices.csv", index=False)
    print(f"  Saved: sobol_global_indices.csv ({len(global_df)} rows)")

    # --- 4. Visualizations ---
    print("\n  Generating visualizations...")

    plot_sobol_by_material(
        sobol_df, output_path=figures_dir / "sobol_s1_bar.png"
    )

    plot_sobol_heatmap(
        sobol_df, output_path=figures_dir / "sobol_summary_heatmap.png"
    )

    plot_sobol_s1_vs_st(
        sobol_df, output_path=figures_dir / "sobol_s1_vs_st.png"
    )

    spearman_path = output_dir / "spearman_sensitivity.csv"
    plot_sobol_vs_spearman(
        sobol_df, spearman_path=spearman_path,
        output_path=figures_dir / "sobol_vs_spearman.png"
    )

    plot_grouped_sobol(
        grouped_df, output_path=figures_dir / "sobol_grouped_bar.png"
    )

    plot_global_sobol(
        global_df, global_metadata,
        output_path=figures_dir / "sobol_global_bar.png"
    )

    return sobol_df, grouped_df, global_df, global_metadata


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run complete Sobol sensitivity analysis."""
    print("=" * 80)
    print("SOBOL SENSITIVITY ANALYSIS FOR MATERIALS DEMAND MODEL")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / "outputs" / "data" / "sensitivity"
    figures_dir = base_dir / "outputs" / "figures" / "sensitivity"
    reports_dir = base_dir / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    intensity_path = base_dir / "data" / "intensity_data.csv"
    capacity_path = base_dir / "data" / "StdScen24_annual_national.csv"

    if not intensity_path.exists() or not capacity_path.exists():
        print(f"  ERROR: Data files not found")
        print(f"    Intensity: {intensity_path} (exists: {intensity_path.exists()})")
        print(f"    Capacity: {capacity_path} (exists: {capacity_path.exists()})")
        sys.exit(1)

    sobol_df, grouped_df, global_df, global_metadata = run_sobol_from_files(
        intensity_path=str(intensity_path),
        capacity_path=str(capacity_path),
        output_dir=str(output_dir),
        figures_dir=str(figures_dir),
        N=1024,
        random_state=42,
    )

    # Generate report
    print("\n  Generating report...")
    report = generate_sobol_report(
        sobol_df, grouped_df, global_df, global_metadata,
        output_path=reports_dir / "sobol_analysis_report.txt"
    )
    print("\n" + report)

    print("\n" + "=" * 80)
    print("SOBOL ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  Data:    {output_dir}/sobol_*.csv")
    print(f"  Figures: {figures_dir}/sobol_*.png")
    print(f"  Report:  {reports_dir}/sobol_analysis_report.txt")


if __name__ == "__main__":
    main()
