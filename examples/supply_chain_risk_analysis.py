"""
Supply Chain Risk Analysis Script
==================================

Production-ready script for analyzing supply chain risk by comparing Monte Carlo
materials demand projections against USGS production and trade data.

This script allocates projected material demand across Country Risk Categories (CRC)
based on US import dependency and source country risk classifications, then visualizes
the results with stacked bar charts showing demand composition by risk category
alongside historical production/consumption reference lines.

USAGE:
------
Basic usage (all scenarios, all materials):
    python supply_chain_risk_analysis.py \\
        --demand_csv outputs/material_demand_by_scenario.csv \\
        --risk_xlsx data/risk_charts_inputs.xlsx

Focus on specific scenarios with log scale:
    python supply_chain_risk_analysis.py \\
        --demand_csv outputs/material_demand_by_scenario.csv \\
        --risk_xlsx data/risk_charts_inputs.xlsx \\
        --scenarios Mid_Case,Mid_Case_100by2035 \\
        --yaxis log \\
        --outdir outputs/focused_analysis

Keep rare earths separate:
    python supply_chain_risk_analysis.py \\
        --demand_csv outputs/material_demand_by_scenario.csv \\
        --risk_xlsx data/risk_charts_inputs.xlsx \\
        --no_aggregate_rare_earths

DEPENDENCIES:
-------------
pandas>=1.5.0, numpy>=1.23.0, matplotlib>=3.6.0, openpyxl>=3.0.0

AUTHOR: Materials Demand Research Team
DATE: January 2026
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

RARE_EARTH_ELEMENTS = ['Dysprosium', 'Neodymium', 'Praseodymium', 'Terbium', 'Yttrium']

CRC_CATEGORIES = ['OECD', 1, 2, 3, 4, 5, 6, 7, 'China', 'Undefined', 'United States']

# Color scheme matching risk_charts.py
CRC_COLORS = {
    'United States': (0.50196078, 0.16535179, 0.50196078, 1.0),  # Purple
    'OECD': (0.04313725, 0.52344483, 0.0, 1.0),                 # Green
    1: (0.59215686, 0.79687812, 0.0, 1.0),                      # Light yellow
    2: (1.0, 0.94048443, 0.0, 1.0),                             # Yellow
    3: (1.0, 0.7467128, 0.0, 1.0),                              # Orange-yellow
    4: (1.0, 0.4567474, 0.0, 1.0),                              # Orange
    5: (1.0, 0.10149942, 0.0, 1.0),                             # Red-orange
    6: (0.85190311, 0.06911188, 0.06911188, 1.0),               # Red
    7: (0.65813149, 0.15953864, 0.15953864, 1.0),               # Dark red
    'China': (0.56796617, 0.34854287, 0.34854287, 1.0),         # Brown-red
    'Undefined': (0.50196078, 0.45471742, 0.50196078, 1.0),     # Gray
}

REQUIRED_DEMAND_COLS = ['scenario', 'year', 'material']

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_demand_data(
    demand_csv: Path,
    unit_scale: float = 1000.0,
    scenarios: Optional[List[str]] = None,
    materials: Optional[List[str]] = None,
    aggregate_rare_earths: bool = True,
    value_col: str = 'p50'
) -> pd.DataFrame:
    """
    Load and prepare Monte Carlo demand projections.

    Parameters
    ----------
    demand_csv : Path
        Path to material_demand_by_scenario.csv
    unit_scale : float
        Division factor for unit conversion (default 1000 = tonnes → kMT)
    scenarios : Optional[List[str]]
        Scenarios to include (None = all)
    materials : Optional[List[str]]
        Materials to include (None = all)
    aggregate_rare_earths : bool
        Whether to sum rare earths into single 'Rare Earths' category
    value_col : str
        Column to use for demand values (default 'p50' = median).
        Options: 'mean', 'p50', 'p25', 'p75', etc.
        Note: Use 'p50' (median) to avoid extreme outliers from fat-tailed distributions.

    Returns
    -------
    pd.DataFrame
        Columns: [scenario, year, material, demand_value, std, p2, p5, p25, p50, p75, p95, p97]
        Units: thousand metric tonnes (if unit_scale=1000)
        The 'demand_value' column contains the selected statistic (median by default)

    Raises
    ------
    FileNotFoundError
        If demand_csv does not exist
    ValueError
        If required columns missing or scenarios/materials not found
    """
    logging.info("Loading demand data...")

    if not demand_csv.exists():
        raise FileNotFoundError(
            f"Demand CSV not found: {demand_csv}\n"
            f"Verify the file exists and path is correct."
        )

    # Load CSV
    df = pd.read_csv(demand_csv)

    # Validate required columns
    missing_cols = set(REQUIRED_DEMAND_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in demand CSV: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Filter scenarios if specified
    if scenarios:
        available_scenarios = set(df['scenario'].unique())
        invalid_scenarios = set(scenarios) - available_scenarios
        if invalid_scenarios:
            raise ValueError(
                f"Scenarios not found in data: {invalid_scenarios}\n"
                f"Available scenarios: {sorted(available_scenarios)}"
            )
        df = df[df['scenario'].isin(scenarios)]
        logging.info(f"  Filtered to scenarios: {scenarios}")

    # Filter materials if specified
    if materials:
        available_materials = set(df['material'].unique())
        invalid_materials = set(materials) - available_materials
        if invalid_materials:
            raise ValueError(
                f"Materials not found in data: {invalid_materials}\n"
                f"Available materials: {sorted(available_materials)}"
            )
        df = df[df['material'].isin(materials)]
        logging.info(f"  Filtered to {len(materials)} specified materials")

    # Apply unit scale conversion
    if unit_scale <= 0:
        raise ValueError(f"unit_scale must be positive, got {unit_scale}")

    numeric_cols = ['mean', 'std', 'p2', 'p5', 'p25', 'p50', 'p75', 'p95', 'p97']
    existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
    df[existing_numeric_cols] = df[existing_numeric_cols] / unit_scale
    logging.info(f"  Applied unit conversion (÷{unit_scale}): tonnes → thousand metric tonnes")

    # Aggregate rare earths if requested
    if aggregate_rare_earths:
        # Identify rare earth rows
        rare_earth_mask = df['material'].isin(RARE_EARTH_ELEMENTS)
        rare_earth_df = df[rare_earth_mask].copy()

        if not rare_earth_df.empty:
            # Group by scenario and year, sum numeric columns
            rare_earth_agg = rare_earth_df.groupby(['scenario', 'year'])[existing_numeric_cols].sum().reset_index()
            rare_earth_agg['material'] = 'Rare Earths'

            # Remove individual rare earths and add aggregated
            df = pd.concat([df[~rare_earth_mask], rare_earth_agg], ignore_index=True)
            logging.info(f"  Aggregated {len(RARE_EARTH_ELEMENTS)} rare earth elements → 'Rare Earths'")
        else:
            logging.warning("  No rare earth elements found in data for aggregation")

    n_scenarios = df['scenario'].nunique()
    n_materials = df['material'].nunique()
    n_years = df['year'].nunique()
    logging.info(f"  Loaded demand data: {n_scenarios} scenarios, {n_materials} materials, {n_years} years")

    return df


def load_usgs_data(risk_xlsx: Path) -> Dict[str, pd.DataFrame]:
    """
    Load USGS import dependency, import shares, CRC, and aggregate data.

    Parameters
    ----------
    risk_xlsx : Path
        Path to risk_charts_inputs.xlsx

    Returns
    -------
    Dict[str, pd.DataFrame]
        {
            'import_dependency': DataFrame with columns [material, 2018-2022, avg],
            'import_shares': DataFrame with columns [material, country, share],
            'crc': DataFrame with columns [country, crc],
            'aggregate': DataFrame with columns [material, year, production,
                                                  consumption, net_import]
        }

    Raises
    ------
    FileNotFoundError
        If risk_xlsx does not exist
    KeyError
        If required sheets missing

    Notes
    -----
    - Treats 'E' in import_dependency as 0
    - Averages import_dependency across 2018-2022
    """
    logging.info("Loading USGS data...")

    if not risk_xlsx.exists():
        raise FileNotFoundError(
            f"Risk workbook not found: {risk_xlsx}\n"
            f"Verify the file exists and path is correct."
        )

    required_sheets = ['import_dependency', 'import_shares', 'crc', 'aggregate']

    try:
        # Load import_dependency
        import_dependency = pd.read_excel(risk_xlsx, sheet_name='import_dependency')
        import_dependency = import_dependency.set_index('material')

        # Replace 'E' with 0 and calculate average
        import_dependency = import_dependency.replace({'E': 0})
        year_cols = [col for col in import_dependency.columns if isinstance(col, (int, str)) and str(col).isdigit()]
        import_dependency['avg'] = import_dependency[year_cols].mean(axis=1)
        logging.info(f"  Loaded import_dependency: {len(import_dependency)} materials")

        # Load import_shares
        import_shares = pd.read_excel(risk_xlsx, sheet_name='import_shares')
        logging.info(f"  Loaded import_shares: {len(import_shares)} entries")

        # Load CRC mapping
        crc = pd.read_excel(risk_xlsx, sheet_name='crc')
        crc = crc.iloc[:, 0:2]  # Take only first two columns
        logging.info(f"  Loaded crc mapping: {len(crc)} countries")

        # Load aggregate data
        aggregate = pd.read_excel(risk_xlsx, sheet_name='aggregate')
        aggregate = aggregate.replace('-', np.nan)
        logging.info(f"  Loaded aggregate data: {len(aggregate)} entries")

    except KeyError as e:
        raise KeyError(
            f"Missing required sheet in {risk_xlsx}: {e}\n"
            f"Required sheets: {required_sheets}"
        )

    return {
        'import_dependency': import_dependency,
        'import_shares': import_shares,
        'crc': crc,
        'aggregate': aggregate
    }


# ============================================================================
# 2. DATA PREPARATION
# ============================================================================

def prepare_crc_shares(
    import_shares: pd.DataFrame,
    import_dependency: pd.DataFrame,
    crc: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate CRC-allocated import shares following risk_charts.py logic.

    This function replicates the exact allocation logic from risk_charts.py
    lines 190-196 to ensure consistency with published results.

    Parameters
    ----------
    import_shares : pd.DataFrame
        Columns: [material, country, share]
    import_dependency : pd.DataFrame
        Index: material, Columns: [avg] where avg = average import reliance 2018-2022
    crc : pd.DataFrame
        Columns: [country, crc]

    Returns
    -------
    pd.DataFrame
        Columns: [material, crc, share_pct]
        CRC categories: ['OECD', 1, 2, 3, 4, 5, 6, 7, 'China',
                         'Undefined', 'United States']

    Logic (from risk_charts.py lines 190-196)
    ------------------------------------------
    1. Merge import_shares with CRC mapping
    2. Override CRC for China → 'China'
    3. Group by material + CRC, sum shares
    4. For each material:
       - Imported share (non-US) = import_shares * (import_reliance / 100)
       - US domestic share = 100 - import_reliance
    5. Fill missing CRC categories with 0
    """
    logging.info("Preparing CRC share allocations...")

    # Step 1: Merge import_shares with CRC mapping
    import_shares_crc = pd.merge(
        import_shares,
        crc[['country', 'crc']],
        on='country',
        how='left'
    )
    import_shares_crc['crc'] = import_shares_crc['crc'].fillna('Undefined')

    # Step 2: Override China to its own category
    import_shares_crc['crc'] = np.where(
        import_shares_crc['country'] == 'China',
        'China',
        import_shares_crc['crc']
    )

    # Step 3: Group by (material, crc), sum shares
    import_shares_crc = import_shares_crc.groupby(
        ['material', 'crc']
    )['share'].sum().reset_index()

    # Step 4: Fill missing CRC categories with 0
    idx = pd.MultiIndex.from_product(
        [import_shares_crc['material'].unique(), CRC_CATEGORIES],
        names=['material', 'crc']
    )
    import_shares_crc = import_shares_crc.set_index(
        ['material', 'crc']
    ).reindex(idx, fill_value=0).reset_index()

    # Step 5: Merge with import_dependency (avg column)
    crc_shares = pd.merge(
        import_shares_crc,
        import_dependency[['avg']],
        left_on='material',
        right_index=True,
        how='left'
    )

    # Step 6: Apply allocation formula (CRITICAL - exact replication)
    # For non-US CRCs: multiply share by import reliance
    crc_shares['share_pct'] = crc_shares['share'] * (crc_shares['avg'] / 100)

    # For US: domestic production = 100 - import reliance
    crc_shares['share_pct'] = np.where(
        crc_shares['crc'] == 'United States',
        100 - crc_shares['avg'],
        crc_shares['share_pct']
    )

    # Return final columns
    result = crc_shares[['material', 'crc', 'share_pct']].copy()

    n_materials = result['material'].nunique()
    logging.info(f"  Prepared CRC shares for {n_materials} materials across {len(CRC_CATEGORIES)} categories")

    return result


def prepare_usgs_comparison_data(aggregate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average production, consumption, net import per material.

    Parameters
    ----------
    aggregate_df : pd.DataFrame
        USGS aggregate data with columns [material, year, production,
                                          consumption, net_import]

    Returns
    -------
    pd.DataFrame
        Columns: [material, avg_production, avg_consumption, avg_net_import]
        Averaged across available years (typically 2018-2022)

    Notes
    -----
    - Handles missing values (NaN, '-') by excluding from average
    - Returns NaN for materials with no data
    """
    logging.info("Preparing USGS comparison data...")

    # Group by material and calculate averages
    usgs_avg = aggregate_df.groupby('material').agg({
        'production': 'mean',
        'consumption': 'mean',
        'net_import': 'mean'
    }).reset_index()

    usgs_avg = usgs_avg.rename(columns={
        'production': 'avg_production',
        'consumption': 'avg_consumption',
        'net_import': 'avg_net_import'
    })

    logging.info(f"  Prepared comparison data for {len(usgs_avg)} materials")

    return usgs_avg


# ============================================================================
# 3. CRC ALLOCATION
# ============================================================================

def allocate_demand_by_crc(
    demand_df: pd.DataFrame,
    crc_shares: pd.DataFrame
) -> pd.DataFrame:
    """
    Allocate demand projections across CRC categories.

    Parameters
    ----------
    demand_df : pd.DataFrame
        Harmonized demand with columns [scenario, year, material, mean, ...]
    crc_shares : pd.DataFrame
        CRC shares from prepare_crc_shares() [material, crc, share_pct]

    Returns
    -------
    pd.DataFrame (tidy format)
        Columns: [scenario, year, material, crc, demand_allocated,
                  demand_mean_total, share_pct]
        demand_allocated = demand_mean_total * (share_pct / 100)

    Handling Missing Materials
    --------------------------
    - Materials in demand_df but not in crc_shares:
        * Log warning with material name
        * Skip allocation (not included in output)
    - Materials in crc_shares but not in demand_df:
        * No action (silent skip)
    """
    logging.info("Allocating demand by CRC categories...")

    # Merge demand with CRC shares
    allocated = pd.merge(
        demand_df,
        crc_shares,
        on='material',
        how='left'
    )

    # Identify materials without USGS data
    missing_usgs = allocated[allocated['share_pct'].isna()]['material'].unique()
    if len(missing_usgs) > 0:
        logging.warning(
            f"Skipping {len(missing_usgs)} materials without USGS import data:\n"
            f"  {', '.join(sorted(missing_usgs))}\n"
            f"  These materials will not appear in CRC-allocated outputs."
        )
        # Remove rows without CRC shares
        allocated = allocated.dropna(subset=['share_pct'])

    # Calculate allocated demand
    allocated['demand_mean_total'] = allocated['mean']
    allocated['demand_allocated'] = allocated['mean'] * (allocated['share_pct'] / 100)

    # Keep relevant columns
    result = allocated[[
        'scenario', 'year', 'material', 'crc',
        'demand_allocated', 'demand_mean_total', 'share_pct'
    ]].copy()

    n_rows = len(result)
    n_materials = result['material'].nunique()
    logging.info(f"  Allocated demand: {n_rows} rows across {n_materials} materials")

    return result


# ============================================================================
# 4. VALIDATION & REPORTING
# ============================================================================

def validate_crc_shares(
    crc_shares: pd.DataFrame,
    tolerance: float = 0.1
) -> pd.DataFrame:
    """
    Validate that CRC shares sum to 100% per material.

    Parameters
    ----------
    crc_shares : pd.DataFrame
        Output from prepare_crc_shares()
    tolerance : float
        Acceptable deviation from 100% (percentage points)

    Returns
    -------
    pd.DataFrame
        Audit table with columns [material, total_share, deviation, valid]

    Raises
    ------
    ValueError
        If any material exceeds tolerance threshold
    """
    logging.info("Validating CRC share allocations...")

    # Sum shares by material
    validation = crc_shares.groupby('material')['share_pct'].sum().reset_index()
    validation = validation.rename(columns={'share_pct': 'total_share'})

    # Calculate deviation from 100
    validation['deviation'] = validation['total_share'] - 100
    validation['valid'] = validation['deviation'].abs() <= tolerance

    # Check if all materials passed
    failed = validation[~validation['valid']]
    if not failed.empty:
        error_msg = "CRC share validation failed for the following materials:\n"
        for _, row in failed.iterrows():
            error_msg += f"  {row['material']}: total={row['total_share']:.2f}%, deviation={row['deviation']:.2f}%\n"
        raise ValueError(error_msg)

    n_materials = len(validation)
    logging.info(f"  All {n_materials} materials passed CRC share validation (tolerance: ±{tolerance}%)")

    return validation


def create_comparison_table(
    allocated_demand: pd.DataFrame,
    usgs_comparison: pd.DataFrame,
    target_year: int = 2035
) -> pd.DataFrame:
    """
    Create comparison table: demand vs USGS production/consumption.

    Parameters
    ----------
    allocated_demand : pd.DataFrame
        Output from allocate_demand_by_crc()
    usgs_comparison : pd.DataFrame
        Output from prepare_usgs_comparison_data()
    target_year : int
        Year to extract demand projections for comparison

    Returns
    -------
    pd.DataFrame
        Columns: [material, scenario, demand_{year}, avg_production,
                  avg_consumption, demand_to_production_ratio,
                  demand_to_consumption_ratio]
    """
    logging.info(f"Creating demand vs production comparison table for year {target_year}...")

    # Filter to target year
    demand_year = allocated_demand[allocated_demand['year'] == target_year].copy()

    if demand_year.empty:
        logging.warning(f"No demand data found for year {target_year}")
        return pd.DataFrame()

    # Sum demand across CRC categories
    demand_total = demand_year.groupby(['scenario', 'material'])['demand_allocated'].sum().reset_index()
    demand_total = demand_total.rename(columns={'demand_allocated': f'demand_{target_year}'})

    # Merge with USGS data
    comparison = pd.merge(
        demand_total,
        usgs_comparison,
        on='material',
        how='left'
    )

    # Calculate ratios
    comparison['demand_to_production_ratio'] = (
        comparison[f'demand_{target_year}'] / comparison['avg_production']
    )
    comparison['demand_to_consumption_ratio'] = (
        comparison[f'demand_{target_year}'] / comparison['avg_consumption']
    )

    logging.info(f"  Created comparison table: {len(comparison)} entries")

    return comparison


def create_summary_report(
    allocated_demand: pd.DataFrame,
    crc_shares: pd.DataFrame,
    usgs_comparison: pd.DataFrame,
    validation_audit: pd.DataFrame,
    comparison_table: pd.DataFrame,
    output_path: Path,
    input_files: Dict[str, Path],
    config: Dict
) -> None:
    """
    Generate text summary report of analysis.

    Parameters
    ----------
    allocated_demand : pd.DataFrame
        CRC-allocated demand
    crc_shares : pd.DataFrame
        CRC share allocation
    usgs_comparison : pd.DataFrame
        USGS comparison data
    validation_audit : pd.DataFrame
        Validation results from validate_crc_shares()
    comparison_table : pd.DataFrame
        Demand vs production comparison
    output_path : Path
        Where to save report.txt
    input_files : Dict[str, Path]
        Paths to input files
    config : Dict
        Configuration parameters
    """
    logging.info("Generating summary report...")

    with open(output_path, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("SUPPLY CHAIN RISK ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Input files
        f.write("INPUT FILES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Demand CSV: {input_files['demand_csv']}\n")
        f.write(f"Risk XLSX: {input_files['risk_xlsx']}\n")
        f.write(f"Unit scale: {config['unit_scale']} (tonnes → thousand metric tonnes)\n\n")

        # Analysis scope
        f.write("ANALYSIS SCOPE\n")
        f.write("-" * 80 + "\n")
        scenarios = sorted(allocated_demand['scenario'].unique())
        f.write(f"Scenarios analyzed: {', '.join(scenarios)}\n")

        materials_with_usgs = sorted(allocated_demand['material'].unique())
        f.write(f"Materials analyzed: {len(materials_with_usgs)} (with USGS data)\n")

        if config.get('aggregate_rare_earths'):
            f.write(f"Rare earths aggregated: Yes ({len(RARE_EARTH_ELEMENTS)} elements → 'Rare Earths')\n")
        else:
            f.write("Rare earths aggregated: No (kept separate)\n")

        f.write("\n")

        # CRC validation
        f.write("CRC SHARE VALIDATION\n")
        f.write("-" * 80 + "\n")
        if validation_audit['valid'].all():
            f.write(f"✓ All materials passed validation (tolerance: ±{config['tolerance']}%)\n")
        else:
            failed = validation_audit[~validation_audit['valid']]
            f.write(f"✗ {len(failed)} materials failed validation:\n")
            for _, row in failed.iterrows():
                f.write(f"  {row['material']}: deviation={row['deviation']:.2f}%\n")
        f.write("\n")

        # Top materials by demand
        if not comparison_table.empty:
            f.write(f"TOP 10 MATERIALS BY DEMAND ({config['target_year']})\n")
            f.write("-" * 80 + "\n")
            for scenario in scenarios:
                scenario_data = comparison_table[comparison_table['scenario'] == scenario]
                top10 = scenario_data.nlargest(10, f"demand_{config['target_year']}")
                f.write(f"\nScenario: {scenario}\n")
                f.write(f"{'Rank':<6}{'Material':<20}{'Demand (kMT)':<20}{'Avg Production':<20}\n")
                for i, (_, row) in enumerate(top10.iterrows(), 1):
                    demand = row[f"demand_{config['target_year']}"]
                    prod = row['avg_production']
                    prod_str = f"{prod:,.0f}" if pd.notna(prod) else "N/A"
                    f.write(f"{i:<6}{row['material']:<20}{demand:,.0f}{prod_str:<20}\n")
            f.write("\n")

            # High demand vs production ratio
            f.write("MATERIALS WITH DEMAND > 2X HISTORICAL PRODUCTION\n")
            f.write("-" * 80 + "\n")
            high_demand = comparison_table[comparison_table['demand_to_production_ratio'] > 2.0]
            if not high_demand.empty:
                f.write(f"{'Material':<20}{'Scenario':<25}{'Demand':<15}{'Avg Prod':<15}{'Ratio':<10}\n")
                for _, row in high_demand.iterrows():
                    f.write(
                        f"{row['material']:<20}"
                        f"{row['scenario']:<25}"
                        f"{row[f'demand_{config['target_year']}']:>14,.0f}"
                        f"{row['avg_production']:>14,.0f}"
                        f"{row['demand_to_production_ratio']:>9.2f}\n"
                    )
            else:
                f.write("None\n")
            f.write("\n")

        # Import dependency
        f.write("MATERIALS WITH HIGHEST IMPORT RELIANCE\n")
        f.write("-" * 80 + "\n")
        # Get average import reliance from CRC shares where crc != 'United States'
        import_reliance = crc_shares[crc_shares['crc'] != 'United States'].groupby('material')['share_pct'].sum()
        import_reliance = import_reliance.sort_values(ascending=False).head(10)
        f.write(f"{'Material':<20}{'Import Reliance (%)':<20}\n")
        for material, reliance in import_reliance.items():
            f.write(f"{material:<20}{reliance:>19.1f}\n")
        f.write("\n")

        # Footer
        f.write("=" * 80 + "\n")
        f.write("End of report\n")
        f.write("=" * 80 + "\n")

    logging.info(f"  Summary report saved to: {output_path}")


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def setup_crc_colors() -> Dict[Union[str, int], Tuple[float, ...]]:
    """
    Return CRC color scheme matching risk_charts.py.

    Returns
    -------
    Dict mapping CRC category to RGBA tuple
    """
    return CRC_COLORS.copy()


def create_stacked_bar_chart(
    allocated_demand: pd.DataFrame,
    usgs_comparison: pd.DataFrame,
    materials: List[str],
    scenarios: List[str],
    yaxis: str = 'linear',
    output_path: Optional[Path] = None,
    overlay_scenario: Optional[str] = None,
    overlay_demand: Optional[pd.DataFrame] = None
) -> plt.Figure:
    """
    Create multi-panel stacked bar chart with production reference lines.

    Parameters
    ----------
    allocated_demand : pd.DataFrame
        CRC-allocated demand data
    usgs_comparison : pd.DataFrame
        USGS production/consumption/net_import data
    materials : List[str]
        Materials to plot (one subplot per material)
    scenarios : List[str]
        Scenarios to plot (one bar per scenario)
    yaxis : str
        'linear' or 'log'
    output_path : Optional[Path]
        If provided, save figure to this path
    overlay_scenario : Optional[str]
        Name of scenario to overlay as a line (e.g., 'Mid_Case_100by2035')
    overlay_demand : Optional[pd.DataFrame]
        Demand data for the overlay scenario (must contain 'year', 'material', 'demand_mean_total')

    Returns
    -------
    matplotlib.figure.Figure
    """
    logging.info(f"Creating stacked bar chart ({yaxis} scale)...")

    # Setup matplotlib for publication quality
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['figure.dpi'] = 100

    colors = setup_crc_colors()

    # Calculate grid dimensions
    n_materials = len(materials)
    n_cols = 4
    n_rows = math.ceil(n_materials / n_cols)

    fig_height = 3.5 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, fig_height))

    # Flatten axes array for easier iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    # Get unique years and create index mapping
    years = np.array(sorted(allocated_demand['year'].unique()))
    n_years = len(years)
    year_to_idx = {year: idx for idx, year in enumerate(years)}

    # CRC stacking order (bottom to top)
    crc_stack_order = ['United States', 'OECD', 1, 2, 3, 4, 5, 6, 7, 'China', 'Undefined']

    # Bar positioning: use indices (0, 1, 2, ...) for evenly spaced bars
    n_scenarios = len(scenarios)
    bar_width = 0.8 / max(n_scenarios, 1)  # Total width 0.8, divided among scenarios

    if n_scenarios == 1:
        offsets = [0]
    else:
        # Center the bars around each index position
        total_group_width = bar_width * n_scenarios
        offsets = np.linspace(
            -total_group_width / 2 + bar_width / 2,
            total_group_width / 2 - bar_width / 2,
            n_scenarios
        )

    for idx, material in enumerate(materials):
        ax = axes_flat[idx]

        # Filter data for this material
        mat_data = allocated_demand[allocated_demand['material'] == material]

        if mat_data.empty:
            ax.text(0.5, 0.5, f"No data\nfor {material}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(material)
            continue

        # Plot bars for each scenario
        for scenario_idx, scenario in enumerate(scenarios):
            scenario_data = mat_data[mat_data['scenario'] == scenario]

            if scenario_data.empty:
                continue

            # Pivot to get CRC as columns, years as rows
            pivot = scenario_data.pivot_table(
                index='year',
                columns='crc',
                values='demand_allocated',
                fill_value=0
            )

            # Ensure all CRC categories present
            for crc in crc_stack_order:
                if crc not in pivot.columns:
                    pivot[crc] = 0

            # Reorder columns to match stacking order
            pivot = pivot[crc_stack_order]

            # Convert years to index positions for proper spacing
            pivot_years = pivot.index.values
            x_indices = np.array([year_to_idx[y] for y in pivot_years])
            x_positions = x_indices + offsets[scenario_idx]

            # Create stacked bars
            bottom = np.zeros(len(pivot_years))
            for crc in crc_stack_order:
                values = pivot[crc].values
                ax.bar(
                    x_positions,
                    values,
                    bar_width,
                    bottom=bottom,
                    color=colors[crc],
                    label=crc if scenario_idx == 0 else None,  # Only label once
                    alpha=0.9
                )
                bottom += values

        # Add USGS reference lines (spanning full index range)
        usgs_mat = usgs_comparison[usgs_comparison['material'] == material]
        if not usgs_mat.empty:
            row = usgs_mat.iloc[0]
            x_range = [-0.5, n_years - 0.5]

            if pd.notna(row['avg_production']):
                ax.plot(x_range, [row['avg_production']] * 2,
                       'k-', linewidth=1.5, label='Avg Production' if idx == 0 else None)

            if pd.notna(row['avg_consumption']):
                ax.plot(x_range, [row['avg_consumption']] * 2,
                       'k--', linewidth=1.5, label='Avg Consumption' if idx == 0 else None)

            if pd.notna(row['avg_net_import']):
                ax.plot(x_range, [row['avg_net_import']] * 2,
                       'k:', linewidth=1.5, label='Avg Net Import' if idx == 0 else None)

        # Add overlay scenario line if provided
        if overlay_scenario and overlay_demand is not None:
            overlay_mat = overlay_demand[overlay_demand['material'] == material]
            if not overlay_mat.empty:
                # Sum across CRC categories for total demand per year
                overlay_total = overlay_mat.groupby('year')['demand_mean_total'].first().reset_index()
                overlay_years = overlay_total['year'].values
                overlay_x = np.array([year_to_idx[y] for y in overlay_years if y in year_to_idx])
                overlay_y = overlay_total[overlay_total['year'].isin(year_to_idx.keys())]['demand_mean_total'].values
                if len(overlay_x) > 0:
                    ax.plot(overlay_x, overlay_y, 'b-', linewidth=2.5, marker='o', markersize=3,
                           label=overlay_scenario if idx == 0 else None, zorder=10)

        # Formatting
        ax.set_title(material, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Annual demand (thousand mt)')

        # Set x-axis limits based on indices
        ax.set_xlim(-0.5, n_years - 0.5)

        # Create x-tick labels: show every 5 years or reasonable subset
        if n_years <= 10:
            # Show all years
            tick_indices = list(range(n_years))
            tick_labels = [str(int(y)) for y in years]
        else:
            # Show subset of years (every 5 years approximately)
            tick_indices = []
            tick_labels = []
            for i, year in enumerate(years):
                if year % 5 == 0 or i == 0 or i == n_years - 1:
                    tick_indices.append(i)
                    tick_labels.append(str(int(year)))

        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')

        if yaxis == 'log':
            ax.set_yscale('log')

        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

    # Hide unused subplots
    for idx in range(n_materials, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Create shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()

    # Sort handles/labels to group CRC and reference lines
    crc_handles = []
    crc_labels = []
    ref_handles = []
    ref_labels = []

    for h, l in zip(handles, labels):
        if l in crc_stack_order:
            crc_handles.append(h)
            crc_labels.append(l)
        else:
            ref_handles.append(h)
            ref_labels.append(l)

    # Combine with CRC first
    all_handles = crc_handles + ref_handles
    all_labels = crc_labels + ref_labels

    fig.legend(all_handles, all_labels,
              loc='lower center',
              bbox_to_anchor=(0.5, -0.02),
              ncol=min(7, len(all_labels)),
              frameon=True,
              fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    # Save figure
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"  Saved chart to: {output_path}")

    return fig


# ============================================================================
# 6. CLI & MAIN
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Supply chain risk analysis: Monte Carlo demand vs USGS production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (all scenarios, all materials)
  python supply_chain_risk_analysis.py \\
    --demand_csv outputs/material_demand_by_scenario.csv \\
    --risk_xlsx data/risk_charts_inputs.xlsx

  # Focus on specific scenarios with log scale
  python supply_chain_risk_analysis.py \\
    --demand_csv outputs/material_demand_by_scenario.csv \\
    --risk_xlsx data/risk_charts_inputs.xlsx \\
    --scenarios Mid_Case,Mid_Case_100by2035 \\
    --yaxis log \\
    --outdir outputs/focused_analysis

  # Keep rare earths separate
  python supply_chain_risk_analysis.py \\
    --demand_csv outputs/material_demand_by_scenario.csv \\
    --risk_xlsx data/risk_charts_inputs.xlsx \\
    --no_aggregate_rare_earths
"""
    )

    # Required arguments
    parser.add_argument(
        '--demand_csv',
        type=Path,
        required=True,
        help='Path to material_demand_by_scenario.csv'
    )

    parser.add_argument(
        '--risk_xlsx',
        type=Path,
        required=True,
        help='Path to risk_charts_inputs.xlsx (USGS data)'
    )

    # Optional arguments
    parser.add_argument(
        '--outdir',
        type=Path,
        default=Path('./outputs/figures/supply_chain_risk'),
        help='Output directory (default: ./outputs/figures/supply_chain_risk)'
    )

    parser.add_argument(
        '--unit_scale',
        type=float,
        default=1000.0,
        help='Conversion factor for demand units. Default 1000 = tonnes → kMT'
    )

    parser.add_argument(
        '--scenarios',
        type=str,
        default=None,
        help='Comma-separated scenario names to analyze (default: all)'
    )

    parser.add_argument(
        '--materials',
        type=str,
        default=None,
        help='Comma-separated material names to analyze (default: all)'
    )

    parser.add_argument(
        '--yaxis',
        type=str,
        choices=['linear', 'log'],
        default='linear',
        help='Y-axis scale for plots (default: linear)'
    )

    parser.add_argument(
        '--aggregate_rare_earths',
        action='store_true',
        default=True,
        help='Aggregate rare earths into single category (default: True)'
    )

    parser.add_argument(
        '--no_aggregate_rare_earths',
        action='store_false',
        dest='aggregate_rare_earths',
        help='Keep rare earths separate'
    )

    parser.add_argument(
        '--target_year',
        type=int,
        default=2035,
        help='Year for demand vs production comparison table (default: 2035)'
    )

    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.1,
        help='CRC share validation tolerance in percentage points (default: 0.1)'
    )

    parser.add_argument(
        '--overlay_scenario',
        type=str,
        default=None,
        help='Scenario to overlay as a line (e.g., Mid_Case_100by2035)'
    )

    return parser.parse_args()


def main():
    """
    Main execution flow.

    Steps
    -----
    1. Parse arguments
    2. Load data (demand + USGS)
    3. Prepare CRC shares
    4. Validate CRC shares
    5. Allocate demand by CRC
    6. Prepare USGS comparison data
    7. Create comparison table
    8. Create visualizations
    9. Generate summary report
    10. Save outputs
    """
    try:
        # Parse arguments
        args = parse_arguments()

        # Create output directory
        args.outdir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory: {args.outdir}")

        # Parse scenario and material filters
        scenarios = args.scenarios.split(',') if args.scenarios else None
        materials = args.materials.split(',') if args.materials else None

        # Step 1: Load demand data
        demand_df = load_demand_data(
            args.demand_csv,
            unit_scale=args.unit_scale,
            scenarios=scenarios,
            materials=materials,
            aggregate_rare_earths=args.aggregate_rare_earths
        )

        # Step 2: Load USGS data
        usgs_data = load_usgs_data(args.risk_xlsx)

        # Step 3: Prepare CRC shares
        crc_shares = prepare_crc_shares(
            usgs_data['import_shares'],
            usgs_data['import_dependency'],
            usgs_data['crc']
        )

        # Step 4: Validate CRC shares
        validation_audit = validate_crc_shares(crc_shares, tolerance=args.tolerance)

        # Step 5: Allocate demand by CRC
        allocated_demand = allocate_demand_by_crc(demand_df, crc_shares)

        # Step 6: Prepare USGS comparison data
        usgs_comparison = prepare_usgs_comparison_data(usgs_data['aggregate'])

        # Step 7: Create comparison table
        comparison_table = create_comparison_table(
            allocated_demand,
            usgs_comparison,
            target_year=args.target_year
        )

        # Step 8: Save CSV outputs
        logging.info("Saving output files...")

        allocated_demand.to_csv(
            args.outdir / 'allocated_demand_by_crc.csv',
            index=False
        )
        logging.info(f"  Saved: allocated_demand_by_crc.csv")

        validation_audit.to_csv(
            args.outdir / 'crc_shares_audit.csv',
            index=False
        )
        logging.info(f"  Saved: crc_shares_audit.csv")

        if not comparison_table.empty:
            comparison_table.to_csv(
                args.outdir / 'demand_vs_production_comparison.csv',
                index=False
            )
            logging.info(f"  Saved: demand_vs_production_comparison.csv")

        # Step 9: Load overlay scenario data if specified
        overlay_allocated = None
        if args.overlay_scenario:
            logging.info(f"Loading overlay scenario: {args.overlay_scenario}")
            overlay_demand_df = load_demand_data(
                args.demand_csv,
                unit_scale=args.unit_scale,
                scenarios=[args.overlay_scenario],
                materials=materials,
                aggregate_rare_earths=args.aggregate_rare_earths
            )
            overlay_allocated = allocate_demand_by_crc(overlay_demand_df, crc_shares)
            logging.info(f"  Overlay data loaded: {len(overlay_allocated)} rows")

        # Step 10: Create visualization
        plot_materials = sorted(allocated_demand['material'].unique())
        plot_scenarios = sorted(allocated_demand['scenario'].unique())

        chart_filename = f'risk_analysis_stacked_bars_{args.yaxis}.png'
        create_stacked_bar_chart(
            allocated_demand,
            usgs_comparison,
            plot_materials,
            plot_scenarios,
            yaxis=args.yaxis,
            output_path=args.outdir / chart_filename,
            overlay_scenario=args.overlay_scenario,
            overlay_demand=overlay_allocated
        )

        # Step 11: Generate summary report
        create_summary_report(
            allocated_demand,
            crc_shares,
            usgs_comparison,
            validation_audit,
            comparison_table,
            args.outdir / 'risk_analysis_summary.txt',
            input_files={
                'demand_csv': args.demand_csv,
                'risk_xlsx': args.risk_xlsx
            },
            config={
                'unit_scale': args.unit_scale,
                'aggregate_rare_earths': args.aggregate_rare_earths,
                'target_year': args.target_year,
                'tolerance': args.tolerance
            }
        )

        logging.info("\n" + "=" * 80)
        logging.info("ANALYSIS COMPLETE")
        logging.info("=" * 80)
        logging.info(f"All outputs saved to: {args.outdir}")
        logging.info("=" * 80)

    except Exception as e:
        logging.error(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
