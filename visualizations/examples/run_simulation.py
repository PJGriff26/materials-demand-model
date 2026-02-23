"""
Demo: Stock-Flow Monte Carlo Simulation for Materials Demand
=============================================================

This script demonstrates the complete workflow for running a stock-flow
Monte Carlo simulation to estimate material demand uncertainty.

Author: Materials Demand Research Team
Date: 2024
Updated: January 2026 - Added full 95% CI support (p2.5, p97.5)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our modules
from src.stock_flow_simulation import (
    run_full_simulation,
    MaterialsStockFlowSimulation
)


# ============================================================================
# CONFIGURATION - Edit these paths for your system
# ============================================================================

# Input data directory
DATA_DIR = Path(__file__).parent.parent / 'data'

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Input files
INTENSITY_FILE = DATA_DIR / 'intensity_data.csv'
CAPACITY_FILE = DATA_DIR / 'StdScen24_annual_national.csv'

# Output files
DETAILED_RESULTS_FILE = OUTPUT_DIR / 'material_demand_by_scenario.csv'
SUMMARY_RESULTS_FILE = OUTPUT_DIR / 'material_demand_summary.csv'
REPORT_FILE = OUTPUT_DIR / 'simulation_report.txt'

# Simulation parameters
N_ITERATIONS = 10000 
RANDOM_SEED = 42     # For reproducibility

# Focus scenarios (None = all scenarios)
FOCUS_SCENARIOS = None  # or ['Mid_Case', 'Mid_Case_No_IRA', 'Mid_Case_100by2035']

# ============================================================================


def main():
    """Run the complete simulation workflow"""
    
    print("="*80)
    print("MATERIALS DEMAND STOCK-FLOW MONTE CARLO SIMULATION")
    print("="*80)
    print()
    
    print(f"Configuration:")
    print(f"  Intensity data: {INTENSITY_FILE}")
    print(f"  Capacity data: {CAPACITY_FILE}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Iterations: {N_ITERATIONS:,}")
    print(f"  Random seed: {RANDOM_SEED}")
    if FOCUS_SCENARIOS:
        print(f"  Focus scenarios: {FOCUS_SCENARIOS}")
    print()
    
    # Verify files exist
    if not INTENSITY_FILE.exists():
        print(f"ERROR: Cannot find {INTENSITY_FILE}")
        sys.exit(1)
    if not CAPACITY_FILE.exists():
        print(f"ERROR: Cannot find {CAPACITY_FILE}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 1: RUN SIMULATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("RUNNING SIMULATION")
    print("="*80)
    print()
    
    simulation, results = run_full_simulation(
        intensity_path=INTENSITY_FILE,
        capacity_path=CAPACITY_FILE,
        n_iterations=N_ITERATIONS,
        scenarios_to_run=FOCUS_SCENARIOS,
        random_state=RANDOM_SEED
    )
    
    # ========================================================================
    # STEP 2: EXTRACT STATISTICS
    # ========================================================================
    
    print("\n" + "="*80)
    print("EXTRACTING STATISTICS")
    print("="*80)
    print()
    
    # Detailed results (by scenario) - NOW INCLUDES 2.5 and 97.5 percentiles
    print("Calculating detailed statistics (by scenario)...")
    detailed_stats = results.get_statistics(percentiles=[2.5, 5, 25, 50, 75, 95, 97.5])
    
    # Debug: Check what columns were actually created
    print(f"Available columns: {detailed_stats.columns.tolist()}")
    
    # Summary results (aggregated across scenarios) - NOW INCLUDES 2.5 and 97.5 percentiles
    print("Calculating summary statistics (across scenarios)...")
    summary_stats = results.get_summary_by_material(percentiles=[2.5, 5, 25, 50, 75, 95, 97.5])
    
    print(f"\n✓ Statistics calculated")
    print(f"  Detailed results: {len(detailed_stats):,} rows")
    print(f"  Summary results: {len(summary_stats):,} rows")
    print(f"  Percentiles requested: 2.5, 5, 25, 50, 75, 95, 97.5")
    
    # Determine the actual column names for p2.5 and p97.5
    # They might be 'p2.5', 'p2_5', '2.5', etc.
    p2_5_col = None
    p97_5_col = None
    
    for col in detailed_stats.columns:
        col_str = str(col).lower()
        if '2.5' in col_str or '2_5' in col_str:
            p2_5_col = col
        elif '97.5' in col_str or '97_5' in col_str:
            p97_5_col = col
    
    if not p2_5_col or not p97_5_col:
        print(f"\nWARNING: Could not find p2.5 or p97.5 columns.")
        print(f"Available columns: {[c for c in detailed_stats.columns if c.startswith('p') or str(c).replace('.','').replace('_','').isdigit()]}")
        print(f"Will use p5 and p95 instead for confidence intervals.")
        p2_5_col = 'p5'
        p97_5_col = 'p95'
    else:
        print(f"  95% CI columns found: {p2_5_col} to {p97_5_col}")
    
    # ========================================================================
    # STEP 3: PREVIEW RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("PREVIEW: KEY RESULTS")
    print("="*80)
    print()
    
    # Show top materials for 2035 across all scenarios
    print("Top 10 materials by median demand (2035, all scenarios combined):")
    print("-" * 80)
    
    summary_2035 = summary_stats[summary_stats['year'] == 2035].copy()
    summary_2035_sorted = summary_2035.sort_values('p50', ascending=False).head(10)
    
    for idx, row in summary_2035_sorted.iterrows():
        ci_text = f"(95% CI: [{row[p2_5_col]:>10,.0f}, {row[p97_5_col]:>10,.0f}])"
        print(f"{row['material']:20s}: {row['p50']:>12,.0f} mt {ci_text}")
    
    # Show example scenario comparison for Copper
    print("\n" + "-" * 80)
    print("Copper demand by scenario (2035, median with 95% CI):")
    print("-" * 80)
    
    copper_2035 = detailed_stats[
        (detailed_stats['material'] == 'Copper') &
        (detailed_stats['year'] == 2035)
    ].copy()
    
    if len(copper_2035) > 0:
        copper_sorted = copper_2035.sort_values('p50', ascending=False).head(10)
        for idx, row in copper_sorted.iterrows():
            ci_text = f"(95% CI: [{row[p2_5_col]:>10,.0f}, {row[p97_5_col]:>10,.0f}])"
            print(f"{row['scenario']:30s}: {row['p50']:>12,.0f} mt {ci_text}")
    else:
        print("No copper data available")
    
    # ========================================================================
    # STEP 4: EXPORT RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80)
    print()
    
    # Export detailed results
    detailed_stats.to_csv(DETAILED_RESULTS_FILE, index=False)
    print(f"✓ Detailed results saved: {DETAILED_RESULTS_FILE}")
    print(f"  95% CI columns: '{p2_5_col}' to '{p97_5_col}'")
    
    # Export summary results
    summary_stats.to_csv(SUMMARY_RESULTS_FILE, index=False)
    print(f"✓ Summary results saved: {SUMMARY_RESULTS_FILE}")
    print(f"  95% CI columns: '{p2_5_col}' to '{p97_5_col}'")
    
    # Create detailed report
    create_report(
        simulation=simulation,
        results=results,
        detailed_stats=detailed_stats,
        summary_stats=summary_stats,
        output_path=REPORT_FILE,
        p2_5_col=p2_5_col,
        p97_5_col=p97_5_col
    )
    print(f"✓ Detailed report saved: {REPORT_FILE}")
    
    # ========================================================================
    # STEP 5: SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print()
    print("Key Outputs:")
    print(f"  1. {DETAILED_RESULTS_FILE}")
    print(f"     - Material demand by scenario, year, material")
    print(f"     - Includes percentiles: p2.5, p5, p25, p50, p75, p95, p97.5")
    print(f"     - 95% confidence intervals available (p2.5 to p97.5)")
    print(f"     - {len(detailed_stats):,} rows")
    print()
    print(f"  2. {SUMMARY_RESULTS_FILE}")
    print(f"     - Material demand aggregated across scenarios")
    print(f"     - Useful for overall material requirements")
    print(f"     - {len(summary_stats):,} rows")
    print()
    print(f"  3. {REPORT_FILE}")
    print(f"     - Comprehensive text report with analysis")
    print()
    print("Next Steps:")
    print("  - Review the report for quality assessment")
    print("  - Analyze scenario comparisons in detailed results")
    print("  - Visualize uncertainty bands using 95% CI (p2.5-p97.5)")
    print("  - Compare IRA vs No-IRA scenarios")
    print()


def create_report(
    simulation: MaterialsStockFlowSimulation,
    results,
    detailed_stats: pd.DataFrame,
    summary_stats: pd.DataFrame,
    output_path: Path,
    p2_5_col: str,
    p97_5_col: str
):
    """Create comprehensive text report"""
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MATERIALS DEMAND STOCK-FLOW MONTE CARLO SIMULATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Simulation parameters
        f.write("SIMULATION PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Monte Carlo iterations: {results.n_iterations:,}\n")
        f.write(f"Scenarios analyzed: {len(results.scenarios)}\n")
        f.write(f"Years: {min(results.years)} - {max(results.years)}\n")
        f.write(f"Materials tracked: {len(results.materials)}\n")
        f.write(f"Random seed: {simulation.random_state}\n")
        f.write(f"Percentiles reported: 2.5, 5, 25, 50 (median), 75, 95, 97.5\n")
        f.write(f"  95% CI: p2.5 to p97.5\n")
        f.write(f"  90% CI: p5 to p95\n")
        f.write("\n")
        
        # Model description
        f.write("MODEL DESCRIPTION\n")
        f.write("-" * 80 + "\n")
        f.write("Stock-Flow Model:\n")
        f.write("  - Tracks installed capacity stock over time\n")
        f.write("  - Calculates annual capacity additions\n")
        f.write("  - Models retirements based on technology-specific lifetimes\n")
        f.write("  - Material demand = capacity additions × material intensity\n")
        f.write("\n")
        f.write("Uncertainty Quantification:\n")
        f.write("  - Monte Carlo sampling from fitted distributions\n")
        f.write("  - Material intensity uncertainty propagated through model\n")
        f.write("  - Results reported as percentiles (2.5, 5, 25, 50, 75, 95, 97.5)\n")
        f.write("  - 95% confidence intervals: [p2.5, p97.5]\n")
        f.write("  - 90% confidence intervals: [p5, p95]\n")
        f.write("\n")
        
        # Technology mapping summary
        f.write("TECHNOLOGY MAPPING\n")
        f.write("-" * 80 + "\n")
        mapped_count = sum(1 for v in simulation.technology_mapping.values() if v)
        f.write(f"Capacity technologies: {len(simulation.capacity_tech_cols)}\n")
        f.write(f"  Mapped to intensity data: {mapped_count}\n")
        f.write(f"  Unmapped (skipped): {len(simulation.capacity_tech_cols) - mapped_count}\n")
        f.write("\n")
        
        # Key results
        f.write("KEY RESULTS\n")
        f.write("-" * 80 + "\n\n")
        
        # Top 10 materials in 2035
        f.write("Top 10 Materials by Median Demand (2035, All Scenarios)\n")
        f.write("-" * 80 + "\n")
        summary_2035 = summary_stats[summary_stats['year'] == 2035].copy()
        summary_2035_sorted = summary_2035.sort_values('p50', ascending=False).head(10)
        
        f.write(f"{'Material':<20} {'Median (mt)':>15} {'95% CI Lower':>15} {'95% CI Upper':>15}\n")
        f.write("-" * 80 + "\n")
        for _, row in summary_2035_sorted.iterrows():
            f.write(f"{row['material']:<20} {row['p50']:>15,.0f} "
                   f"{row[p2_5_col]:>15,.0f} {row[p97_5_col]:>15,.0f}\n")
        f.write("\n")
        
        # Material demand over time (key materials)
        key_materials = ['Copper', 'Aluminum', 'Steel', 'Lithium', 'Silicon']
        f.write("\nMaterial Demand Over Time (Median, All Scenarios)\n")
        f.write("-" * 80 + "\n")
        
        for material in key_materials:
            mat_data = summary_stats[summary_stats['material'] == material].copy()
            if len(mat_data) == 0:
                continue
            
            f.write(f"\n{material}:\n")
            mat_data_sorted = mat_data.sort_values('year')
            f.write(f"  {'Year':<6} {'Median (mt)':>15} {'95% CI':>30}\n")
            for _, row in mat_data_sorted.iterrows():
                ci = f"[{row[p2_5_col]:,.0f}, {row[p97_5_col]:,.0f}]"
                f.write(f"  {row['year']:<6} {row['p50']:>15,.0f} {ci:>30}\n")
        
        f.write("\n")
        
        # Scenario comparison for key year
        f.write("\n" + "="*80 + "\n")
        f.write("SCENARIO COMPARISON (2035, Copper)\n")
        f.write("="*80 + "\n\n")
        
        copper_2035 = detailed_stats[
            (detailed_stats['material'] == 'Copper') &
            (detailed_stats['year'] == 2035)
        ].copy()
        
        if len(copper_2035) > 0:
            copper_sorted = copper_2035.sort_values('p50', ascending=False)
            f.write(f"{'Scenario':<35} {'Median (mt)':>15} {'Std Dev':>15} {'95% CI':>30}\n")
            f.write("-" * 80 + "\n")
            for _, row in copper_sorted.iterrows():
                ci = f"[{row[p2_5_col]:,.0f}, {row[p97_5_col]:,.0f}]"
                f.write(f"{row['scenario']:<35} {row['p50']:>15,.0f} "
                       f"{row['std']:>15,.0f} {ci:>30}\n")
        else:
            f.write("No copper data available\n")
        
        f.write("\n")
        
        # Data quality notes
        f.write("\n" + "="*80 + "\n")
        f.write("DATA QUALITY & LIMITATIONS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Material Intensity Data:\n")
        f.write("  - Source: NREL material intensity database\n")
        f.write("  - Uncertainty captured via parametric distributions\n")
        f.write("  - Small sample sizes use empirical distributions (bootstrap)\n")
        f.write("\n")
        
        f.write("Capacity Projections:\n")
        f.write("  - Source: NREL Standard Scenarios\n")
        f.write("  - Years: 3-year intervals (2026, 2029, 2032, etc.)\n")
        f.write("  - Linear interpolation between data points\n")
        f.write("\n")
        
        f.write("Model Limitations:\n")
        f.write("  - Technology mapping based on expert judgment\n")
        f.write("  - Some technologies lack intensity data (batteries, DAC, etc.)\n")
        f.write("  - No material recovery/recycling modeled\n")
        f.write("  - Assumes constant material intensity over time\n")
        f.write("\n")
        
        # Statistics
        f.write("\n" + "="*80 + "\n")
        f.write("STATISTICAL SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total data points: {len(detailed_stats):,}\n")
        f.write(f"  Scenarios: {len(results.scenarios)}\n")
        f.write(f"  Years: {len(results.years)}\n")
        f.write(f"  Materials: {len(results.materials)}\n")
        f.write("\n")
        
        f.write(f"Monte Carlo iterations: {results.n_iterations:,}\n")
        f.write(f"Percentiles reported: 2.5, 5, 25, 50 (median), 75, 95, 97.5\n")
        f.write(f"Confidence intervals:\n")
        f.write(f"  - 95% CI: [p2.5, p97.5] (recommended for publication)\n")
        f.write(f"  - 90% CI: [p5, p95] (alternative)\n")
        f.write(f"  - IQR: [p25, p75] (interquartile range)\n")
        f.write("\n")
        
        f.write("End of Report\n")
        f.write("="*80 + "\n")


if __name__ == '__main__':
    main()