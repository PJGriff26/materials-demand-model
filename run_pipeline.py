#!/usr/bin/env python3
"""
Materials Demand Model — Full Reproducibility Pipeline
======================================================

Runs the complete analysis pipeline from raw data to publication figures.
Execute from the repository root:

    python run_pipeline.py            # Run everything
    python run_pipeline.py --step 1   # Run only Step 1 (simulation)
    python run_pipeline.py --from 3   # Resume from Step 3
    python run_pipeline.py --skip-simulation --from 2  # Skip MC, start at step 2

Steps:
    1.  Monte Carlo simulation (10,000 iterations) → demand projections
    2.  Sensitivity analysis → variance decomposition, Spearman correlations
    3.  Dimensionality reduction (PCA, Sparse PCA, NMF comparison)
    4.  Sparse PCA interpretation → named components, quadrant plots
    5.  K-means clustering (production pipeline, SPCA input)
    6.  4-method clustering comparison (VIF / PCA / SPCA / FA)
    7.  Supply chain risk analysis → CRC sourcing, reserve adequacy
    8.  Manuscript figures → Fig. 2, SI capacity/additions/intensity
    9.  Manuscript Figure 1 → demand curves + cumulative demand
    10. Exploratory figures → scatterplots, correlation heatmaps

Total runtime: ~15–25 minutes (dominated by Step 1).

Author: Materials Demand Research Team
Date: February 2026
"""

import subprocess
import sys
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_script(script_path, description, step_num):
    """Run a Python script as a subprocess and report timing."""
    rel = script_path.relative_to(ROOT)
    print(f"\n{'─' * 70}")
    print(f"  Step {step_num}: {description}")
    print(f"  Script: {rel}")
    print(f"{'─' * 70}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  ✗ FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        print(f"    Re-run with: python {rel}")
        sys.exit(result.returncode)

    print(f"\n  ✓ Done ({elapsed:.1f}s)")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run the full materials demand analysis pipeline."
    )
    parser.add_argument(
        "--step", type=int, default=None,
        help="Run only this step number (1–10)."
    )
    parser.add_argument(
        "--from", dest="from_step", type=int, default=1,
        help="Start from this step number (default: 1)."
    )
    parser.add_argument(
        "--skip-simulation", action="store_true",
        help="Skip Step 1 (Monte Carlo simulation) if outputs already exist."
    )
    args = parser.parse_args()

    # Define the pipeline
    steps = [
        (1, ROOT / "examples" / "run_simulation.py",
         "Monte Carlo simulation (10,000 iterations)"),
        (2, ROOT / "analysis" / "sensitivity_analysis.py",
         "Sensitivity analysis (variance decomposition + Spearman)"),
        (3, ROOT / "clustering" / "sparse_nmf_analysis.py",
         "Dimensionality reduction (PCA, Sparse PCA, NMF comparison)"),
        (4, ROOT / "clustering" / "sparse_pca_story.py",
         "Sparse PCA interpretation (named components, quadrants)"),
        (5, ROOT / "clustering" / "main_analysis.py",
         "Production clustering (SPCA → K-means)"),
        (6, ROOT / "clustering" / "clustering_comparison.py",
         "4-method clustering comparison (VIF / PCA / SPCA / FA)"),
        (7, ROOT / "clustering" / "supply_chain_analysis.py",
         "Supply chain risk analysis (CRC sourcing, reserve adequacy)"),
        (8, ROOT / "visualizations" / "manuscript_figures.py",
         "Manuscript figures (Fig. 2, SI capacity/additions/intensity)"),
        (9, ROOT / "visualizations" / "manuscript_fig1.py",
         "Manuscript Figure 1 (demand curves + cumulative demand)"),
        (10, ROOT / "visualizations" / "feature_scatterplots.py",
          "Exploratory figures (scatterplots, correlation heatmaps)"),
    ]

    # Filter steps
    if args.step is not None:
        steps = [(n, p, d) for n, p, d in steps if n == args.step]
        if not steps:
            print(f"Error: no step {args.step}. Valid range: 1–10.")
            sys.exit(1)
    else:
        steps = [(n, p, d) for n, p, d in steps if n >= args.from_step]

    # Check for skip-simulation
    if args.skip_simulation:
        demand_file = ROOT / "outputs" / "material_demand_by_scenario.csv"
        if demand_file.exists():
            steps = [(n, p, d) for n, p, d in steps if n != 1]
            print("Skipping Step 1 (simulation outputs already exist).")
        else:
            print("Warning: --skip-simulation ignored; output file not found.")

    # Run
    print("=" * 70)
    print("  MATERIALS DEMAND MODEL — FULL PIPELINE")
    print(f"  Steps to run: {[n for n, _, _ in steps]}")
    print("=" * 70)

    total = 0
    for step_num, script_path, description in steps:
        elapsed = run_script(script_path, description, step_num)
        total += elapsed

    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE — Total time: {total / 60:.1f} minutes")
    print(f"{'=' * 70}")
    print(f"\nOutputs:")
    print(f"  Data:    outputs/data/")
    print(f"  Figures: outputs/figures/")
    print(f"  Reports: outputs/reports/")


if __name__ == "__main__":
    main()
