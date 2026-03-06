#!/usr/bin/env python3
"""
Unified Manuscript Figure Generation
=====================================

Single script to regenerate all figures that appear in the manuscript.
Produces main text figures (fig1-fig5) and supplementary figures (si_fig_s1-s21).

Source modules:
    visualizations/manuscript_fig1.py     — demand curve figures
    visualizations/manuscript_figures.py  — tech breakdown, capacity, intensity
    analysis/sensitivity_analysis.py      — variance decomposition, sensitivity
    clustering/visualization.py           — PCA biplots, silhouette, profiles
    clustering/supply_chain_analysis.py   — supply risk matrices, CRC sourcing

Usage:
    python visualizations/generate_manuscript_figures.py                # all figures
    python visualizations/generate_manuscript_figures.py --only fig1    # single figure
    python visualizations/generate_manuscript_figures.py --only fig1 fig3 si_s13  # subset
    python visualizations/generate_manuscript_figures.py --list         # list available figures
    python visualizations/generate_manuscript_figures.py --output-dir /path/to/dir  # custom output

Output: outputs/figures/manuscript/  (main text + si_figures/ subdirectory)
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent    # materials_demand_model/

MAIN_FIG_DIR = BASE_DIR / "outputs" / "figures" / "manuscript"
SI_FIG_DIR = MAIN_FIG_DIR / "si_figures"

# Module search paths
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "clustering"))

# ── Data paths ────────────────────────────────────────────────────────────────
DEMAND_CSV       = BASE_DIR / "outputs" / "data" / "material_demand_by_scenario.csv"
TECH_CONTRIB_CSV = BASE_DIR / "outputs" / "data" / "sensitivity" / "technology_contributions.csv"
NREL_CSV         = BASE_DIR / "data" / "StdScen24_annual_national.csv"
INTENSITY_CSV    = BASE_DIR / "data" / "intensity_data.csv"

# Clustering pre-computed outputs
CLUSTER_DIR      = BASE_DIR / "outputs" / "data" / "clustering"
SPCA_MAT_CSV     = CLUSTER_DIR / "spca_scores_materials.csv"
SPCA_SCEN_CSV    = CLUSTER_DIR / "spca_scores_scenarios.csv"
MAT_CLUSTERS_CSV = CLUSTER_DIR / "material_clusters.csv"
SCEN_CLUSTERS_CSV = CLUSTER_DIR / "scenario_clusters.csv"
MAT_PROFILES_CSV = CLUSTER_DIR / "material_cluster_profiles.csv"
SCEN_PROFILES_CSV = CLUSTER_DIR / "scenario_cluster_profiles.csv"
STRESS_MATRIX_CSV = CLUSTER_DIR / "stress_matrix.csv"

# Sensitivity pre-computed outputs
SENS_DIR         = BASE_DIR / "outputs" / "data" / "sensitivity"
VAR_DECOMP_CSV   = SENS_DIR / "variance_decomposition.csv"
VAR_BY_YEAR_CSV  = SENS_DIR / "variance_decomposition_by_year.csv"
ELASTICITY_CSV   = SENS_DIR / "intensity_elasticity.csv"
SPEARMAN_CSV     = SENS_DIR / "spearman_sensitivity.csv"

# Default output dirs (where clustering/supply-chain functions save)
CLUSTERING_FIG_DIR = BASE_DIR / "outputs" / "figures" / "clustering" / "kmeans"
MANUSCRIPT_FIG_DIR = BASE_DIR / "outputs" / "figures" / "manuscript"

# ── Constants ─────────────────────────────────────────────────────────────────
KEY_MATERIALS = [
    "Copper", "Steel", "Aluminum", "Cement", "Silicon",
    "Neodymium", "Dysprosium", "Glass", "Fiberglass",
    "Nickel", "Manganese", "Chromium",
]

SENSITIVITY_MATERIALS = [
    "Copper", "Steel", "Aluminum", "Silicon", "Neodymium",
    "Cement", "Nickel", "Chromium", "Dysprosium",
]

INTENSITY_MATERIALS = {
    "si_s15": ("Steel",      "si_fig_s15_intensity_steel.png"),
    "si_s16": ("Copper",     "si_fig_s16_intensity_copper.png"),
    "si_s17": ("Silicon",    "si_fig_s17_intensity_silicon.png"),
    "si_s18": ("Neodymium",  "si_fig_s18_intensity_neodymium.png"),
    "si_s19": ("Aluminum",   "si_fig_s19_intensity_aluminum.png"),
    "si_s20": ("Dysprosium", "si_fig_s20_intensity_dysprosium.png"),
}

# Interpretive cluster labels (from main_analysis.py, correspond to latest run)
SCEN_CLUSTER_NAMES = {
    0: "Volatile / peak-and-decline",
    1: "Aggressive decarbonization",
    2: "Moderate / sustained growth",
}
MAT_CLUSTER_NAMES = {
    0: "Base metals (OECD-sourced)",
    1: "Bulk industrial",
    2: "Import-dependent specialty",
    3: "Low-volume, concentrated-source",
    4: "Neodymium (REE outlier)",
    5: "REE permanent magnet elements",
}

# SPCA component labels (from main_analysis.py)
SCEN_COMPONENT_LABELS = {
    "SPC1": "Demand Scale & Wind",
    "SPC2": "Demand Uncertainty",
    "SPC3": "Supply Chain Stress",
    "SPC4": "Solar & Storage Mix",
}
MAT_COMPONENT_LABELS = {
    "SPC1": "Demand Scale",
    "SPC2": "Geopolitical Risk",
    "SPC3": "Reserve Geography",
    "SPC4": "Import Concentration",
    "SPC5": "Capacity Stress",
}


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_all_data():
    """Load all datasets, skipping any that don't exist."""
    data = {}

    def _load_csv(key, path, **kwargs):
        if path.exists():
            data[key] = pd.read_csv(path, **kwargs)
            print(f"    {key}: {len(data[key]):,} rows")
        else:
            print(f"    {key}: NOT FOUND ({path.name})")

    print("  Loading data files...")
    _load_csv("demand", DEMAND_CSV)
    _load_csv("tech_contrib", TECH_CONTRIB_CSV)
    _load_csv("nrel", NREL_CSV, skiprows=3)
    _load_csv("intensity", INTENSITY_CSV)

    # Clustering outputs
    _load_csv("spca_materials", SPCA_MAT_CSV, index_col=0)
    _load_csv("spca_scenarios", SPCA_SCEN_CSV, index_col=0)
    _load_csv("material_clusters", MAT_CLUSTERS_CSV)
    _load_csv("scenario_clusters", SCEN_CLUSTERS_CSV)
    _load_csv("mat_profiles", MAT_PROFILES_CSV, index_col=0)
    _load_csv("scen_profiles", SCEN_PROFILES_CSV, index_col=0)
    _load_csv("stress_matrix", STRESS_MATRIX_CSV, index_col=0)

    # Sensitivity outputs
    _load_csv("var_decomp", VAR_DECOMP_CSV)
    _load_csv("var_by_year", VAR_BY_YEAR_CSV)
    _load_csv("elasticity", ELASTICITY_CSV)
    _load_csv("spearman", SPEARMAN_CSV)

    # Apply SPCA component labels
    if "spca_materials" in data:
        data["spca_materials"].columns = [
            MAT_COMPONENT_LABELS.get(c, c) for c in data["spca_materials"].columns
        ]
    if "spca_scenarios" in data:
        data["spca_scenarios"].columns = [
            SCEN_COMPONENT_LABELS.get(c, c) for c in data["spca_scenarios"].columns
        ]

    # Supply chain risk data (Excel)
    if (BASE_DIR / "data" / "supply_chain" / "risk_charts_inputs.xlsx").exists():
        try:
            from supply_chain_analysis import load_data as _load_sc
            sc_demand, sc_risk = _load_sc()
            data["sc_demand"] = sc_demand
            data["sc_risk"] = sc_risk
            print(f"    supply_chain: loaded ({sc_demand['material'].nunique()} materials)")
        except Exception as e:
            print(f"    supply_chain: FAILED ({e})")
    else:
        print("    supply_chain: NOT FOUND (risk_charts_inputs.xlsx)")

    return data


def _require(data, *keys):
    """Check that all required data keys are present."""
    missing = [k for k in keys if k not in data]
    if missing:
        raise KeyError(f"Missing data: {', '.join(missing)}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN TEXT FIGURES
# ═════════════════════════════════════════════════════════════════════════════

def gen_fig1(data):
    """Fig 1: Annual material demand with uncertainty bands (12 panels)."""
    _require(data, "demand")
    from visualizations.manuscript_fig1 import create_fig1_demand_curves
    fig = create_fig1_demand_curves(
        data["demand"],
        materials=KEY_MATERIALS,
        output_path=MAIN_FIG_DIR / "fig1_demand_projections.png",
    )
    plt.close(fig)


def gen_fig2(data):
    """Fig 2: Technology decomposition (100% stacked bar)."""
    _require(data, "tech_contrib")
    from visualizations.manuscript_figures import create_technology_breakdown_chart
    fig, _ = create_technology_breakdown_chart(
        data["tech_contrib"],
        output_path=MAIN_FIG_DIR / "fig2_technology_decomposition.png",
    )
    plt.close(fig)


def gen_fig3(data):
    """Fig 3: Variance decomposition (intensity vs scenario)."""
    from analysis.sensitivity_analysis import (
        compute_variance_decomposition, plot_variance_decomposition,
    )
    # Use pre-computed if available, otherwise compute from demand
    if "var_decomp" in data:
        vd = data["var_decomp"]
    else:
        _require(data, "demand")
        vd = compute_variance_decomposition(data["demand"])
    fig = plot_variance_decomposition(
        vd, output_path=MAIN_FIG_DIR / "fig3_variance_decomposition.png",
    )
    plt.close(fig)


def gen_fig4(data):
    """Fig 4: Material clusters PCA biplot."""
    _require(data, "spca_materials", "material_clusters")
    import visualization as cluster_viz

    X_mat = data["spca_materials"]
    labels = data["material_clusters"]["cluster"].values

    # Redirect output and call
    cluster_viz.FIGURES_KMEANS_DIR = MAIN_FIG_DIR
    cluster_viz.plot_pca_biplot_centroid_labels(
        X_mat.values, labels, list(X_mat.columns),
        "materials",
        entity_names=list(X_mat.index),
        cluster_names=MAT_CLUSTER_NAMES,
        raw_features=X_mat,
    )
    # Rename from default stem to manuscript filename
    src = MAIN_FIG_DIR / "pca_biplot_materials_centroid_labels.png"
    dst = MAIN_FIG_DIR / "fig4_material_clusters.png"
    if src.exists():
        dst.unlink(missing_ok=True)
        src.rename(dst)


def gen_fig5a_risk(data):
    """Fig 5a: Global supply risk matrix (scatter)."""
    _require(data, "sc_demand", "sc_risk")
    import supply_chain_analysis as sca

    sca.FIGURES_MANUSCRIPT_DIR = MAIN_FIG_DIR
    sca.fig3_supply_risk_matrix(data["sc_demand"], data["sc_risk"])
    src = MAIN_FIG_DIR / "fig3_supply_risk_matrix.png"
    dst = MAIN_FIG_DIR / "fig5a_supply_risk_matrix.png"
    if src.exists():
        dst.unlink(missing_ok=True)
        src.rename(dst)


def gen_fig5b_risk(data):
    """Fig 5b: US supply risk matrix (scatter)."""
    _require(data, "sc_demand", "sc_risk")
    import supply_chain_analysis as sca

    sca.FIGURES_MANUSCRIPT_DIR = MAIN_FIG_DIR
    sca.fig3b_us_supply_risk_matrix(data["sc_demand"], data["sc_risk"])
    src = MAIN_FIG_DIR / "fig3b_us_supply_risk_matrix.png"
    dst = MAIN_FIG_DIR / "fig5b_us_supply_risk_matrix.png"
    if src.exists():
        dst.unlink(missing_ok=True)
        src.rename(dst)


def gen_fig5a_sourcing(data):
    """Fig 5a alt: Demand by CRC sourcing category."""
    _require(data, "sc_demand", "sc_risk")
    import supply_chain_analysis as sca

    sca.FIGURES_MANUSCRIPT_DIR = MAIN_FIG_DIR
    sca.fig3_demand_sourcing(data["sc_demand"], data["sc_risk"])
    src = MAIN_FIG_DIR / "fig3_demand_sourcing.png"
    dst = MAIN_FIG_DIR / "fig5a_supply_chain_sourcing.png"
    if src.exists():
        dst.unlink(missing_ok=True)
        src.rename(dst)


def gen_fig5b_reserves(data):
    """Fig 5b alt: Reserve adequacy by CRC category."""
    _require(data, "sc_demand", "sc_risk")
    import supply_chain_analysis as sca

    sca.FIGURES_MANUSCRIPT_DIR = MAIN_FIG_DIR
    sca.fig4_reserve_adequacy(data["sc_demand"], data["sc_risk"])
    src = MAIN_FIG_DIR / "fig4_reserve_adequacy.png"
    dst = MAIN_FIG_DIR / "fig5b_reserve_adequacy.png"
    if src.exists():
        dst.unlink(missing_ok=True)
        src.rename(dst)


def gen_fig5c_reserves_us(data):
    """Fig 5c: US reserve adequacy."""
    _require(data, "sc_demand", "sc_risk")
    import supply_chain_analysis as sca

    sca.FIGURES_MANUSCRIPT_DIR = MAIN_FIG_DIR
    sca.fig4_reserve_adequacy_us(data["sc_demand"], data["sc_risk"])
    src = MAIN_FIG_DIR / "fig4_reserve_adequacy_us.png"
    dst = MAIN_FIG_DIR / "fig5c_reserve_adequacy_us.png"
    if src.exists():
        dst.unlink(missing_ok=True)
        src.rename(dst)


# ═════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURES
# ═════════════════════════════════════════════════════════════════════════════

def gen_si_s1(data):
    """SI Fig S1: Capacity projections (stacked area)."""
    _require(data, "nrel")
    from visualizations.manuscript_figures import create_capacity_projections_chart
    fig, _ = create_capacity_projections_chart(
        data["nrel"],
        scenario="Mid_Case",
        output_path=SI_FIG_DIR / "si_fig_s1_capacity.png",
    )
    if fig is not None:
        plt.close(fig)


def gen_si_s2(data):
    """SI Fig S2: Annual capacity additions (stacked bar)."""
    _require(data, "nrel")
    from visualizations.manuscript_figures import create_capacity_additions_chart
    fig, _ = create_capacity_additions_chart(
        data["nrel"],
        scenario="Mid_Case",
        output_path=SI_FIG_DIR / "si_fig_s2_additions.png",
    )
    if fig is not None:
        plt.close(fig)


def gen_si_s3(data):
    """SI Fig S3: Material intensity distributions (violin + scatter)."""
    _require(data, "intensity")
    from visualizations.manuscript_figures import create_intensity_distributions_chart
    fig = create_intensity_distributions_chart(
        data["intensity"],
        output_path=SI_FIG_DIR / "si_fig_s3_intensity_dists.png",
    )
    plt.close(fig)


def gen_si_s4(data):
    """SI Fig S4: Variance decomposition over time."""
    from analysis.sensitivity_analysis import (
        compute_variance_decomposition_by_year, plot_variance_by_year,
    )
    if "var_by_year" in data:
        vby = data["var_by_year"]
    else:
        _require(data, "demand")
        vby = compute_variance_decomposition_by_year(data["demand"])
    fig = plot_variance_by_year(
        vby,
        materials=SENSITIVITY_MATERIALS,
        output_path=SI_FIG_DIR / "si_fig_s4_variance_by_year.png",
    )
    plt.close(fig)


def gen_si_s5(data):
    """SI Fig S5: Spearman correlation heatmap."""
    _require(data, "spearman")
    from analysis.sensitivity_analysis import plot_spearman_heatmap
    fig = plot_spearman_heatmap(
        data["spearman"],
        output_path=SI_FIG_DIR / "si_fig_s5_spearman_heatmap.png",
    )
    plt.close(fig)


def gen_si_s6(data):
    """SI Fig S6: Spearman correlation tornado."""
    _require(data, "spearman")
    from analysis.sensitivity_analysis import plot_spearman_tornado
    fig = plot_spearman_tornado(
        data["spearman"],
        output_path=SI_FIG_DIR / "si_fig_s6_spearman_tornado.png",
    )
    plt.close(fig)


def gen_si_s7(data):
    """SI Fig S7: Intensity elasticity tornado."""
    from analysis.sensitivity_analysis import (
        compute_intensity_elasticity, plot_tornado_diagram,
    )
    if "elasticity" in data:
        el = data["elasticity"]
    else:
        _require(data, "tech_contrib", "demand")
        el = compute_intensity_elasticity(data["tech_contrib"], data["demand"])
    fig = plot_tornado_diagram(
        el, output_path=SI_FIG_DIR / "si_fig_s7_tornado_elasticity.png",
    )
    plt.close(fig)


def gen_si_s8(data):
    """SI Fig S8: Scenario clusters PCA biplot."""
    _require(data, "spca_scenarios", "scenario_clusters")
    import visualization as cluster_viz

    X_scen = data["spca_scenarios"]
    labels = data["scenario_clusters"]["cluster"].values

    cluster_viz.FIGURES_KMEANS_DIR = SI_FIG_DIR
    cluster_viz.plot_pca_biplot_centroid_labels(
        X_scen.values, labels, list(X_scen.columns),
        "scenarios",
        entity_names=list(X_scen.index),
        cluster_names=SCEN_CLUSTER_NAMES,
        raw_features=X_scen,
    )
    src = SI_FIG_DIR / "pca_biplot_scenarios_centroid_labels.png"
    dst = SI_FIG_DIR / "si_fig_s8_scenario_clusters.png"
    if src.exists():
        dst.unlink(missing_ok=True)
        src.rename(dst)


def gen_si_s9(data):
    """SI Fig S9: Material cluster profiles heatmap."""
    _require(data, "mat_profiles")
    import visualization as cluster_viz

    cluster_viz.FIGURES_KMEANS_DIR = SI_FIG_DIR
    cluster_viz.plot_cluster_profiles(data["mat_profiles"], "materials")
    src = SI_FIG_DIR / "cluster_profiles_materials.png"
    dst = SI_FIG_DIR / "si_fig_s9_cluster_profiles.png"
    if src.exists():
        dst.unlink(missing_ok=True)
        src.rename(dst)


def gen_si_s10(data):
    """SI Fig S10: Scenario x Material stress matrix heatmap."""
    _require(data, "stress_matrix")
    import visualization as cluster_viz

    cluster_viz.FIGURES_KMEANS_DIR = SI_FIG_DIR
    cluster_viz.plot_stress_matrix(data["stress_matrix"].values, "demand")
    src = SI_FIG_DIR / "stress_matrix_demand.png"
    dst = SI_FIG_DIR / "si_fig_s10_stress_matrix.png"
    if src.exists():
        dst.unlink(missing_ok=True)
        src.rename(dst)


def gen_si_s11(data):
    """SI Fig S11: Silhouette analysis (materials)."""
    _require(data, "spca_materials", "material_clusters")
    import visualization as cluster_viz

    X_mat = data["spca_materials"]
    labels = data["material_clusters"]["cluster"].values

    cluster_viz.FIGURES_KMEANS_DIR = SI_FIG_DIR
    cluster_viz.plot_silhouette(X_mat.values, labels, "materials")
    src = SI_FIG_DIR / "silhouette_materials.png"
    dst = SI_FIG_DIR / "si_fig_s11_silhouette.png"
    if src.exists():
        dst.unlink(missing_ok=True)
        src.rename(dst)


def gen_si_s12(data):
    """SI Fig S12: Production shares by CRC category."""
    _require(data, "sc_risk")
    import supply_chain_analysis as sca

    sca.FIGURES_MANUSCRIPT_DIR = SI_FIG_DIR
    sca.figSI_production_shares_crc(data["sc_risk"])
    src = SI_FIG_DIR / "figSI_production_shares_crc.png"
    dst = SI_FIG_DIR / "si_fig_s12_crc_production.png"
    if src.exists():
        dst.unlink(missing_ok=True)
        src.rename(dst)


def gen_si_s13(data):
    """SI Fig S13: Cumulative demand dot-with-error-bar."""
    _require(data, "demand")
    from visualizations.manuscript_fig1 import create_fig1_cumulative
    fig = create_fig1_cumulative(
        data["demand"],
        output_path=SI_FIG_DIR / "si_fig_s13_cumulative_demand.png",
    )
    plt.close(fig)


def gen_si_s14(data):
    """SI Fig S14: Demand curves (log scale, single panel)."""
    _require(data, "demand")
    from visualizations.manuscript_fig1 import create_fig1_single_panel
    fig = create_fig1_single_panel(
        data["demand"],
        materials=["Copper", "Steel", "Aluminum", "Cement", "Silicon",
                    "Neodymium", "Nickel", "Glass"],
        output_path=SI_FIG_DIR / "si_fig_s14_demand_log.png",
    )
    plt.close(fig)


def _gen_intensity_by_tech(data, material, filename):
    """Helper for SI Figs S15-S20: intensity by technology for one material."""
    _require(data, "intensity")
    from visualizations.manuscript_figures import create_intensity_by_technology_chart
    result = create_intensity_by_technology_chart(
        data["intensity"],
        material=material,
        output_path=SI_FIG_DIR / filename,
    )
    if result is not None:
        fig, _ = result
        plt.close(fig)


def gen_si_s15(data):
    """SI Fig S15: Steel intensity by technology."""
    _gen_intensity_by_tech(data, "Steel", "si_fig_s15_intensity_steel.png")


def gen_si_s16(data):
    """SI Fig S16: Copper intensity by technology."""
    _gen_intensity_by_tech(data, "Copper", "si_fig_s16_intensity_copper.png")


def gen_si_s17(data):
    """SI Fig S17: Silicon intensity by technology."""
    _gen_intensity_by_tech(data, "Silicon", "si_fig_s17_intensity_silicon.png")


def gen_si_s18(data):
    """SI Fig S18: Neodymium intensity by technology."""
    _gen_intensity_by_tech(data, "Neodymium", "si_fig_s18_intensity_neodymium.png")


def gen_si_s19(data):
    """SI Fig S19: Aluminum intensity by technology."""
    _gen_intensity_by_tech(data, "Aluminum", "si_fig_s19_intensity_aluminum.png")


def gen_si_s20(data):
    """SI Fig S20: Dysprosium intensity by technology."""
    _gen_intensity_by_tech(data, "Dysprosium", "si_fig_s20_intensity_dysprosium.png")


def gen_si_s21(data):
    """SI Fig S21: All materials demand grid."""
    _require(data, "demand")
    from visualizations.manuscript_fig1 import create_all_materials_demand
    fig = create_all_materials_demand(
        data["demand"],
        output_path=SI_FIG_DIR / "si_fig_s21_all_materials_demand.png",
    )
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE REGISTRY
# ═════════════════════════════════════════════════════════════════════════════

FIGURES = {
    # ── Main text ──
    "fig1":          ("Fig 1:  Demand projections (12 panels)",       gen_fig1),
    "fig2":          ("Fig 2:  Technology decomposition",              gen_fig2),
    "fig3":          ("Fig 3:  Variance decomposition",               gen_fig3),
    "fig4":          ("Fig 4:  Material clusters biplot",             gen_fig4),
    "fig5a_risk":    ("Fig 5a: Supply risk matrix (global)",          gen_fig5a_risk),
    "fig5b_risk":    ("Fig 5b: Supply risk matrix (US)",              gen_fig5b_risk),
    "fig5a_sourcing":("Fig 5a: CRC sourcing breakdown",              gen_fig5a_sourcing),
    "fig5b_reserves":("Fig 5b: Reserve adequacy (global)",           gen_fig5b_reserves),
    "fig5c_reserves":("Fig 5c: Reserve adequacy (US)",               gen_fig5c_reserves_us),
    # ── Supplementary ──
    "si_s1":  ("SI S1:  Capacity projections",             gen_si_s1),
    "si_s2":  ("SI S2:  Capacity additions",               gen_si_s2),
    "si_s3":  ("SI S3:  Intensity distributions",          gen_si_s3),
    "si_s4":  ("SI S4:  Variance by year",                 gen_si_s4),
    "si_s5":  ("SI S5:  Spearman heatmap",                 gen_si_s5),
    "si_s6":  ("SI S6:  Spearman tornado",                 gen_si_s6),
    "si_s7":  ("SI S7:  Elasticity tornado",               gen_si_s7),
    "si_s8":  ("SI S8:  Scenario clusters biplot",         gen_si_s8),
    "si_s9":  ("SI S9:  Cluster profiles heatmap",         gen_si_s9),
    "si_s10": ("SI S10: Stress matrix",                    gen_si_s10),
    "si_s11": ("SI S11: Silhouette analysis",              gen_si_s11),
    "si_s12": ("SI S12: CRC production shares",            gen_si_s12),
    "si_s13": ("SI S13: Cumulative demand",                gen_si_s13),
    "si_s14": ("SI S14: Demand (log scale)",               gen_si_s14),
    "si_s15": ("SI S15: Steel intensity by technology",    gen_si_s15),
    "si_s16": ("SI S16: Copper intensity by technology",   gen_si_s16),
    "si_s17": ("SI S17: Silicon intensity by technology",  gen_si_s17),
    "si_s18": ("SI S18: Neodymium intensity by technology",gen_si_s18),
    "si_s19": ("SI S19: Aluminum intensity by technology", gen_si_s19),
    "si_s20": ("SI S20: Dysprosium intensity by technology",gen_si_s20),
    "si_s21": ("SI S21: All materials demand grid",        gen_si_s21),
}


# ═════════════════════════════════════════════════════════════════════════════
# CLI & MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global MAIN_FIG_DIR, SI_FIG_DIR

    parser = argparse.ArgumentParser(
        description="Generate all manuscript figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only", nargs="+", metavar="FIG",
        help="Generate only specific figures (e.g. --only fig1 fig3 si_s13)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available figures and exit",
    )
    parser.add_argument(
        "--output-dir", type=Path, metavar="DIR",
        help="Override output directory (default: outputs/figures/manuscript/)",
    )
    args = parser.parse_args()

    if args.output_dir:
        MAIN_FIG_DIR = args.output_dir.resolve()
        SI_FIG_DIR = MAIN_FIG_DIR / "si_figures"

    if args.list:
        print("\nAvailable figures:")
        print("-" * 60)
        for key, (desc, _) in FIGURES.items():
            print(f"  {key:<18s} {desc}")
        print(f"\nTotal: {len(FIGURES)} figures")
        return

    # Determine which figures to generate
    if args.only:
        targets = []
        for t in args.only:
            if t not in FIGURES:
                print(f"Unknown figure: {t}")
                print(f"Run with --list to see available figures.")
                sys.exit(1)
            targets.append(t)
    else:
        targets = list(FIGURES.keys())

    print("=" * 70)
    print("GENERATING MANUSCRIPT FIGURES")
    print("=" * 70)

    # Create output directories
    MAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)
    SI_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print()
    data = load_all_data()
    print()

    # Generate figures
    succeeded = []
    failed = []
    skipped = []

    for key in targets:
        desc, func = FIGURES[key]
        print(f"  [{key}] {desc} ...", end=" ", flush=True)
        try:
            func(data)
            print("OK")
            succeeded.append(key)
        except KeyError as e:
            print(f"SKIP ({e})")
            skipped.append((key, str(e)))
        except Exception as e:
            print(f"FAIL ({e})")
            failed.append((key, str(e)))
        finally:
            plt.close("all")

    # Summary
    print()
    print("=" * 70)
    print(f"COMPLETE: {len(succeeded)} generated, {len(skipped)} skipped, {len(failed)} failed")
    print("=" * 70)

    if skipped:
        print(f"\nSkipped (missing data):")
        for key, reason in skipped:
            print(f"  {key}: {reason}")

    if failed:
        print(f"\nFailed:")
        for key, reason in failed:
            print(f"  {key}: {reason}")

    print(f"\nMain figures: {MAIN_FIG_DIR}")
    print(f"SI figures:   {SI_FIG_DIR}")


if __name__ == "__main__":
    main()
