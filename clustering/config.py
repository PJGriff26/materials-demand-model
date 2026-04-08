# config.py - Configuration for clustering analysis
# Edit paths and parameters as needed

from pathlib import Path
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Input data
DEMAND_FILE = OUTPUTS_DIR / "data" / "material_demand_by_scenario.csv"
NREL_SCENARIOS_FILE = DATA_DIR / "StdScen24_annual_national.csv"
MATERIAL_INTENSITY_FILE = DATA_DIR / "intensity_data.csv"
# USGS_SUPPLY_CHAIN_FILE removed — superseded by MCS 2025 salient CSVs

# Supply-chain / risk data
# Primary: USGS MCS 2025 raw CSVs (DOI: 10.5066/P13XCP3R) + OECD CRC 2026
MCS2025_DIR = DATA_DIR / "usgs_mcs_2025"
OECD_CRC_DIR = DATA_DIR / "oecd_crc"
# Legacy fallback (for import shares only — percentages not in MCS CSVs)
SUPPLY_CHAIN_DIR = DATA_DIR / "supply_chain"
RISK_INPUTS_FILE = SUPPLY_CHAIN_DIR / "risk_charts_inputs.xlsx"
USGS_2023_DIR = SUPPLY_CHAIN_DIR  # old thin-film CSVs (superseded by MCS 2025)

# Output directories
FIGURES_DIR = OUTPUTS_DIR / "figures" / "clustering"
RESULTS_DIR = OUTPUTS_DIR / "data" / "clustering"

# ── Clustering parameters ──────────────────────────────────────────────────────
RANDOM_SEED = 42
K_RANGE = range(2, 11)          # Test k from 2 to 10
N_INIT = 20                     # Number of k-means random initializations
MAX_ITER = 300
TOLERANCE = 1e-4

EXPECTED_K_SCENARIOS = range(3, 6)   # Narrower search for scenarios
EXPECTED_K_MATERIALS = range(4, 7)   # Narrower search for materials

# ── Validation parameters ──────────────────────────────────────────────────────
N_STABILITY_RUNS = 20
SILHOUETTE_THRESHOLD = 0.5
ARI_THRESHOLD = 0.8

# ── Visualization parameters ──────────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_FORMAT = ["png"]
FIGSIZE_STANDARD = (10, 8)
FIGSIZE_WIDE = (14, 8)

# USGS_TO_DEMAND mapping removed — superseded by MCS 2025 loader

# ── Demand material → risk_charts_inputs.xlsx material mapping ────────────────
# 18 direct matches + rare earths mapped to aggregate "Rare Earths" entry
DEMAND_TO_RISK = {
    "Aluminum": "Aluminum", "Boron": "Boron", "Cement": "Cement",
    "Chromium": "Chromium", "Copper": "Copper", "Lead": "Lead",
    "Magnesium": "Magnesium", "Manganese": "Manganese",
    "Molybdenum": "Molybdenum", "Nickel": "Nickel", "Niobium": "Niobium",
    "Silicon": "Silicon", "Silver": "Silver", "Steel": "Steel",
    "Tin": "Tin", "Vanadium": "Vanadium", "Yttrium": "Yttrium",
    "Zinc": "Zinc",
    # Rare earth elements → aggregate
    # ("Gadium" is a typo in the source intensity_data.csv for Gadolinium;
    #  preserved here so the existing demand series matches.)
    "Dysprosium": "Rare Earths", "Neodymium": "Rare Earths",
    "Praseodymium": "Rare Earths", "Terbium": "Rare Earths",
    "Gadium": "Rare Earths",
}

# ── USGS 2023 individual CSV file mapping for thin-film materials ─────────────
# These materials are NOT in risk_charts_inputs.xlsx but have USGS 2023 CSVs
USGS_2023_FILES = {
    "Cadmium": "mcs2023-cadmi_salient.csv",
    "Gallium": "mcs2023-galli_salient.csv",
    "Germanium": "mcs2023-germa_salient.csv",
    "Indium": "mcs2023-indiu_salient.csv",
    "Selenium": "mcs2023-selen_salient.csv",
    "Tellurium": "mcs2023tellu_salient.csv",
}

# ── Figure subdirectories ────────────────────────────────────────────────────
FIGURES_KMEANS_DIR = FIGURES_DIR / "kmeans"
FIGURES_PCA_DIR = FIGURES_DIR / "pca_analysis"
FIGURES_DIMRED_DIR = FIGURES_DIR / "dimensionality_reduction"
FIGURES_SPCA_STORY_DIR = FIGURES_DIR / "spca_story"
FIGURES_FA_DIR = FIGURES_DIR / "factor_analysis"

# Manuscript figures (supply chain figs 3, 4, SI live here alongside figs 1, 2, S1, S3)
FIGURES_MANUSCRIPT_DIR = OUTPUTS_DIR / "figures" / "manuscript"

# ── Clustering comparison parameters ─────────────────────────────────────────
COMPARISON_FIGURES_DIR = FIGURES_DIR / "comparison"
COMPARISON_DATA_DIR = RESULTS_DIR / "comparison"

METHOD_LABELS = ["VIF-Pruned", "PCA", "Sparse PCA", "Factor Analysis"]
METHOD_KEYS = ["vif", "pca", "spca", "fa"]

SPCA_N_COMPONENTS = {"scenarios": 4, "materials": 5}
SPCA_ALPHA = {"scenarios": 1.0, "materials": 2.0}
FA_N_COMPONENTS = {"scenarios": 4, "materials": 5}

# Log-transform candidates for the reduced feature set (2026-04-08).
# Selected by abs(skewness) > 1.5 on the current 8 scenario / 13 material features.
# Stale entries from the pre-reduction era have been removed.
SCENARIO_LOG_FEATURES = [
    "mean_cv",                       # skew 2.04
]
MATERIAL_LOG_FEATURES = [
    "global_capacity_ratio",         # skew 5.57 — heavy-tailed ratio
    "global_reserve_coverage",       # skew 5.56 — heavy-tailed ratio
    "domestic_reserve_share",        # skew 4.37 — heavy-tailed share
    "us_capacity_ratio",             # skew 3.84 — heavy-tailed ratio
    # growth_rate_long_pct EXCLUDED from log-transform 2026-04-08:
    # All 31 materials have negative CAGR (NREL scenarios frontload buildout).
    # log_transform_features clips to ≥0 first, which would zero-out the entire
    # column and trip the zero-variance filter. The negative-CAGR pattern is
    # itself a finding worth keeping on the raw scale.
    "import_china_frac",             # skew 2.86 — many-zeros distribution
    # scenario_cv excluded from log-transform (2026-04-08): log-transforming
    # it pushed it into VIF >10 collinearity with growth_rate_long_pct in
    # the iterative VIF drop, which would have removed scenario_cv from the
    # final feature set — but it carries the only cross-scenario uncertainty
    # signal we have at the material level. Raw form (skew 2.28) is acceptable.
    # production_hhi excluded for the same reason: log-transform pushed it
    # into VIF collinearity with hhi_wgi/reserves features and it would have
    # been auto-dropped. production_hhi is the y-axis of the locked material
    # clustering pre-registration scatter, so it must survive preprocessing.
]

# ── Create output directories ─────────────────────────────────────────────────
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_KMEANS_DIR, exist_ok=True)
os.makedirs(FIGURES_PCA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIMRED_DIR, exist_ok=True)
os.makedirs(FIGURES_SPCA_STORY_DIR, exist_ok=True)
os.makedirs(FIGURES_FA_DIR, exist_ok=True)
os.makedirs(FIGURES_MANUSCRIPT_DIR, exist_ok=True)
os.makedirs(COMPARISON_FIGURES_DIR, exist_ok=True)
os.makedirs(COMPARISON_DATA_DIR, exist_ok=True)
