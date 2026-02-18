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
USGS_SUPPLY_CHAIN_FILE = DATA_DIR / "input_usgs.csv"

# Supply-chain / risk data (local copies in data/supply_chain/)
SUPPLY_CHAIN_DIR = DATA_DIR / "supply_chain"
RISK_INPUTS_FILE = SUPPLY_CHAIN_DIR / "risk_charts_inputs.xlsx"
USGS_2023_DIR = SUPPLY_CHAIN_DIR  # thin-film CSVs live here too

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

# ── USGS commodity name mapping (USGS name → demand data name) ────────────────
USGS_TO_DEMAND = {
    "Aluminum": "Aluminum",
    "Steel": "Steel",
    "Cement": "Cement",
    "Cu": "Copper",
    "Mn": "Manganese",
    "Ni": "Nickel",
    "Ag": "Silver",
}

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
    "Dysprosium": "Rare Earths", "Neodymium": "Rare Earths",
    "Praseodymium": "Rare Earths", "Terbium": "Rare Earths",
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

SCENARIO_LOG_FEATURES = [
    "total_cumulative_demand", "peak_demand", "mean_demand_early",
    "total_import_exposed_demand",
]
MATERIAL_LOG_FEATURES = [
    "mean_demand", "peak_demand", "demand_volatility",
    "domestic_production", "cumulative_demand",
    "mean_capacity_ratio", "max_capacity_ratio",
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
