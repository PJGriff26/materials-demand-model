# Pipeline Documentation: Materials Demand Model

> **Purpose:** Complete technical reference for all data processing and modeling steps.
> Intended for writing the Data Sources and Methods sections of the thesis proposal.

### Companion Documentation

| File | Purpose |
|------|---------|
| `docs/variable_reference.csv` | Master reference for all variables, parameters, and features (169 entries) |
| `docs/visualization_inventory.csv` | Catalog of all visualizations with axes, data sources, and scripts (76 entries) |
| `docs/data_sources.csv` | Input data file inventory with formats, descriptions, and locations |
| `docs/manuscript_figure_map.csv` | Maps manuscript figures to output files and generation scripts |

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Data Sources and Loading](#2-data-sources-and-loading)
3. [Material Intensity Processing](#3-material-intensity-processing)
4. [Capacity Projection Processing](#4-capacity-projection-processing)
5. [Technology Mapping](#5-technology-mapping)
6. [Distribution Fitting](#6-distribution-fitting)
7. [Stock-Flow Simulation](#7-stock-flow-simulation)
8. [Monte Carlo Simulation](#8-monte-carlo-simulation)
9. [Output: Demand Data](#9-output-demand-data)
10. [Sensitivity Analysis](#10-sensitivity-analysis)
11. [Clustering: Feature Engineering](#11-clustering-feature-engineering)
12. [Clustering: Preprocessing](#12-clustering-preprocessing)
13. [Clustering: K-Means Analysis](#13-clustering-k-means-analysis)
14. [Dimensionality Reduction](#14-dimensionality-reduction)
15. [Supply Chain Risk Analysis](#15-supply-chain-risk-analysis)
16. [Validation and Diagnostics](#16-validation-and-diagnostics)
17. [Known Limitations](#17-known-limitations)

---

## 1. Pipeline Overview

**Master orchestration:** `run_pipeline.py` runs all 11 steps end-to-end (~15–25 min). Supports `--step N`, `--from N`, and `--skip-simulation` flags.

The pipeline has two major stages:

**Stage A — Monte Carlo Demand Simulation** (`src/` + `examples/run_simulation.py`):
Load material intensity data → apply data quality corrections → consolidate technologies → fit parametric distributions to intensity values → build a stock-flow model of capacity additions → run up to 10,000 Monte Carlo iterations (with convergence-based early stopping) → output demand statistics (mean, std, percentiles) per scenario × year × material.

**Stage B — Analysis and Clustering** (`analysis/`, `clustering/`, `visualizations/`):
Sensitivity analysis (variance decomposition, Spearman correlations, Sobol indices) → dimensionality reduction (PCA, Sparse PCA, NMF) → K-means clustering with silhouette-based k selection → supply chain risk analysis (CRC sourcing, reserve adequacy) → manuscript figure generation.

### Pipeline Steps (run_pipeline.py)

| Step | Script | Description |
|------|--------|-------------|
| 1 | `examples/run_simulation.py` | Monte Carlo simulation (10,000 iterations) |
| 2 | `analysis/sensitivity_analysis.py` | Variance decomposition, Spearman correlations |
| 3 | `clustering/sparse_nmf_analysis.py` | Dimensionality reduction (PCA, Sparse PCA, NMF) |
| 4 | `clustering/sparse_pca_story.py` | Sparse PCA interpretation (named components) |
| 5 | `clustering/main_analysis.py` | Production clustering (SPCA → K-means) |
| 6 | `clustering/clustering_comparison.py` | 4-method comparison (VIF / PCA / SPCA / FA) |
| 7 | `clustering/supply_chain_analysis.py` | Supply chain risk (CRC sourcing, reserves) |
| 8 | `visualizations/manuscript_figures.py` | Manuscript figures (Fig. 2, SI figures) |
| 9 | `visualizations/manuscript_fig1.py` | Figure 1 (demand curves + cumulative) |
| 10 | `visualizations/feature_scatterplots.py` | Exploratory figures (scatterplots, heatmaps) |
| 11 | `analysis/sobol_analysis.py` | Sobol sensitivity analysis (per-material + grouped + global) |

### Key Dimensions

| Dimension | Count | Source |
|-----------|-------|--------|
| Scenarios | 61 | NREL Standard Scenarios 2024 |
| Years | 9 | 2026, 2029, 2032, 2035, 2038, 2041, 2044, 2047, 2050 |
| Materials | 31 | Derived from intensity data tech-material pairs |
| Capacity technologies | 23 | NREL StdScen24 columns ending `_MW` |
| Intensity technologies | 19 (after consolidation from 21 raw); 18 capacity techs mapped, 5 unmapped | `technology_mapping.py` |
| MC iterations | up to 10,000 (convergence-based early stopping) | `src/stock_flow_simulation.py` |

---

## 2. Data Sources and Loading

### 2.1 Material Intensity Data

- **File:** `data/intensity_data.csv`
- **Loader:** `src/data_ingestion.py` → `MaterialIntensityLoader`
- **Required columns:** `technology`, `Material`, `value`
- **Raw units:** tonnes per gigawatt (t/GW)
- **Conversion applied:** `intensity_t_per_mw = value / 1000.0` (t/GW → t/MW)
- **Column renaming:** `Material` → `material`, `value` → `intensity_raw` → `intensity_t_per_mw`
- **Validation:** checks for missing columns, null values, negative values, zero values (warning only), duplicate (technology, material) combinations (expected for uncertainty representation)
- **Location:** `src/data_ingestion.py:77–243`

### 2.2 NREL Standard Scenarios 2024 (Capacity Projections)

- **File:** `data/StdScen24_annual_national.csv`
- **Loader:** `src/data_ingestion.py` → `CapacityProjectionLoader(level='national')`
- **Header handling:** `skiprows=3` (3 header rows: description, note, column names)
- **Key columns:** `scenario`, `t` (renamed to `year`), plus 23 capacity columns (`battery_4_MW`, `battery_8_MW`, `bio_MW`, `bio-ccs_MW`, `coal_ccs_MW`, `coal_MW`, `csp_MW`, `distpv_MW`, `gas_cc_ccs_MW`, `gas_cc_MW`, `gas_ct_MW`, `geo_MW`, `hydro_MW`, `nuclear_MW`, `nuclear_smr_MW`, `o-g-s_MW`, `pumped-hydro_MW`, `h2-ct_MW`, `upv_MW`, `wind_offshore_MW`, `wind_onshore_MW`, `dac_MW`, `electrolyzer_MW`)
- **Units:** megawatts (MW) — total installed capacity per technology per year
- **Years:** 3-year intervals from 2026 to 2050 (9 time steps)
- **Scenarios:** 61 NREL Standard Scenarios
- **Validation:** checks for scenario/year columns, capacity columns (missing ones produce warnings), year range sanity (2020–2060)
- **Location:** `src/data_ingestion.py:272–490`

### 2.3 Supply Chain Risk Data

- **File:** `data/supply_chain/risk_charts_inputs.xlsx`
- **Loader:** `clustering/feature_engineering.py` → `load_risk_data()`
- **Sheets loaded:**
  - `aggregate` — annual US production, import, export, consumption, net_import by material (19 materials). Non-numeric entries (`'-----'`) coerced to NaN via `pd.to_numeric(errors="coerce")`.
  - `import_dependency` — annual net import reliance (%) by material. Values like `'E'` (withheld) skipped.
  - `production` — production data by country and material
  - `reserves` — reserves (kt) by country and material. Special rows: `Global`, `United States`, `Other`.
  - `import_shares` — columns: `material`, `country`, `share` (% of US imports from each country)
  - `crc` — Country Risk Classification. First 2 columns: `country`, `crc` (integer 1–7, or special categories)
- **Location:** `clustering/feature_engineering.py:38–52`

### 2.4 USGS 2023 Thin-Film Material CSVs

- **Directory:** `data/supply_chain/`
- **Files:** 6 individual CSVs for materials not in `risk_charts_inputs.xlsx`:

| Material | Filename |
|----------|----------|
| Cadmium | `mcs2023-cadmi_salient.csv` |
| Gallium | `mcs2023-galli_salient.csv` |
| Germanium | `mcs2023-germa_salient.csv` |
| Indium | `mcs2023-indiu_salient.csv` |
| Selenium | `mcs2023-selen_salient.csv` |
| Tellurium | `mcs2023tellu_salient.csv` |

- **Extracted fields:**
  - `production_t` — average US production (tonnes). Columns starting with `USprod`. If column name contains `_kg`, divides by 1000.
  - `nir_pct` — Net Import Reliance (%). Column containing `NIR` or `nir`. `<` and `>` stripped. Divided by 100 to get 0–1 fraction.
- **Loader:** `clustering/feature_engineering.py:55–101`

### 2.5 USGS Supply Chain Summary (Legacy)

- **File:** `data/input_usgs.csv`
- **Loader:** `clustering/feature_engineering.py` → `load_usgs_data()`
- **Usage:** Limited — only 7 materials mapped via `USGS_TO_DEMAND`. Superseded by `risk_charts_inputs.xlsx` for most features.
- **Mapping:** `clustering/config.py:50–58`

---

## 3. Material Intensity Processing

**Code:** `src/data_ingestion.py` → `MaterialIntensityLoader._standardize()`

### 3a. Data Quality: Corrections and Outlier Removal

**Code:** `src/data_quality.py`

Before standardization, the raw intensity data undergoes quality control:

1. **Known corrections** (3 documented data entry errors):
   - CIGS / Indium: 44155 → 44.155 (decimal placement error, 1000× too high)
   - utility-scale solar pv / Silicon: 4 → 4000 (decimal placement error, 1000× too low)
   - Solar Distributed / Silicon: 4 → 4000 (same error)

2. **Known single-point outlier removals** (6 isolated extreme values):
   - CIGS / Cadmium 265.0 (204× median; other values are 1.3)
   - CdTe / Tellurium 500.0 (z=5.22; 1.92× gap above next-highest 260)
   - onshore wind / Aluminum 13200.0 (z=4.93; 2.49× gap above next-highest 5300)
   - Solar Distributed / Lead 336.0 (14.8× median)
   - utility-scale solar pv / Lead 336.0 (14.8× median)
   - offshore wind / Nickel 376.57 (3.4× median; other 3 values are all 111)

3. **Statistical outlier detection** (for flagging, not automatic removal):
   - IQR method: flag values outside Q1 − 3×IQR to Q3 + 3×IQR (`IQR_MULTIPLIER = 3.0`)
   - Z-score method: flag values with |z| > 4.0 (`Z_SCORE_THRESHOLD = 4.0`)
   - Ratio method: flag values > 100× group median (`RATIO_THRESHOLD = 100`)

### 3b. Technology Consolidation

**Code:** `src/data_ingestion.py` (Step 1c), configured by `src/technology_mapping.py` → `TECHNOLOGY_CONSOLIDATION`

The raw `intensity_data.csv` contains 21 technologies. Two are alternate-casing BOS (balance-of-system) entries that are merged into their parent technologies during preprocessing:

| Raw Technology | Merged Into | Rows | Content |
|---------------|-------------|------|---------|
| `CDTE` | `CdTe` | 15 | BOS materials (Aluminum, Cement, Copper, Glass, Steel) |
| `ASIGE` | `a-Si` | 15 | BOS materials (identical values to CDTE) |

After consolidation: 21 → **19 intensity technologies**. Without this step, the CdTe fraction of UPV would be missing BOS material demand (~15% underestimate of structural materials from solar PV).

### 3c. Standardization

1. **Rename columns:** `Material` → `material`, `value` → `intensity_raw`
2. **Strip whitespace** from `technology` and `material` string columns
3. **Unit conversion:** `intensity_t_per_mw = intensity_raw / 1000.0`
   - Source data is in t/GW; capacity data is in MW
   - 1 GW = 1000 MW, so divide by 1000
4. **Sort** by (technology, material, intensity_t_per_mw) for reproducibility
5. **Drop** temporary `intensity_raw` column

**Output columns:** `technology`, `material`, `intensity_t_per_mw`

Multiple rows per (technology, material) pair represent empirical uncertainty in intensity values. These are used to fit parametric distributions (Section 6).

---

## 4. Capacity Projection Processing

**Code:** `src/data_ingestion.py` → `CapacityProjectionLoader._standardize()`

1. **Rename:** `t` → `year`
2. **Select columns:** `scenario`, `year`, + all `*_MW` columns present
3. **Type coercion:** `year` → int, all `*_MW` columns → numeric (coerce errors to NaN)
4. **Sort** by (scenario, year)

The capacity data represents **total installed capacity** (stock) at each time step — not additions. The stock-flow model (Section 7) derives additions from consecutive capacity differences.

---

## 5. Technology Mapping

**Code:** `src/technology_mapping.py`

Maps each NREL capacity technology (23 total) to one or more material intensity technologies with weights summing to 1.0.

### Complete Mapping Table

| Capacity Tech | Intensity Tech(s) | Weight(s) | Lifetime (yr) |
|--------------|-------------------|-----------|---------------|
| `upv` | utility-scale solar pv / CdTe / CIGS | 0.90 / 0.07 / 0.03 | 30 |
| `distpv` | Solar Distributed | 1.0 | 30 |
| `csp` | CSP | 1.0 | 30 |
| `wind_onshore` | onshore wind | 1.0 | 25 |
| `wind_offshore` | offshore wind | 1.0 | 25 |
| `coal` | Coal | 1.0 | 40 |
| `coal_ccs` | Coal CCS | 1.0 | 40 |
| `gas_cc` | NGCC | 1.0 | 30 |
| `gas_ct` | NGGT | 1.0 | 30 |
| `gas_cc_ccs` | Gas CCS | 1.0 | 30 |
| `h2-ct` | NGGT (proxy) | 1.0 | 30 |
| `bio` | Biomass | 1.0 | 30 |
| `bio-ccs` | Bio CCS | 1.0 | 30 |
| `nuclear` | Nuclear New | 1.0 | 60 |
| `nuclear_smr` | Nuclear New (proxy) | 1.0 | 60 |
| `hydro` | Hydro | 1.0 | 80 |
| `pumped-hydro` | Hydro (proxy) | 1.0 | 80 |
| `geo` | Geothermal | 1.0 | 30 |
| `battery_4` | *(unmapped)* | — | 15 |
| `battery_8` | *(unmapped)* | — | 15 |
| `o-g-s` | *(unmapped)* | — | 30 |
| `dac` | *(unmapped)* | — | 30 |
| `electrolyzer` | *(unmapped)* | — | 20 |

**Key design choice:** `upv` (utility-scale PV) uses a **90/7/3 split** across crystalline silicon (c-Si), CdTe, and CIGS thin-film technologies, based on market share data from Fraunhofer ISE, IEA-PVPS, and EIA. This is the only capacity technology with a multi-technology mapping.

**Unmapped technologies:** `battery_4`, `battery_8`, `o-g-s`, `dac`, `electrolyzer` have empty mappings `{}` and are skipped during simulation. Their material demands are **not included** in the output.

---

## 6. Distribution Fitting

**Code:** `src/distribution_fitting.py` → `DistributionFitter`

For each (intensity_technology, material) pair, fits parametric distributions to the raw intensity values.

### Process

1. **Filter** raw data for each (technology, material) group
2. **Remove** NaN and infinite values
3. **Special case n=1:** Create narrow uniform distribution ±10% around the single value
4. **Special case zero variance:** Only uniform distribution is attempted; all others get dummy AIC=999999
5. **Fit candidate distributions** (MLE):
   - **Truncated normal** (truncated at 0): Fitted via moment matching using Nelder-Mead optimization to match data mean and std
   - **Lognormal** (`scipy.stats.lognorm.fit`)
   - **Gamma** (`scipy.stats.gamma.fit`)
   - **Uniform** (min to max of data)
6. **Rank** by AIC (Akaike Information Criterion, lower = better)
7. **Validate tail behavior** for each candidate in AIC order:
   - Lognormal shape parameter `s` must be < 3.0
   - Generate 10,000 test samples; max/median ratio must be < 100×
   - First candidate passing both checks is selected
8. **Fallback:** If all candidates fail validation, use uniform distribution with 10% padding

### Goodness-of-Fit Tests

- **Kolmogorov-Smirnov test:** `scipy.stats.kstest` (α = 0.05)
- **Anderson-Darling test:** General implementation using CDF values (critical value ≈ 2.492 at α = 0.05)
- **Information criteria:** AIC = 2k − 2ln(L), BIC = k·ln(n) − 2ln(L)

### Sampling

`MaterialIntensityDistribution.sample(n)` draws `n` random values from the best-fit parametric distribution. Always parametric — no empirical fallback.

---

## 7. Stock-Flow Simulation

**Code:** `src/stock_flow_simulation.py` → `StockFlowState`, `MaterialsStockFlowSimulation`

### Stock-Flow Model

For each (scenario, capacity_technology) pair:

```
Stock(t) = total installed capacity at year t (from NREL data)
Additions(t) = Stock(t) − Stock(t−1) + Retirements(t)
Retirements(t) = Additions(t − lifetime)   [if that year exists, else 0]
```

**First year (2026):** Treated as baseline. `Stock(2026) = capacity value`, `Additions(2026) = 0`, `Retirements(2026) = 0`. No material demand is counted for the baseline year.

**Negative additions handling:** If `Stock(t) < Stock(t−1) − Retirements(t)` (i.e., capacity decreases faster than retirements), additions are set to 0 and retirements are adjusted to `Stock(t−1) − Stock(t)`.

**Important note on retirements within the modeling window:** Since the simulation starts in 2026 and the shortest technology lifetime is 15 years (batteries), the earliest retirement would occur at 2026 + 15 = 2041 — but only for additions made in 2026, which are set to 0 (baseline). The first non-zero additions occur in 2029, so the earliest possible retirement is 2029 + 15 = 2044. For most technologies (25–80 year lifetimes), no retirements fire within the 2026–2050 window. This means the model primarily captures **new build** material demand.

### Material Demand Calculation

For a single Monte Carlo iteration:

```
demand(scenario, year, material) = Σ over capacity_technologies [
    Σ over intensity_technologies [
        Additions(scenario, cap_tech, year) × weight(cap_tech → int_tech)
        × sampled_intensity(int_tech, material)
    ]
]
```

**Units:** MW × (t/MW) × dimensionless_weight = tonnes

Only positive additions contribute. Zero or negative additions are skipped.

---

## 8. Monte Carlo Simulation

**Code:** `src/stock_flow_simulation.py` → `MaterialsStockFlowSimulation.run_monte_carlo()`

**Orchestration:** `run_pipeline.py` (Step 1) or `examples/run_simulation.py`

### Process

1. **Build stock-flow model** once (deterministic; depends only on capacity projections and lifetimes)
2. **For each of up to 10,000 iterations:**
   a. Sample one intensity value per (intensity_technology, material) pair from its fitted distribution
   b. Calculate material demand for all (scenario, year, material) combinations using the sampled intensities
   c. Store in 4D array: `results[iteration, scenario, year, material]`
3. **Convergence check** (optional early stopping):
   - After a minimum of 1,000 iterations, check every 500 iterations
   - Compute running means across all (scenario, year, material) cells
   - If the maximum relative change from previous check falls below `rtol = 0.01` (1%), stop early
   - Set `convergence_rtol = 0` to disable and always run full 10,000 iterations
4. **Compute statistics** across the iteration axis:
   - Mean, standard deviation
   - Percentiles: 2.5, 5, 25, 50, 75, 95, 97.5

### Parameters

- `n_iterations = 10,000` (hard maximum)
- `random_seed = 42`
- `convergence_rtol = 0.01` (1% relative tolerance for early stopping)
- `convergence_check_every = 500` (check interval)
- `convergence_min_iterations = 1,000` (minimum before checking)
- All 61 scenarios run by default (configurable via `FOCUS_SCENARIOS`)

### Result Shape

`iterations_data` array shape: `(N, 61, 9, 31)` = iterations × scenarios × years × materials, where N ≤ 10,000 (may be fewer if convergence triggers early stopping)

---

## 9. Output: Demand Data

### Primary Output

- **File:** `outputs/material_demand_by_scenario.csv` (written by `examples/run_simulation.py`)
- **Note:** Clustering scripts read from `outputs/data/material_demand_by_scenario.csv` (configured in `clustering/config.py`). Ensure this file is kept in sync or symlinked.
- **Columns:** `scenario`, `year`, `material`, `mean`, `std`, `p2` (p2.5), `p5`, `p25`, `p50`, `p75`, `p95`, `p97` (p97.5)
- **Rows:** 61 scenarios × 9 years × 31 materials = 17,019 rows
- **Units:** tonnes (metric tons)
- **Interpretation:** For each scenario-year-material triple, the statistics summarize the distribution of demand across MC iterations. Uncertainty derives from material intensity distributions — each iteration uses the same capacity projections.

### Secondary Output

- **File:** `outputs/material_demand_summary.csv`
- **Aggregation:** Sums across all scenarios for each (year, material) combination

### Fitted Distribution Data

- **Directory:** `outputs/data/` (written by `run_full_simulation()` via `output_dir` parameter)
- **Files:** `fitted_distributions.csv`, `fit_summary.csv`

---

## 10. Sensitivity Analysis

**Code:** `analysis/sensitivity_analysis.py` (Step 2 of `run_pipeline.py`)

Identifies key factors driving material demand variability. Run after the Monte Carlo simulation completes.

### Analyses Performed

1. **Technology contribution:** For each material, quantifies which capacity technologies drive the largest share of demand (based on additions × intensity × weight).
2. **Variance decomposition:** Decomposes total demand variance into intensity uncertainty (from MC sampling) vs. scenario-driven capacity differences.
3. **Spearman rank correlations:** Computes Spearman ρ between material intensities and total demand across MC iterations, identifying which intensity parameters most influence output.
4. **Intensity elasticity:** Measures percentage change in demand per percentage change in each material's intensity.

### Output Files

- `outputs/data/sensitivity/technology_contributions.csv` — Technology × material contribution matrix
- `outputs/data/sensitivity/variance_decomposition.csv` — Intensity vs. capacity variance shares
- `outputs/data/sensitivity/spearman_correlations.csv` — Intensity-demand rank correlations
- `outputs/data/sensitivity/intensity_elasticity.csv` — Demand elasticity to each intensity

### 10.1 Sobol Sensitivity Analysis

**Code:** `analysis/sobol_analysis.py` (Step 11 of `run_pipeline.py`)

Variance-based global sensitivity analysis using Sobol indices (SALib). Decomposes output variance into contributions from individual intensity parameters, providing a rigorous complement to the Spearman and elasticity methods above.

#### Key Model Property

For a given scenario and year, `Demand(m) = Σ_i [a_i × I_i]` — the model is **linear** in intensity parameters. This means first-order indices (S1) account for all variance (S1 ≈ ST, negligible interactions), which validates the simpler methods used in Step 2.

#### Three Analysis Levels

1. **Per-material individual Sobol:** For each of the 31 output materials, identifies contributing intensity parameters (D = 1–21 per material) and computes S1 and ST for each. Materials with D = 1 (e.g., Yttrium, Selenium) get S1 = 1.0 analytically since Sobol requires D ≥ 2.

2. **Per-material grouped Sobol:** Groups intensity parameters by technology sector (Solar, Wind, Nuclear, Hydro, Fossil, Biomass, Geothermal) using SALib's `groups` key. Quantifies which sectors drive each material's demand uncertainty.

3. **Global grouped Sobol:** Aggregates demand across 17 critical materials and decomposes total variance by technology sector. Provides a single high-level view of which sectors matter most for overall critical material demand.

#### Method

- **Sampling:** Saltelli quasi-random sampling (N = 1024 base samples, `calc_second_order=False`)
- **Evaluation:** Precomputed linear coefficients `a_i = Σ(additions_MW × weight)` enable vectorized evaluation `Y = X @ coefficients` (~microseconds per sample)
- **Analysis:** `SALib.analyze.sobol` for S1, ST, and 95% confidence intervals
- **Total evaluations:** N × (2D + 2) per material; worst case D = 21 → ~45K evaluations, sub-second runtime

#### Output Files

- `outputs/data/sensitivity/sobol_indices.csv` — Per-material individual Sobol indices (S1, S1_conf, ST, ST_conf, coefficient, n_params)
- `outputs/data/sensitivity/sobol_grouped_indices.csv` — Per-material grouped indices by technology sector
- `outputs/data/sensitivity/sobol_global_indices.csv` — Global indices for aggregate critical material demand

#### Visualizations (6 figures)

- `sobol_s1_bar.png` — Multi-panel horizontal bar chart of S1 by technology for top 12 materials
- `sobol_summary_heatmap.png` — Material × technology heatmap of S1 values
- `sobol_s1_vs_st.png` — S1 vs ST scatter validating model linearity (points on y = x line)
- `sobol_vs_spearman.png` — Cross-validation: Sobol S1 vs Spearman ρ²
- `sobol_grouped_bar.png` — Stacked bars showing sector contributions per material
- `sobol_global_bar.png` — Aggregate critical material demand variance by sector

#### Report

- `outputs/reports/sobol_analysis_report.txt` — Text summary with top parameters, linearity validation (R²), cross-method comparison

---

## 11. Clustering: Feature Engineering

**Code:** `clustering/feature_engineering.py`

### 11.1 Scenario Features (16 features, 61 scenarios)

Source: demand output + NREL capacity data at 2035.

| # | Feature | Formula / Description | Code location |
|---|---------|----------------------|---------------|
| 1 | `total_cumulative_demand` | Sum of total demand across all years | line 398 |
| 2 | `peak_demand` | Max total demand across years | line 401 |
| 3 | `mean_demand_early` | Mean total demand for years 2029–2035 | lines 404–405 |
| 4 | `year_of_peak` | Year with highest total demand | line 408 |
| 5 | `demand_slope` | Linear regression slope (total demand vs. year) via `np.polyfit(years, values, 1)[0]` | lines 411–417 |
| 6 | `temporal_concentration` | early_half_sum / (late_half_sum + 1), split at median year | lines 420–425 |
| 7 | `mean_cv` | Mean coefficient of variation (std/mean) across all material-year rows for that scenario | lines 428–430 |
| 8 | `mean_ci_width` | Mean relative CI width: (p97 − p2) / mean across all material-year rows | lines 433–435 |
| 9 | `solar_fraction_2035` | (upv + distpv + csp capacity) / total capacity at 2035 | lines 438–446 |
| 10 | `wind_fraction_2035` | (wind_onshore + wind_offshore capacity) / total capacity at 2035 | line 447 |
| 11 | `storage_fraction_2035` | (battery + pumped-hydro capacity) / total capacity at 2035 | line 448 |
| 12 | `supply_chain_stress` | Demand-weighted mean of (import_dependency × CRC_risk/8) across materials and years | line 577 |
| 13 | `peak_supply_chain_stress` | Maximum annual supply chain stress index | line 579 |
| 14 | `mean_n_exceeding_production` | Mean number of materials exceeding US production per year | line 581 |
| 15 | `peak_n_exceeding_production` | Peak number of materials exceeding US production in any year | line 583 |
| 16 | `total_import_exposed_demand` | Cumulative demand weighted by import dependency | line 585 |

**Note:** "Total demand" for a scenario-year means `sum of mean demand across all 31 materials` for that (scenario, year) pair.

### 11.2 Material Features (23 features, 31 materials)

#### Demand-derived features (6)

| # | Feature | Formula | Code location |
|---|---------|---------|---------------|
| 1 | `mean_demand` | Grand mean of `mean` across all scenarios and years | line 636 |
| 2 | `peak_demand` | Max `mean` across all scenarios and years | line 639 |
| 3 | `scenario_cv` | For each scenario: sum `mean` across years → compute CV across scenarios | line 650 |
| 4 | `mean_ci_width` | Mean of (p97 − p2) / mean across all scenario-year rows | line 655 |
| 5 | `demand_volatility` | Std of `mean` across years (within each scenario), then averaged across scenarios | line 662 |
| 6 | `demand_slope` | Average slope of mean demand vs. year (averaged across scenarios) | line 672 |

#### Supply-chain features (17)

| # | Feature | Source | Formula | Code location |
|---|---------|--------|---------|---------------|
| 7 | `domestic_production` | aggregate sheet + USGS 2023 CSVs | Average annual US production (tonnes). Default 0 if missing | line 677 |
| 8 | `import_dependency` | import_dependency sheet + USGS NIR | 0–1 fraction (1 = fully imported). Default 1.0 if missing | line 680 |
| 9 | `crc_weighted_risk` | import_shares + crc sheets | Σ(import_share × crc_weight) / Σ(import_share). Weights: US=0, OECD=1, CRC1→2, CRC2→3, ..., CRC7→8, China=7, Undefined=5. Scale 0–8. Default 5.0 if missing | line 683 |
| 10 | `mean_capacity_ratio` | demand + production | mean_scenario_total_demand / US_production | line 687 |
| 11 | `max_capacity_ratio` | demand + production | max_scenario_total_demand / US_production | line 692 |
| 12 | `exceedance_frequency` | demand + production | Fraction of scenarios where total demand > US production | line 704 |
| 13 | `cumulative_demand` | demand output | Median cumulative demand across scenarios (sum over years, then median across scenarios) | line 716 |
| 14 | `reserve_consumption_pct` | demand + reserves | (cumulative_demand / global_reserves) × 100. Percentage of global reserves consumed | line 722 |
| 15 | `domestic_reserve_coverage` | reserves + demand | US_reserves / cumulative_demand. Fraction of demand coverable by domestic reserves | line 731 |
| 16 | `global_reserve_coverage` | reserves + demand | global_reserves / cumulative_demand. Values >1 indicate adequate reserves | line 739 |
| 17 | `reserves_high_risk_frac` | reserves + crc | Fraction of global reserves in CRC 5–7 + China | line 747 |
| 18 | `reserves_oecd_frac` | reserves + crc | Fraction in OECD + United States | line 752 |
| 19 | `reserves_china_frac` | reserves + crc | Fraction in China | line 757 |
| 20 | `import_china_frac` | import_shares + crc | Fraction of imports from China | line 764 |
| 21 | `import_high_risk_frac` | import_shares + crc | Fraction from CRC 5–7 + China | line 769 |
| 22 | `import_oecd_frac` | import_shares + crc | Fraction from OECD | line 774 |
| 23 | `import_hhi` | import_shares | Herfindahl-Hirschman Index of import country concentration. HHI = Σ(share²), 0–1 scale (higher = more concentrated) | line 779 |

**Material name mapping:** `DEMAND_TO_RISK` in `clustering/config.py:62–73` maps 22 demand material names to 19 risk material names (4 rare earth elements → aggregate "Rare Earths" entry). Materials not in this mapping get default/zero values for supply-chain features.

**Missing value handling:** All inf/NaN replaced with 0 at the end (`feature_engineering.py:639`).

---

## 12. Clustering: Preprocessing

**Code:** `clustering/preprocessing.py` → `preprocess_pipeline()`

### Pipeline Steps

1. **Log transformation:** `log10(clip(x, 0) + 1)` applied in-place to specified columns.

   Scenario log features: `total_cumulative_demand`, `peak_demand`, `mean_demand_early`, `total_import_exposed_demand`

   Material log features: `mean_demand`, `peak_demand`, `demand_volatility`, `domestic_production`, `cumulative_demand`, `mean_capacity_ratio`, `max_capacity_ratio`

2. **Drop zero-variance columns:** Any column with `std == 0` removed.

3. **VIF check (before pruning):** Compute Variance Inflation Factor for all remaining features. VIF_j = 1 / (1 − R²_j) where R²_j is from regressing feature j on all other features. Uses `statsmodels.stats.outliers_influence.variance_inflation_factor`.

4. **Iterative VIF pruning:** While any feature has VIF > 10: drop the feature with the highest VIF. Repeat. This removes multicollinear features.

5. **Z-score standardization:** `(x − mean) / std` via `sklearn.preprocessing.StandardScaler`. Applied after log transform and VIF pruning.

### Typical Pruning Results

**Scenarios:** Starting from 16 features, VIF pruning typically retains ~2 features (e.g., `demand_slope`, `storage_fraction_2035`). Most demand-magnitude features are highly correlated and dropped.

**Materials:** Starting from 23 features, VIF pruning typically retains ~13 features (e.g., `mean_ci_width`, `demand_volatility`, `demand_slope`, `import_dependency`, `mean_capacity_ratio`, `reserve_consumption_pct`, `domestic_reserve_coverage`, `global_reserve_coverage`, `reserves_oecd_frac`, `reserves_china_frac`, `import_china_frac`, `import_high_risk_frac`, `import_oecd_frac`).

---

## 13. Clustering: K-Means Analysis

**Code:** `clustering/clustering.py` → `ClusterAnalyzer`

### K Selection

- Test k = 2 through 10 (`config.py:30`)
- For each k: run K-means with `n_init=20`, `max_iter=300`, `tol=1e-4`, `random_state=42`, `init="k-means++"`
- Record WCSS (within-cluster sum of squares) and mean silhouette score
- Recommend k with highest silhouette score
- **Manual override for materials:** `best_k_mat = 4` hardcoded in `main_analysis.py:104` (auto-selected k=2 had poor stability)

### Final Model

- K-means with selected k, same parameters as above
- Per-sample silhouette coefficients stored for export
- Per-cluster summary: size and mean silhouette

### Stability Validation

- Run K-means 20 times with `n_init=1` and different random seeds (0 through 19)
- Compute pairwise Adjusted Rand Index (ARI) across all 20 × 19 / 2 = 190 pairs
- Report mean ARI ± std
- Interpretation: >0.9 = very stable, 0.7–0.9 = moderately stable, <0.7 = unstable

### Feature Sensitivity

- Leave-one-feature-out: for each feature, re-cluster without it and compute ARI vs. full model
- Lower ARI = more influential feature (removing it changes the clustering more)

### Cluster Profiles

- For each cluster: compute mean of raw (un-standardized) feature values across cluster members
- Visualized as z-scored heatmap (standardized across clusters for color comparability)

### Stress Matrix

**Code:** `clustering/main_analysis.py:132–153`

Cross-tabulates scenario clusters × material clusters. For each (scen_cluster, mat_cluster) cell: sums the `mean` demand across all constituent scenarios and materials in the demand dataset. Visualized as a heatmap.

---

## 14. Dimensionality Reduction

**Code:** `clustering/sparse_nmf_analysis.py` (Step 3), `clustering/sparse_pca_story.py` (Step 4), `clustering/clustering_comparison.py` (Step 6)

Before K-means clustering, multiple dimensionality reduction methods are applied and compared.

### Methods

1. **PCA (Principal Component Analysis):** Standard PCA on standardized features. Used as baseline for comparison. All components retained for analysis.
2. **Sparse PCA:** Regularized PCA with L1 penalty encouraging zero loadings for interpretability. Components: scenarios = 4, materials = 5. Alpha: scenarios = 1.0, materials = 2.0 (`clustering/config.py:102–103`).
3. **NMF (Non-negative Matrix Factorization):** Applied to non-negative feature matrices. Alternative factorization for comparison.
4. **Factor Analysis:** Maximum likelihood factor analysis with varimax rotation. Components: scenarios = 4, materials = 5 (`clustering/config.py:104`).

### Sparse PCA Interpretation (Step 4)

`clustering/sparse_pca_story.py` assigns interpretable labels to Sparse PCA components based on their loading patterns. These named components are used as input to the production K-means clustering (Step 5, `clustering/main_analysis.py`).

### 4-Method Clustering Comparison (Step 6)

`clustering/clustering_comparison.py` runs K-means on four different feature representations and compares results:

| Method | Input Features |
|--------|---------------|
| VIF-Pruned | Raw features after VIF pruning + z-score standardization |
| PCA | PCA scores (all components) |
| Sparse PCA | Sparse PCA scores |
| Factor Analysis | Factor scores |

Comparison metrics: silhouette scores, ARI agreement between methods, cluster membership overlap.

### Output Files

- `outputs/figures/clustering/dimensionality_reduction/` — PCA/SPCA/NMF comparison plots
- `outputs/figures/clustering/spca_story/` — Named component visualizations
- `outputs/figures/clustering/comparison/` — 4-method comparison figures
- `outputs/data/clustering/comparison/` — Comparison data tables

---

## 15. Supply Chain Risk Analysis

**Code:** `clustering/supply_chain_analysis.py`

Standalone script generating six manuscript-style figures.

### Fig. 3: Demand with CRC Sourcing Breakdown

For each material with risk data:
1. Compute peak demand across all scenarios and years
2. Allocate peak demand by sourcing category:
   - **US domestic** share = `1 − import_dependency`
   - **Import CRC shares:** For each CRC category, share = `(country_share / total_import_share) × import_dependency`
3. Overlay US production and consumption baselines (from aggregate sheet)

CRC categories (in stacking order): United States, OECD, CRC 1–7, China, Undefined.

### Fig. 4: Reserve Adequacy by CRC Category

For each material:
1. Get reserves by country from reserves sheet
2. Map countries to CRC categories
3. Compute reserve coverage ratio: `(reserves_kt × 1000) / cumulative_demand` per CRC category
4. Stack bars by CRC category; vertical line at coverage = 1

### Fig. 5A: Global Supply Risk Matrix (`fig3_supply_risk_matrix`)

Log-log scatter plot of **reserve adequacy** (X) vs. **production stress** (Y):
- **X-axis:** `global_reserves / cumulative_demand_2026_2050` (median scenario)
- **Y-axis:** `peak_annual_demand / US_apparent_consumption`
- **Color:** Net import reliance (0–100%, green→red colormap)
- **Markers:** Circles = reserves reported by USGS; Diamonds = reserves not separately reported (Al, Si, Steel, Y, Cement)
- REEs aggregated as single point (Dy+Nd+Pr+Tb)
- Horizontal dashed line at production stress = 100% (peak demand = current consumption)
- Quadrant labels: Reserve-/production-constrained, Adequate reserves, Low risk

### Fig. 5B: US Domestic Supply Risk Matrix (`fig3b_us_supply_risk_matrix`)

Same axes and methodology as Fig. 5A but using **US economic reserves**:
- Vertical dashed line at reserve adequacy = 1 (US reserves = cumulative demand)
- **Squares** at left margin for materials with zero reported US reserves (e.g., Mn, Tin)
- Pink-shaded region marks the zero-US-reserves zone

### Fig. SI: Production Shares by CRC Category (`figSI_production_shares_crc`)

Stacked horizontal bars showing global production distribution by CRC category.

---

## 16. Validation and Diagnostics

### Monte Carlo Simulation
- Random seed = 42 for reproducibility
- Up to 10,000 iterations with convergence-based early stopping (rtol = 1%, checked every 500 iterations after minimum 1,000)
- 95% CI via p2.5–p97.5 percentiles

### Distribution Fitting
- KS test and AD test for each fit
- AIC/BIC for model comparison
- Tail validation: rejects distributions producing extreme samples

### Clustering
- **Silhouette analysis:** Per-sample and mean silhouette scores. Threshold = 0.5 in config (informational).
- **Stability (ARI):** Threshold = 0.8 in config (informational).
- **Feature sensitivity:** Leave-one-out ARI.
- **VIF pruning:** Threshold = 10 to remove multicollinear features.
- **Elbow plots** and **silhouette vs. k** plots for k selection.
- **PCA biplot:** 2D projection with cluster coloring, loading vectors, and entity labels.

### Output Files

| File | Content |
|------|---------|
| `outputs/data/clustering/scenario_clusters.csv` | scenario, cluster, silhouette |
| `outputs/data/clustering/material_clusters.csv` | material, cluster, silhouette |
| `outputs/data/clustering/scenario_cluster_profiles.csv` | Mean raw features per scenario cluster |
| `outputs/data/clustering/material_cluster_profiles.csv` | Mean raw features per material cluster |
| `outputs/data/clustering/stress_matrix.csv` | Scenario cluster × material cluster demand sums |
| `outputs/data/clustering/vif_scenarios.csv` | Final VIF values for scenario features |
| `outputs/data/clustering/vif_materials.csv` | Final VIF values for material features |
| `outputs/data/clustering/validation_metrics.txt` | k, silhouette, ARI, features used/dropped |
| `outputs/data/clustering/scenario_features_raw.csv` | Raw scenario features before preprocessing |
| `outputs/data/clustering/material_features_raw.csv` | Raw material features before preprocessing |

### Figure Files

All saved in `outputs/figures/clustering/` as PNG at 300 DPI:
`elbow_scenarios`, `elbow_materials`, `pca_biplot_scenarios`, `pca_biplot_materials`, `silhouette_scenarios`, `silhouette_materials`, `cluster_profiles_scenarios`, `cluster_profiles_materials`, `stress_matrix_demand`, `feature_sensitivity_scenarios`, `feature_sensitivity_materials`, `fig3_demand_sourcing`, `fig4_reserve_adequacy`, `fig4_reserve_adequacy_us`, `figSI_production_shares_crc`, `fig3_supply_risk_matrix`, `fig3b_us_supply_risk_matrix`.

---

## 17. Known Limitations

1. **No retirements within modeling window** for most technologies. Lifetimes (25–80 years) exceed the 24-year projection horizon. The model primarily captures new-build demand.

2. **Battery, DAC, and electrolyzer** technologies are unmapped — their material demands are missing from output.

3. **Constant material intensity** over time. No learning curves or technology evolution modeled.

4. **No recycling or material recovery** modeled. All demand is assumed primary/virgin material.

5. **Supply-chain data coverage** is incomplete: production data for 18/31 materials, import dependency for 21/31, CRC risk for 22/31. Missing materials get default values (0 for production, 1.0 for import dependency, 5.0 for CRC risk).

6. **Rare earth aggregation:** Dysprosium, Neodymium, Praseodymium, and Terbium all map to a single "Rare Earths" entry in the risk data. Individual rare earth supply-chain features are identical across these four materials.

7. **3-year interval capacity data** means demand is computed at discrete time steps, not annually. No interpolation between steps.

8. **VIF pruning aggressiveness:** For scenarios, VIF pruning reduces 16 features to ~2, meaning much of the feature engineering does not contribute to the final clustering. This is a consequence of high inter-correlation among demand-magnitude features.

9. **Material k=4 is manually set** rather than auto-selected. The silhouette-optimal k=2 produced unstable clustering (ARI ≈ 0.135).

10. **Single technology mix assumption for UPV:** The 90/7/3 split (c-Si/CdTe/CIGS) is fixed and does not vary by scenario or over time.
