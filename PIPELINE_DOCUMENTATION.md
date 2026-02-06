# Pipeline Documentation: Materials Demand Model

> **Purpose:** Complete technical reference for all data processing and modeling steps.
> Intended for writing the Data Sources and Methods sections of the thesis proposal.

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
10. [Clustering: Feature Engineering](#10-clustering-feature-engineering)
11. [Clustering: Preprocessing](#11-clustering-preprocessing)
12. [Clustering: K-Means Analysis](#12-clustering-k-means-analysis)
13. [Supply Chain Risk Analysis](#13-supply-chain-risk-analysis)
14. [Validation and Diagnostics](#14-validation-and-diagnostics)
15. [Known Limitations](#15-known-limitations)

---

## 1. Pipeline Overview

The pipeline has two major stages:

**Stage A — Monte Carlo Demand Simulation** (`src/` + `examples/run_simulation.py`):
Load material intensity data and NREL capacity projections → fit parametric distributions to intensity values → build a stock-flow model of capacity additions → run 10,000 Monte Carlo iterations sampling from fitted distributions → output demand statistics (mean, std, percentiles) per scenario × year × material.

**Stage B — Clustering Analysis** (`clustering/`):
Load the demand output from Stage A plus supply-chain risk data → engineer scenario-level and material-level features → preprocess (log transform, VIF pruning, z-score standardization) → K-means clustering with silhouette-based k selection → validation (stability ARI, feature sensitivity) → export cluster assignments, profiles, and stress matrix.

### Key Dimensions

| Dimension | Count | Source |
|-----------|-------|--------|
| Scenarios | 61 | NREL Standard Scenarios 2024 |
| Years | 9 | 2026, 2029, 2032, 2035, 2038, 2041, 2044, 2047, 2050 |
| Materials | 31 | Derived from intensity data tech-material pairs |
| Capacity technologies | 23 | NREL StdScen24 columns ending `_MW` |
| Intensity technologies | 18 mapped (5 unmapped) | `technology_mapping.py` |
| MC iterations | 10,000 | `examples/run_simulation.py` |

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
| `upv` | utility-scale solar pv / CIGS / CdTe | 0.70 / 0.15 / 0.15 | 30 |
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

**Key design choice:** `upv` (utility-scale PV) uses a 70/15/15 split across crystalline silicon, CIGS, and CdTe thin-film technologies. This is the only capacity technology with a multi-technology mapping.

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

**Orchestration:** `examples/run_simulation.py`

### Process

1. **Build stock-flow model** once (deterministic; depends only on capacity projections and lifetimes)
2. **For each of 10,000 iterations:**
   a. Sample one intensity value per (intensity_technology, material) pair from its fitted distribution
   b. Calculate material demand for all (scenario, year, material) combinations using the sampled intensities
   c. Store in 4D array: `results[iteration, scenario, year, material]`
3. **Compute statistics** across the iteration axis:
   - Mean, standard deviation
   - Percentiles: 2.5, 5, 25, 50, 75, 95, 97.5

### Parameters

- `n_iterations = 10,000`
- `random_seed = 42`
- All 61 scenarios run by default (configurable via `FOCUS_SCENARIOS`)

### Result Shape

`iterations_data` array shape: `(10000, 61, 9, 31)` = iterations × scenarios × years × materials

---

## 9. Output: Demand Data

### Primary Output

- **File:** `outputs/material_demand_by_scenario.csv`
- **Columns:** `scenario`, `year`, `material`, `mean`, `std`, `p2` (p2.5), `p5`, `p25`, `p50`, `p75`, `p95`, `p97` (p97.5)
- **Rows:** 61 scenarios × 9 years × 31 materials = 17,019 rows
- **Units:** tonnes (metric tons)
- **Interpretation:** For each scenario-year-material triple, the statistics summarize the distribution of demand across 10,000 MC iterations. Uncertainty derives from material intensity distributions — each iteration uses the same capacity projections.

### Secondary Output

- **File:** `outputs/material_demand_summary.csv`
- **Aggregation:** Sums across all scenarios for each (year, material) combination

---

## 10. Clustering: Feature Engineering

**Code:** `clustering/feature_engineering.py`

### 10.1 Scenario Features (12 features, 61 scenarios)

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
| 8 | `mean_ci_width` | Mean relative CI width: (p95 − p5) / mean across all material-year rows | lines 433–435 |
| 9 | `solar_fraction_2035` | (upv + distpv + csp capacity) / total capacity at 2035 | lines 438–446 |
| 10 | `wind_fraction_2035` | (wind_onshore + wind_offshore capacity) / total capacity at 2035 | line 447 |
| 11 | `storage_fraction_2035` | (battery + pumped-hydro capacity) / total capacity at 2035 | line 448 |
| 12 | `n_active_materials` | Number of materials with non-zero demand in that scenario | lines 451–456 |

**Note:** "Total demand" for a scenario-year means `sum of mean demand across all 31 materials` for that (scenario, year) pair.

### 10.2 Material Features (23 features, 31 materials)

#### Demand-derived features (7)

| # | Feature | Formula | Code location |
|---|---------|---------|---------------|
| 1 | `mean_demand` | Grand mean of `mean` across all scenarios and years | line 499 |
| 2 | `peak_demand` | Max `mean` across all scenarios and years | line 502 |
| 3 | `scenario_cv` | For each scenario: sum `mean` across years → compute CV across scenarios | lines 505–513 |
| 4 | `mean_ci_width` | Mean of (p95 − p5) / mean across all scenario-year rows | lines 516–518 |
| 5 | `demand_volatility` | Std of `mean` across years (within each scenario), then averaged across scenarios | lines 521–525 |
| 6 | `demand_slope` | Average slope of mean demand vs. year (averaged across scenarios) | lines 528–535 |
| 7 | `n_active_scenarios` | Number of scenarios with non-zero demand for that material | lines 538–542 |

#### Supply-chain features (16)

| # | Feature | Source | Formula | Coverage |
|---|---------|--------|---------|----------|
| 8 | `domestic_production` | aggregate sheet + USGS 2023 CSVs | Average annual US production (tonnes) | 18/31 |
| 9 | `import_dependency` | import_dependency sheet + USGS NIR | 0–1 fraction (1 = fully imported). Default 1.0 if missing | 21/31 |
| 10 | `crc_weighted_risk` | import_shares + crc sheets | Σ(import_share × crc_weight) / Σ(import_share). Weights: US=0, OECD=1, CRC1=2, ..., CRC7=8, China=7, Undefined=5. Scale 0–8. Default 5.0 if missing | 22/31 |
| 11 | `mean_capacity_ratio` | demand + production | mean_scenario_total_demand / US_production | 18/31 |
| 12 | `max_capacity_ratio` | demand + production | max_scenario_total_demand / US_production | 18/31 |
| 13 | `exceedance_frequency` | demand + production | Fraction of scenarios where total demand > US production | 18/31 |
| 14 | `reserve_depletion_rate` | demand + reserves | peak_demand / global_reserves_kt. 0 if no reserves data | varies |
| 15 | `domestic_reserves_years` | reserves + demand | US_reserves_kt / peak_demand | varies |
| 16 | `global_reserves_years` | reserves + demand | global_reserves_kt / peak_demand | varies |
| 17 | `reserves_high_risk_frac` | reserves + crc | Fraction of global reserves in CRC 5–7 + China | varies |
| 18 | `reserves_oecd_frac` | reserves + crc | Fraction in OECD + United States | varies |
| 19 | `reserves_china_frac` | reserves + crc | Fraction in China | varies |
| 20 | `import_china_frac` | import_shares + crc | Fraction of imports from China | varies |
| 21 | `import_high_risk_frac` | import_shares + crc | Fraction from CRC 5–7 + China | varies |
| 22 | `import_oecd_frac` | import_shares + crc | Fraction from OECD | varies |
| 23 | `import_hhi` | import_shares | Herfindahl-Hirschman Index of import country concentration. HHI = Σ(share²), 0–1 scale (higher = more concentrated) | varies |

**Material name mapping:** `DEMAND_TO_RISK` in `clustering/config.py:62–73` maps 22 demand material names to 19 risk material names (4 rare earth elements → aggregate "Rare Earths" entry). Materials not in this mapping get default/zero values for supply-chain features.

**Missing value handling:** All inf/NaN replaced with 0 at the end (`feature_engineering.py:639`).

---

## 11. Clustering: Preprocessing

**Code:** `clustering/preprocessing.py` → `preprocess_pipeline()`

### Pipeline Steps

1. **Log transformation:** `log10(clip(x, 0) + 1)` applied in-place to specified columns.

   Scenario log features: `total_cumulative_demand`, `peak_demand`, `mean_demand_early`

   Material log features: `mean_demand`, `peak_demand`, `domestic_production`, `mean_capacity_ratio`, `max_capacity_ratio`, `domestic_reserves_years`, `global_reserves_years`

2. **Drop zero-variance columns:** Any column with `std == 0` removed.

3. **VIF check (before pruning):** Compute Variance Inflation Factor for all remaining features. VIF_j = 1 / (1 − R²_j) where R²_j is from regressing feature j on all other features. Uses `statsmodels.stats.outliers_influence.variance_inflation_factor`.

4. **Iterative VIF pruning:** While any feature has VIF > 10: drop the feature with the highest VIF. Repeat. This removes multicollinear features.

5. **Z-score standardization:** `(x − mean) / std` via `sklearn.preprocessing.StandardScaler`. Applied after log transform and VIF pruning.

### Typical Pruning Results

**Scenarios:** Starting from 12 features, VIF pruning typically retains ~2 features (e.g., `demand_slope`, `storage_fraction_2035`). Most demand-magnitude features are highly correlated and dropped.

**Materials:** Starting from 23 features, VIF pruning typically retains ~11 features (e.g., `mean_ci_width`, `demand_volatility`, `demand_slope`, `domestic_production`, `import_dependency`, `reserve_depletion_rate`, `domestic_reserves_years`, `global_reserves_years`, `reserves_oecd_frac`, `reserves_china_frac`, `import_china_frac`).

---

## 12. Clustering: K-Means Analysis

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

## 13. Supply Chain Risk Analysis

**Code:** `clustering/supply_chain_analysis.py`

Standalone script generating two manuscript-style figures.

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
3. Compute years of reserves: `(reserves_kt × 1000) / peak_demand` per CRC category
4. Stack bars by CRC category

---

## 14. Validation and Diagnostics

### Monte Carlo Simulation
- Random seed = 42 for reproducibility
- 10,000 iterations (convergence assumed; no formal convergence diagnostic)
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
| `outputs/clustering/results/scenario_clusters.csv` | scenario, cluster, silhouette |
| `outputs/clustering/results/material_clusters.csv` | material, cluster, silhouette |
| `outputs/clustering/results/scenario_cluster_profiles.csv` | Mean raw features per scenario cluster |
| `outputs/clustering/results/material_cluster_profiles.csv` | Mean raw features per material cluster |
| `outputs/clustering/results/stress_matrix.csv` | Scenario cluster × material cluster demand sums |
| `outputs/clustering/results/vif_scenarios.csv` | Final VIF values for scenario features |
| `outputs/clustering/results/vif_materials.csv` | Final VIF values for material features |
| `outputs/clustering/results/validation_metrics.txt` | k, silhouette, ARI, features used/dropped |
| `outputs/clustering/results/scenario_features_raw.csv` | Raw scenario features before preprocessing |
| `outputs/clustering/results/material_features_raw.csv` | Raw material features before preprocessing |

### Figure Files

All saved in `outputs/clustering/figures/` as both PNG and PDF at 300 DPI:
`elbow_scenarios`, `elbow_materials`, `pca_biplot_scenarios`, `pca_biplot_materials`, `silhouette_scenarios`, `silhouette_materials`, `cluster_profiles_scenarios`, `cluster_profiles_materials`, `stress_matrix_demand`, `feature_sensitivity_scenarios`, `feature_sensitivity_materials`, `fig3_demand_sourcing`, `fig4_reserve_adequacy`.

---

## 15. Known Limitations

1. **No retirements within modeling window** for most technologies. Lifetimes (25–80 years) exceed the 24-year projection horizon. The model primarily captures new-build demand.

2. **Battery, DAC, and electrolyzer** technologies are unmapped — their material demands are missing from output.

3. **Constant material intensity** over time. No learning curves or technology evolution modeled.

4. **No recycling or material recovery** modeled. All demand is assumed primary/virgin material.

5. **Supply-chain data coverage** is incomplete: production data for 18/31 materials, import dependency for 21/31, CRC risk for 22/31. Missing materials get default values (0 for production, 1.0 for import dependency, 5.0 for CRC risk).

6. **Rare earth aggregation:** Dysprosium, Neodymium, Praseodymium, and Terbium all map to a single "Rare Earths" entry in the risk data. Individual rare earth supply-chain features are identical across these four materials.

7. **3-year interval capacity data** means demand is computed at discrete time steps, not annually. No interpolation between steps.

8. **VIF pruning aggressiveness:** For scenarios, VIF pruning reduces 12 features to ~2, meaning much of the feature engineering does not contribute to the final clustering. This is a consequence of high inter-correlation among demand-magnitude features.

9. **Material k=4 is manually set** rather than auto-selected. The silhouette-optimal k=2 produced unstable clustering (ARI ≈ 0.135).

10. **Single technology mix assumption for UPV:** The 70/15/15 split (c-Si/CIGS/CdTe) is fixed and does not vary by scenario or over time.
