# Data Provenance

This directory contains input data for a Monte Carlo simulation model of critical material demand arising from U.S. energy technology deployment. The model combines technology deployment scenarios with material intensity distributions to project future material requirements under uncertainty.

All datasets used in this project are derived from publicly available sources. Every data file is traceable to a citable DOI or institutional URL.

**Last updated:** April 2026

---

## 1. Material Intensity Data

| Attribute     | Detail |
|---------------|--------|
| **File**      | `intensity_data.csv` |
| **Format**    | CSV |
| **Rows**      | ~1,326 |
| **Key Columns** | `technology`, `Material`, `value` |
| **Source**    | Literature review compilation of published material intensity studies |
| **License/Terms** | Individual study terms apply; values are factual measurements |

**Technologies covered:** a-Si, CdTe, CIGS, c-Si (utility-scale solar PV, distributed solar), onshore wind, offshore wind, Nuclear New, Hydro, CSP, Biomass, Coal, Gas (NGCC, NGGT), Coal CCS, Gas CCS, Bio CCS, Geothermal, and others.

**Materials covered:** Copper, Silicon, Tellurium, Cadmium, Indium, Gallium, Germanium, Silver, Aluminum, Steel, Concrete, Rare Earth Elements, Lithium, Cobalt, Nickel, Manganese, Graphite, Selenium, Chromium, Molybdenum, Tin, and others.

**Important note:** This file contains multiple measurements per technology-material pair, representing different values reported across the published literature. These multiple observations are used for statistical distribution fitting (parametric) in the Monte Carlo simulation, enabling the model to capture uncertainty in material intensity values rather than relying on single point estimates. Values are in tonnes per GW of installed capacity (converted to t/MW during model execution).

**Technology consolidation note:** The raw data contains some technologies split into cell-specific and balance-of-system (BOS) entries. Specifically, `CDTE` (15 rows) contains BOS materials for CdTe thin-film panels, and `ASIGE` (15 rows) contains BOS materials for a-Si panels. During preprocessing (see `src/data_ingestion.py`), these are consolidated into their parent technologies (`CdTe` and `a-Si` respectively) so that each technology has complete material coverage. The raw CSV is not modified; consolidation occurs at load time. After consolidation, 19 unique technologies are available for analysis (reduced from 21 in the raw file).

**Citation:**

> Material intensity values compiled from published literature. Individual sources span multiple peer-reviewed studies on material requirements for energy technologies. See project documentation for the full list of source publications.

---

## 2. NREL Standard Scenarios 2024

| Attribute     | Detail |
|---------------|--------|
| **File**      | `StdScen24_annual_national.csv` |
| **Format**    | CSV (with 4 header/metadata rows) |
| **Rows**      | ~549 (data rows across 61 scenarios) |
| **Key Columns** | `scenario`, `policy`, `t` (year), technology capacity columns (e.g., `upv_MW`, `wind_onshore_MW`, `wind_offshore_MW`, `nuclear_MW`, `hydro_MW`, `battery_4_MW`, `csp_MW`, etc.), generation columns (`*_MWh`), emissions columns |
| **Source**    | National Renewable Energy Laboratory (NREL), Standard Scenarios 2024 |
| **License/Terms** | Publicly available; U.S. Government work product |

**Description:** National-level annual projections of U.S. electricity generation capacity, generation, and emissions from approximately 2026 to 2050, spanning 61 scenarios that cover a range of technology cost assumptions, policy environments, and demand growth trajectories. Capacity values are in MW; generation values in MWh.

**Citation:**

> National Renewable Energy Laboratory. 2024. "2024 Standard Scenarios." Golden, CO: National Renewable Energy Laboratory. NREL/TP-6A40-92256. https://www.nrel.gov/analysis/standard-scenarios.html

---

## 3. USGS Mineral Commodity Summaries 2025 (Primary Supply Chain Data)

| Attribute     | Detail |
|---------------|--------|
| **Directory** | `usgs_mcs_2025/` |
| **DOI**       | **10.5066/P13XCP3R** |
| **Version**   | 2.0 (April 2025) |
| **Downloaded**| April 2026 |
| **Source**    | U.S. Geological Survey, ScienceBase |
| **License/Terms** | Public domain; U.S. Government work product |

This is the **primary source for all supply chain data** in the pipeline, replacing the previously hand-compiled `risk_charts_inputs.xlsx`.

### 3a. Salient Commodity Data

| Attribute     | Detail |
|---------------|--------|
| **Directory** | `usgs_mcs_2025/salient_commodity/` |
| **Files**     | 85 commodity CSVs (`mcs2025-{commodity}_salient.csv`) + XML metadata |
| **Data years**| 2020-2024 (5-year salient statistics) |
| **Key Columns** | `Year`, `USprod_*` (US production), `Imports_*`, `Exports_*`, `Consump_*` (consumption), `NIR_*` (net import reliance %), `Price_*`, `Stocks_*`, `Employment_*` |

**Materials used in this project (19 core + 6 thin-film):**

| Our Material Name | USGS CSV Prefix | Production Unit | NIR Column |
|-------------------|-----------------|-----------------|------------|
| Aluminum | alumi | kt (primary smelter) | NIR_pct |
| Boron | boron | kt | NIR_pct |
| Cadmium | cadmi | t (refinery) | NIR_Metal_pct |
| Cement | cemen | kt | NIR_pct |
| Chromium | chrom | kt | NIR_pct |
| Copper | coppe | kt (refinery primary+secondary) | NIR_pct |
| Gallium | galli | kg | NIR_pct |
| Germanium | germa | kg | NIR_ct |
| Indium | indiu | t | NIR_pct |
| Lead | lead | kt (mine) | NIR_Metal_pct |
| Magnesium | mgmet | kt | NIR_pct |
| Manganese | manga | kt | NIR_pct |
| Molybdenum | molyb | t | NIR_pct |
| Nickel | nicke | t (mine) | NIR_ct |
| Niobium | niobi | t | NIR_pct |
| Rare Earths | rareee | t | (from NIR figure data) |
| Selenium | selen | t | NIR_pct |
| Silicon | simet | kt (FeSi+Si metal) | NIR_FeSi-Si_pct |
| Silver | silve | t | NIR_pct |
| Steel | feste | mmt (raw steel) | NIR_pct |
| Tellurium | tellu | t | NIR_pct |
| Tin | tin | t | NIR_Refined_pct |
| Vanadium | vanad | t | NIR_pct |
| Yttrium | yttri | t | NIR_pct |
| Zinc | zinc | kt (refined) | NIR_Refined_pct |

**Unit handling:** Column name suffixes indicate units: `_kt` = thousand metric tons, `_t` = metric tons, `_mmt` = million metric tons, `_kg` = kilograms. The loader (`clustering/usgs_mcs2025_loader.py`) converts all values to kt for internal consistency.

### 3b. World Production, Capacity, and Reserves

| Attribute     | Detail |
|---------------|--------|
| **File**      | `usgs_mcs_2025/world_data/MCS2025_World_Data.csv` |
| **Key Columns** | `COMMODITY`, `COUNTRY`, `TYPE`, `UNIT_MEAS`, `PROD_2023`, `PROD_EST_ 2024`, `CAP_2023`, `CAP_EST_ 2024`, `RESERVES_2024`, `RESERVE_NOTES` |
| **Coverage**  | 77 commodities, up to 20+ countries per commodity |

**Reserve unit note:** The `UNIT_MEAS` column applies to production/capacity columns. For reserves, MOST commodities use the same unit, but some (Molybdenum, Vanadium) use a different unit flagged in `RESERVE_NOTES` (e.g., "Reserve data is thousand metric tons"). The loader checks `RESERVE_NOTES` to determine the correct conversion factor. All reserves are stored internally in kt (thousand metric tons).

### 3c. Industry Trends and Statistics

| Attribute     | Detail |
|---------------|--------|
| **Directory** | `usgs_mcs_2025/industry_trends/` |
| **Key Files** | `MCS2025_Fig2_Net_Import_Reliance.csv` (NIR rankings + import source countries), `MCS2025_Fig3_Major_Import_Sources.csv`, `MCS2025_T5_Critical_Minerals_Salient.csv` |
| **Used for**  | NIR rankings, import source country identification, critical mineral classification |

### Citation

> U.S. Geological Survey, 2025, U.S. Geological Survey Mineral Commodity Summaries 2025 data release (ver. 2.0, April 2025): U.S. Geological Survey data release, https://doi.org/10.5066/P13XCP3R.

> U.S. Geological Survey, 2025, Mineral commodity summaries 2025 (ver. 1.2, March 2025): U.S. Geological Survey, 212 p., https://doi.org/10.3133/mcs2025.

---

## 4. OECD Country Risk Classifications (January 2026)

| Attribute     | Detail |
|---------------|--------|
| **File**      | `oecd_crc/oecd_crc_2026.csv` |
| **Format**    | CSV (parsed from OECD PDF) |
| **Rows**      | 201 countries |
| **Columns**   | `country`, `iso3`, `crc`, `notes` |
| **Valid as of**| 30 January 2026 |
| **Source**    | OECD Arrangement on Officially Supported Export Credits |
| **Source URL**| https://www.oecd.org/en/topics/sub-issues/country-risk-classification.html |
| **License/Terms** | Publicly available |

**CRC scale:** 0 = high-income OECD (shown as "-" in source), 1-7 = increasing country risk. In our pipeline, these are mapped to a geopolitical risk weight: US=0, OECD=1, CRC1=2, ..., CRC7=8, China=7, Undefined=5.

**Raw vs derived files:**
- `cre-crc-current-english.pdf` — **Raw source.** Downloaded directly from OECD.
- `oecd_crc_2026.csv` — **Derived.** Parsed from the PDF using `pdfplumber`. Countries with CRC = "-" (high-income/OECD) are mapped to CRC = 0. To re-derive, run the parsing logic in `clustering/usgs_mcs2025_loader.py`.

### Citation

> OECD, 2026, Country risk classifications of the Participants to the Arrangement on Officially Supported Export Credits (valid as of 30 January 2026), Organisation for Economic Co-operation and Development, Paris. https://www.oecd.org/en/topics/sub-issues/country-risk-classification.html

---

## 5. U.S. Census Bureau International Trade Data (Import Shares)

| Attribute     | Detail |
|---------------|--------|
| **Directory** | `census_trade/` |
| **Cache file**| `census_trade/import_shares_cache.json` |
| **Source**    | U.S. Census Bureau, Foreign Trade Division |
| **API**       | `api.census.gov/data/timeseries/intltrade/imports/hs` |
| **Data years**| 2020-2023 (4-year average) |
| **Coverage**  | 25 materials (19 core + 6 thin-film), all source countries |
| **License**   | Public domain; U.S. Government work product |

**Purpose:** Provides country-level U.S. import shares by value for computing the Herfindahl-Hirschman Index (HHI) of import source concentration and CRC-weighted geopolitical risk scores. Replaces the hand-compiled `import_shares` sheet from `risk_charts_inputs.xlsx`.

**How it works:** Each material is mapped to one or more HTS/HS codes (see `clustering/census_import_shares.py:HTS_CODES`). The loader queries general import values (GEN_VAL_YR) by partner country, sums across codes for multi-code materials, averages over 4 years, and computes percentage shares.

**To refresh:** Run `python clustering/census_import_shares.py --no-cache` (or delete `census_trade/import_shares_cache.json`).

### Citation

> U.S. Census Bureau, Foreign Trade Division, USA Trade Online, https://www.census.gov/foreign-trade/data/index.html

---

## 6. Legacy Supply Chain Data

### 6a. Risk Charts Inputs (Fully Superseded)

| Attribute     | Detail |
|---------------|--------|
| **File**      | `supply_chain/risk_charts_inputs.xlsx` |
| **Status**    | **Fully superseded.** All sheets now replaced: aggregate/import_dependency/production/reserves by MCS 2025 CSVs (Section 3), import_shares by Census Bureau API (Section 5), crc by OECD CRC CSV (Section 4). Retained for validation only. |
| **Data years**| 2018-2022 |
| **Sheets**    | aggregate, import_dependency, production, reserves, import_shares, crc |

### 5b. USGS 2023 Thin-Film CSVs (Superseded)

| Attribute     | Detail |
|---------------|--------|
| **Files**     | `supply_chain/mcs2023-*_salient.csv` |
| **Status**    | **Superseded** by MCS 2025 salient commodity CSVs (Section 3a) |
| **Materials** | Cadmium, Gallium, Germanium, Indium, Selenium, Tellurium |

### ~~5c. USGS Aggregated Input~~ (Removed)

`input_usgs.csv` was removed in April 2026. It contained aggregated US production/consumption data for 7 commodities (2018-2022) from MCS 2023. This data is now fully covered by the MCS 2025 salient CSVs (Section 3a) with more materials, more variables, and more recent years (2020-2024).

---

## Data Loading Architecture

```
data/
├── intensity_data.csv              ← Material intensities (t/GW)
├── StdScen24_annual_national.csv   ← NREL 61 scenarios (2026-2050)
│
├── usgs_mcs_2025/                  ← PRIMARY supply chain source
│   ├── Salient_Commodity.zip       ← RAW: original USGS ZIP (DOI: 10.5066/P13XCP3R)
│   ├── World_Data.zip              ← RAW: original USGS ZIP
│   ├── Industry_Trends.zip         ← RAW: original USGS ZIP
│   ├── salient_commodity/          ← RAW: unzipped, per-commodity US stats (2020-2024)
│   │   ├── mcs2025-alumi_salient.csv
│   │   ├── mcs2025-coppe_salient.csv
│   │   └── ... (85 commodities, unmodified from USGS)
│   ├── world_data/
│   │   └── MCS2025_World_Data.csv  ← RAW: global production + reserves by country
│   ├── industry_trends/
│   │   ├── MCS2025_Fig2_Net_Import_Reliance.csv
│   │   └── ... (NIR, import sources, critical minerals)
│   └── DATA_SOURCES.md             ← Detailed source documentation
│
├── oecd_crc/
│   ├── cre-crc-current-english.pdf ← RAW: OECD source PDF (Jan 2026)
│   └── oecd_crc_2026.csv           ← DERIVED: parsed from PDF, 201 countries, CRC 0-7
│
├── census_trade/                   ← Import shares from Census Bureau API
│   └── import_shares_cache.json    ← CACHED: 25 materials, all source countries, 2020-2023
│
└── supply_chain/                   ← LEGACY (fully superseded, retained for validation)
    ├── risk_charts_inputs.xlsx
    └── mcs2023-*_salient.csv
```

The loader module `clustering/usgs_mcs2025_loader.py` reads from `usgs_mcs_2025/` and `oecd_crc/` for most supply chain data, and from `census_trade/` (via `clustering/census_import_shares.py`) for country-level import shares. The legacy `supply_chain/risk_charts_inputs.xlsx` is used only as a fallback if the Census API is unavailable.

---

## Reproducibility Checklist

To reproduce the supply chain data from scratch:

1. **USGS MCS 2025 Data Release:**
   - Go to https://doi.org/10.5066/P13XCP3R
   - Download all three ZIP archives (Salient, World Data, Industry Trends)
   - Unzip into `usgs_mcs_2025/salient_commodity/`, `usgs_mcs_2025/world_data/`, `usgs_mcs_2025/industry_trends/`

2. **OECD CRC:**
   - Go to https://www.oecd.org/en/topics/sub-issues/country-risk-classification.html
   - Download the current classifications PDF
   - Parse with `pdfplumber` (see parsing code in `clustering/usgs_mcs2025_loader.py` comments)
   - Save as `oecd_crc/oecd_crc_2026.csv`

3. **NREL Standard Scenarios:**
   - Go to https://www.nrel.gov/analysis/standard-scenarios.html
   - Download the 2024 national annual data CSV

4. **Material intensities:**
   - Compiled from published literature (see `Scientific_Draft/sections/citations.py` for full source list)

---

## Data Access

All source data used in this project are publicly available:

| Source | URL | DOI |
|--------|-----|-----|
| USGS MCS 2025 Data Release | https://www.sciencebase.gov/catalog/item/677eaf95d34e760b392c4970 | 10.5066/P13XCP3R |
| USGS MCS 2025 Publication | https://pubs.usgs.gov/publication/mcs2025 | 10.3133/mcs2025 |
| OECD Country Risk Classifications | https://www.oecd.org/en/topics/sub-issues/country-risk-classification.html | -- |
| NREL Standard Scenarios 2024 | https://www.nrel.gov/analysis/standard-scenarios.html | -- |
| NREL Materials Database | https://doi.org/10.2172/1995804 | 10.2172/1995804 |

No proprietary or access-restricted data are used in this analysis.
