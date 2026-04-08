# Raw Data Sources

**Purpose:** authoritative reference for every dataset consumed by the Monte Carlo simulation and clustering pipeline. Every row of `clustering_features.csv` traces back to one or more entries here. Methods section text should cite from this file rather than reinventing source descriptions.

**Maintained:** add a new section any time a new file enters `Python/materials_demand_model/data/`. Annotate the change date.

---

## 1. NREL Standard Scenarios 2024 (capacity projections)

| Field | Value |
|---|---|
| **Local file** | `data/StdScen24_annual_national.csv` |
| **Source** | National Renewable Energy Laboratory, *Standard Scenarios 2024 Report*, 2024 |
| **URL** | https://www.nrel.gov/analysis/standard-scenarios.html |
| **DOI** | n/a (annual data release) |
| **License** | Public domain (NREL data products) |
| **Retrieval date** | 2024 (locked at start of project) |
| **Coverage** | 61 scenarios × 26 years (2025–2050) × ~25 technology classes, US national totals |
| **Fields used** | `scenario`, `t` (year), `*_MW` columns (technology-specific capacity in MW) |
| **Loaded by** | `clustering/feature_engineering.load_nrel_data()` |
| **Caveats** | Skip first 3 header rows; scenarios named per NREL convention; tech-class column suffixes evolved across release years |

---

## 2. Material intensity data (technology → material content)

| Field | Value |
|---|---|
| **Local file** | `data/intensity_data.csv` |
| **Source** | Literature compilation (multiple sources, see PIPELINE_DOCUMENTATION.md §3) |
| **License** | Compiled in-house; original sources cited per row |
| **Coverage** | 21 raw technologies → 19 after consolidation (CDTE→CdTe, ASIGE→a-Si); 31 materials |
| **Units** | Originally t/GW; converted to t/MW in pipeline (divide by 1000) — see CLAUDE.md "Critical: Known Issues" |
| **Loaded by** | `src/data_loader.load_intensity_data()` |
| **Caveats** | Source data has typo "Gadium" for Gadolinium; preserved across the pipeline. Distribution fits use lognormal/gamma/truncnorm with strict acceptance criteria |

---

## 3. USGS Mineral Commodity Summaries 2025 — primary risk data

This is the workhorse risk dataset. It enters the pipeline through three different files for historical reasons.

### 3a. risk_charts_inputs.xlsx (legacy multi-sheet workbook)

| Field | Value |
|---|---|
| **Local file** | `data/risk_charts_inputs.xlsx` |
| **Source** | Compiled from USGS MCS 2025 + Census Bureau trade data + OECD CRC 2026 |
| **License** | Compiled in-house from public USGS / Census / OECD data |
| **Sheets** | `aggregate`, `import_dependency`, `import_shares`, `production`, `reserves`, `crc` |
| **Loaded by** | `clustering/feature_engineering.load_risk_data()` (with `usgs_mcs2025_loader` as preferred path) |
| **Caveats** | Production sheet covers 19 of 31 materials in our analysis (missing trace metals); see §3c for the fix |

### 3b. USGS MCS 2025 raw CSVs (loader path)

| Field | Value |
|---|---|
| **Local files** | `data/usgs_mcs_2025/*.csv` (one per commodity) + `data/census_trade/*.csv` |
| **Source** | USGS, 2025, *Mineral Commodity Summaries 2025*, https://pubs.usgs.gov/publication/mcs2025 |
| **License** | Public domain (US government work) |
| **Retrieval date** | 2025-Q1 |
| **Loaded by** | `clustering/usgs_mcs2025_loader.py` (`load_risk_data_mcs2025`, `load_thin_film_data_mcs2025`) |
| **Caveats** | Same coverage as 3a — these are the per-commodity raw extracts that 3a was assembled from |

### 3c. USGS MCS 2025 World Data Release CSV ⭐ (wired into clustering 2026-04-08)

| Field | Value |
|---|---|
| **Local file** | `data/usgs_mcs_2025/world_data/MCS2025_World_Data.csv` |
| **Local metadata** | `data/usgs_mcs_2025/world_data/MCS2025_World_Data.xml` |
| **Source** | U.S. Geological Survey, 2025, *Mineral Commodity Summaries 2025 World Data Release* |
| **DOI** | https://doi.org/10.5066/P13XCP3R |
| **ScienceBase URL** | https://www.sciencebase.gov/catalog/item/677eaf95d34e760b392c4970 |
| **License** | Public domain (US government work) |
| **Retrieval date** | 2026-04-08 |
| **Coverage** | 78 commodities × ~16 countries each = 1,250 rows. Per-country production (2023 actual + 2024 estimate), production capacity, and 2024 reserves |
| **Fields used** | `COMMODITY`, `COUNTRY`, `PROD_2023`, `PROD_EST_2024` (note USGS column name has a stray space: `PROD_EST_ 2024`) |
| **Loaded by** | `clustering/feature_engineering.load_mcs2025_world_data()` |
| **Existed prior to 2026-04-08?** | Yes — file was already in the canonical USGS MCS 2025 directory (see `data/README.md` §3b) but **was not consumed by the clustering feature engineering**. The 2026-04-08 fix wires it into `_build_production_hhi` and `_build_global_production_series` |
| **Why wired in** | Bug 2 (2026-04-08): `production_hhi` was silently defaulting to 0 for ~10 trace materials missing from 3a/3b. This dataset has per-country production for Cadmium, Gallium, Indium, Selenium, Tellurium, and aggregated Rare Earths, fixing 6 of the 10 affected materials directly + 2 more (Y, Gd) via the Rare Earths aggregate proxy |
| **Caveats** | (i) USGS has a typo: "Gemanium" instead of "Germanium" (preserved in `MCS2025_COMMODITY_MAP`); (ii) germanium values are all withheld (NaN) in the CSV — handled via hardcoded HHI from the MCS 2025 germanium chapter PDF; (iii) Y and Gd are not split out — use the aggregated "Rare earths" production HHI as a proxy with explicit methods note; (iv) does not cover Fiberglass or Glass (handled via hardcoded HHI = 0.10 with rationale) |

### 3d. Hardcoded production HHI (last-resort fallback)

| Material | HHI | Source |
|---|---|---|
| Germanium | 0.65 | USGS MCS 2025 Germanium chapter qualitative description (data withheld); China dominant; producing countries: US, Belgium, Canada, China, Germany, Russia. https://pubs.usgs.gov/periodicals/mcs2025/mcs2025-germanium.pdf |
| Fiberglass | 0.10 | Not tracked by USGS as critical commodity; ~100% domestic production from globally diversified silica sand |
| Glass | 0.10 | Same as Fiberglass |

Defined in `clustering/feature_engineering.HARDCODED_PRODUCTION_HHI`. Each entry must carry a citation.

---

## 4. USGS thin-film material CSVs (legacy 2023, kept as fallback)

| Field | Value |
|---|---|
| **Local files** | `data/mcs2023-*_salient.csv` (6 files: Cd, Ga, Ge, In, Se, Te) |
| **Source** | USGS Mineral Commodity Summaries 2023 |
| **License** | Public domain (US government work) |
| **Loaded by** | `clustering/feature_engineering.load_usgs_2023_thin_film()` (only used if MCS 2025 thin-film loader fails) |
| **Caveats** | Superseded by 3b/3c for routine use; preserved as fallback to avoid breaking historical reproducibility |

---

## 5. OECD Country Risk Classification (CRC) 2026

| Field | Value |
|---|---|
| **Local file** | `data/oecd_crc/oecd_crc_2026.csv` |
| **Source** | Organisation for Economic Co-operation and Development, *Country Risk Classifications of the Participants to the Arrangement on Officially Supported Export Credits*, January 2026 |
| **URL** | https://www.oecd.org/trade/topics/export-credits/arrangement-and-sector-understandings/financing-terms-and-conditions/country-risk-classification/ |
| **License** | OECD terms (free for non-commercial reuse with attribution) |
| **Retrieval date** | 2026-Q1 |
| **Coverage** | ~200 countries; CRC scale 0–7 (0 = lowest risk, 7 = highest). OECD members coded as "OECD" (treated as 1 in our scale); China coded separately (treated as 7) |
| **Fields used** | `country`, `crc` |
| **Loaded by** | Surface as the `crc` sheet via `load_risk_data_mcs2025()` |
| **Used in** | `hhi_wgi` (governance-weighted HHI), `reserves_high_risk_frac`, `import_high_risk_frac`, CRC mapping in `_build_crc_sourcing_breakdown` |

---

## 6. Census Bureau import shares (US imports by source country)

| Field | Value |
|---|---|
| **Local files** | `data/census_trade/*.csv` |
| **Source** | US Census Bureau, USA Trade Online, monthly trade statistics |
| **URL** | https://usatrade.census.gov/ |
| **License** | Public domain (US government work) |
| **Retrieval date** | 2025–2026 (updated quarterly) |
| **Loaded by** | `clustering/census_import_shares.py` (cached in `import_shares` sheet of risk_data) |
| **Used in** | `import_china_frac`, `import_hhi`, `hhi_wgi` |

---

## 7. Monte Carlo demand output (downstream, not raw)

| Field | Value |
|---|---|
| **Local file** | `outputs/data/material_demand_by_scenario.csv` |
| **Source** | Generated by `examples/run_simulation.py` |
| **Format** | Long: `(scenario, year, material, mean, std, p2, p5, p25, p50, p75, p95, p97)` |
| **Loaded by** | `clustering/feature_engineering.load_demand_data()` |
| **Note** | Not a raw dataset — included here for completeness because clustering features depend on it. Provenance is the simulation code itself, version-tagged in git |

---

## Citation hierarchy (PJ's standing rule)

When a single fact has multiple sources, use this priority:

1. **Crossref API** (DOI-resolved metadata) — for journal articles
2. **Zotero library** (`references/zotero_library.bib`) — for the curated set
3. **Original publication** (PDF or web page) — for primary data
4. **`.bib` file** — last resort

For data values specifically:
1. **USGS MCS 2025 World Data Release** (machine-readable) — preferred
2. **USGS MCS 2025 individual commodity chapters** (PDFs) — for commodities not in the World CSV
3. **USGS published NIR sheet** (in risk_charts_inputs.xlsx) — preferred over computed trade-balance NIR (the latter has the stage-asymmetry bug for REEs)
4. **Calculated trade-balance NIR** — fallback only
5. **Hardcoded values with citation** — last resort, must have a literature anchor

---

## Change log

| Date | Change | Reason |
|---|---|---|
| 2024 (project start) | Initial NREL StdScen24 + intensity_data.csv lock | Project foundation |
| 2025 | Migrated risk data from manual xlsx to USGS MCS 2025 raw CSVs (`usgs_mcs2025_loader.py`) | Reproducibility |
| 2026-04-08 | Added MCS 2025 World Data Release CSV (`MCS2025_World_Data.csv`) | Bug 2 fix: production_hhi for trace materials |
| 2026-04-08 | Inverted NIR priority order (published USGS sheet before computed trade-balance) + added stage-asymmetry guard | Bug 1 fix: REE NIR was reading 0 instead of 0.94 |
| 2026-04-08 | Added `HARDCODED_PRODUCTION_HHI` for germanium, fiberglass, glass | Bug 2 fix: materials USGS does not publish per-country data for |
| 2026-04-08 | Added Gadolinium ("Gadium") → Rare Earths mapping in `DEMAND_TO_RISK` | Closes a gap that left Gd unmapped |
