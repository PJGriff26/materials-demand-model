# Clustering Feature Reference

**Source of truth:** [`clustering_features.csv`](clustering_features.csv) (machine-readable, github-tracked)
**Locked:** 2026-04-08
**Receipts:** `outputs/data/clustering/_corr_*.csv`, `outputs/data/clustering/_vif_*.csv`, `outputs/figures/clustering/_corr_*.png`

This document is the GitHub-tracked, comprehensive reference for the features used in the scenario and material clustering analyses. Every entry has either a literature citation or a one-sentence defense of the phenomenon it exposes.

After reduction (PI guidance 2026-04-03, 2026-04-08):
- **Scenario set:** 15 → 8 features. Max VIF 12.3, max |r| 0.84.
- **Material set:** 23 → 13 features. Max VIF 8.9 (in-pipeline), max |r| 0.69.

---

## Scenario features (n=61, p=8)

| # | Category | Feature | Units | What it exposes | Citation / rationale |
|---|---|---|---|---|---|
| 1 | Demand growth | `growth_rate_short_pct` | %/yr | Near-term ramp pressure to 2035 (IRA, RPS targets) | CAGR convention; IEA WEO 2024 |
| 2 | Demand growth | `growth_rate_long_pct` | %/yr | Full-horizon trajectory through 2050 | CAGR convention |
| 3 | Demand growth | `peak_annual_growth_short_pct` | %/yr | Spike vs gradual buildout (PJ pre-PI insight) |Captures qualitative scenario distinction |
| 4 | Uncertainty | `mean_cv` | dimensionless | Cross-MC dispersion under intensity uncertainty | MC convention; Graedel 2012 |
| 5 | Tech mix | `solar_fraction_2035` | 0–1 | Tech-mix snapshot at policy target year | NREL StdScen24 |
| 6 | Tech mix | `wind_fraction_2035` | 0–1 | Tech-mix snapshot at policy target year | NREL StdScen24 |
| 7 | Tech mix | `solar_fraction_2050` | 0–1 | End-state tech mix; distinguishes early-vs-late solar buildouts | NREL StdScen24 (PI request 2026-04-03) |
| 8 | Tech mix | `wind_fraction_2050` | 0–1 | End-state tech mix | NREL StdScen24 (PI request 2026-04-03) |

### Flattening discipline (scenario-level features)
Per-material CAGRs are computed first, then **averaged across materials** (not summed in tonnes). This is the "flattened" pattern PI emphasized 2026-04-03 — % growth rates are dimensionless and commensurable, while tonnes are not.

### Defended scenario omissions
| Dropped | Why |
|---|---|
| `storage_fraction_2035`, `storage_fraction_2050` | Compositional with solar+wind, smallest driver, no independent signal |
| `supply_chain_stress`, `peak_supply_chain_stress`, `mean_n_exceeding_production`, `peak_n_exceeding_production` | Cannot be grounded without arbitrary cross-material weighting (PI 2026-04-08); supply-chain analysis lives entirely at material level |
| `mean_ci_width` | r=0.997 with `mean_cv` (same dispersion-relative-to-mean construct) |
| `peak_annual_growth_long_pct` | PI flagged as redundant with `growth_long` + `peak_short`; VIF receipt confirms |
| `wind_fraction_2035` (in-pipeline) | r=0.84 with `wind_fraction_2050`; auto-dropped at VIF threshold 10 |

---

## Material features (n=31, p=13)

| # | Category | Feature | Units | What it exposes | Scope | Citation / rationale |
|---|---|---|---|---|---|---|
| 1 | Demand growth | `growth_rate_long_pct` | %/yr | Material-level demand trajectory through 2050 | demand=US | CAGR convention |
| 2 | Uncertainty | `scenario_cv` | dimensionless | Cross-scenario disagreement; "scenario-sensitive" materials | demand=US | MC convention |
| 3 | Trade exposure | `import_dependency` | 0–1 | Net Import Reliance (NIR) | **US** | **USGS Mineral Commodity Summaries** methodology |
| 4 | Concentration | `hhi_wgi` | 0–1 | Governance-weighted concentration of US import sources | US imports × global CRC | **Blengini et al. 2017 (EU CRM); Schrijvers et al. 2020** |
| 5 | Concentration | `production_hhi` | 0–1 | Geographic concentration of *production* | **global** | **Graedel et al. 2012 ES&T; EU CRM** |
| 6 | Concentration | `import_hhi` | 0–1 | Diversification of US import sources | US imports | **Graedel et al. 2012** |
| 7 | Demand vs production | `us_capacity_ratio` | dimensionless | US demand pressure on US domestic production | **US/US** | Habib & Wenzel 2014 demand-vs-production framework |
| 8 | Demand vs production | `global_capacity_ratio` | dimensionless | US demand alone vs global supply base — single-country squeeze | **US/global** | Demand-vs-production framework (PI 2026-04-08: need both) |
| 9 | Reserves | `global_reserve_coverage` | dimensionless (years-equiv) | How many years of US transition demand the global reserve base supports | **global/US** (mixing flagged) | USGS reserve reporting; Graedel 2012 static framework |
| 10 | Reserves | `domestic_reserve_share` | 0–1 | US share of global reserves (geographic self-sufficiency, demand-independent) | reserves only | Replaces `domestic_reserve_coverage` (r=1.000 collinearity from shared cumulative_demand denominator) |
| 11 | Reserves geography | `reserves_china_frac` | 0–1 | China's share of global reserves | global | USGS + OECD CRC 2026 |
| 12 | Reserves geography | `reserves_high_risk_frac` | 0–1 | Reserve concentration in high-political-risk jurisdictions | global | OECD Country Risk Classification |
| 13 | Import sourcing | `import_china_frac` | 0–1 | Direct US dependence on a single high-risk supplier | US imports | Census Bureau import shares |

### Defended material omissions
| Dropped | Why |
|---|---|
| `peak_annual_growth_short_pct`, `peak_annual_growth_long_pct` | CAGR captures the trend; YoY peak is single-year noise dominated by small bases |
| `mean_ci_width` | Same construct as `scenario_cv` |
| `demand_volatility_cv` | r=0.86 with `growth_rate_short_pct`, r=0.71 with `scenario_cv` — re-encodes growth signal |
| `growth_rate_short_pct` (material) | r=0.90 with `scenario_cv`; the 2035 horizon is a *policy* construct meaningful at scenario level only |
| `mean_capacity_ratio`, `max_capacity_ratio`, `exceedance_frequency` | Three overlapping forms of demand/production; replaced by clean US/global capacity ratio pair |
| `reserve_consumption_pct` | Mathematical inverse of `global_reserve_coverage` |
| `domestic_reserve_coverage` (original) | r=1.000 with `global_reserve_coverage` — both shared cumulative_demand denominator. Replaced with `domestic_reserve_share` (US/global reserves), demand-independent |
| `reserves_oecd_frac` | Compositional with `reserves_china_frac` + `reserves_high_risk_frac` |
| `import_high_risk_frac`, `import_oecd_frac` | Compositional with `import_china_frac` + `import_hhi` |

---

## Scope discipline

Per PI guidance 2026-04-08, every supply-chain feature has an explicit scope:

| Scope | Features |
|---|---|
| **US-only** | `import_dependency`, `us_capacity_ratio`, `import_china_frac`, `import_hhi` |
| **Global-only** | `production_hhi`, `hhi_wgi` (governance), `reserves_china_frac`, `reserves_high_risk_frac`, `domestic_reserve_share`, `global_capacity_ratio` |
| **Mixed (US demand vs global supply, flagged limitation)** | `global_reserve_coverage` |

The single mixed-scope feature is preserved because it carries the irreplaceable "is the global reserve base sufficient for the US transition alone?" question. Methods will state: *"`global_reserve_coverage` represents a worst-case framing in which the US captures supply at-cost; pro-rated allocation would require RoW demand projections that are out of scope."*

---

## Known data quality issues — RESOLVED 2026-04-08

These were discovered while validating the pre-registered clusters against the actual material data and **both have been fixed in code as of 2026-04-08**. Documented here for traceability and reviewer-facing transparency.

### Bug 1 (RESOLVED): REE Net Import Reliance read 0 instead of ~0.94
- **Affected materials:** Dysprosium, Neodymium, Praseodymium, Terbium (all mapped to "Rare Earths" in the aggregate sheet)
- **Root cause:** Priority order in `_build_import_dependency_series` preferred a trade-balance NIR computed from the aggregate sheet over the USGS-published value in the import_dependency sheet. For Rare Earths the aggregate sheet showed US *exports* (~21–47 kt/yr from Mountain Pass concentrate going to China for processing) > *imports* (~7–12 kt/yr of refined REE oxides/metals/magnets), so NIR = (imports − exports)/consumption was negative and got clipped to 0. The USGS-published value (~94%) was already in the same Excel file but never consulted because the aggregate path always returned a value first.
- **Fix applied (2026-04-08):** Priority order reversed — USGS published NIR is now the primary source, with trade-balance as a fallback. Added a stage-asymmetry guard that emits a warning when published ≥ 0.5 but trade-balance ≤ 0.05, so the same class of bug will be caught for any future material with similar characteristics. REE NIR now correctly reads 0.945. See `clustering/feature_engineering.py:_build_import_dependency_series`.

### Bug 2 (RESOLVED): `production_hhi = 0` for ~10 materials (default-to-zero masked data gap)
- **Affected materials:** Cadmium, Fiberglass, Gadolinium ("Gadium" typo), Gallium, Glass, Indium, Yttrium, Selenium, Germanium, Tellurium
- **Root cause:** These materials were missing from the production sheet of the existing risk data; `production_hhi.reindex(...).fillna(0)` silently defaulted them to 0 (perfectly diversified — the *least* risky value, the wrong direction). The USGS MCS 2025 World Data Release CSV was already downloaded in `data/usgs_mcs_2025/world_data/MCS2025_World_Data.csv` (DOI: 10.5066/P13XCP3R) but had never been wired into the clustering feature engineering.
- **Fix applied (2026-04-08):** Added `load_mcs2025_world_data()` helper in `clustering/feature_engineering.py` and patched `_build_production_hhi` and `_build_global_production_series` to fall back to the World CSV for materials missing from the primary source. New HHI values:
  - **Cadmium 0.21** (16 countries; China 39%)
  - **Gallium 0.97** (4 countries; China 98%)
  - **Indium 0.53** (8 countries; China 71%)
  - **Selenium 0.30** (15 countries; China 49%)
  - **Tellurium 0.60** (8 countries; China 76%)
  - **Yttrium and Gadolinium 0.50** (via Rare Earths aggregate proxy; China 69%)
- **Hardcoded (USGS does not publish per-country data):**
  - **Germanium 0.65** — USGS MCS 2025 Germanium chapter explicitly states "global production data were limited"; qualitative description places China >60% (https://pubs.usgs.gov/periodicals/mcs2025/mcs2025-germanium.pdf)
  - **Fiberglass 0.10** — Not tracked as critical commodity; ~100% domestic production from globally diversified silica sand
  - **Glass 0.10** — Same as Fiberglass
- All hardcoded values are stored in `HARDCODED_PRODUCTION_HHI` in `feature_engineering.py` with citations.

**Both bugs were caught by validating the pre-registered clusters against the actual material data on 2026-04-08 — the entire point of pre-registration.** Without this discipline, we would have written a results section claiming "REEs cluster with bulk metals" (with REE NIR = 0) and "trace metals are the least concentrated" (with production_hhi = 0). Pre-registration prevented an inverted central finding.

**Final post-fix material set:** max VIF 5.0, max |r| 0.69, all 31 materials covered, all features lit-grounded or explicitly justified. See `outputs/figures/clustering/_corr_material.png` and `outputs/data/clustering/_vif_material.csv` for receipts.
