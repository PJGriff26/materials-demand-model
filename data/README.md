# Data Provenance

This directory contains input data for a Monte Carlo simulation model of critical material demand arising from U.S. energy technology deployment. The model combines technology deployment scenarios with material intensity distributions to project future material requirements under uncertainty.

All datasets used in this project are derived from publicly available sources.

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

## 3. Supply Chain Risk Data

### 3a. Risk Charts Inputs

| Attribute     | Detail |
|---------------|--------|
| **File**      | `supply_chain/risk_charts_inputs.xlsx` (also mirrored at `risk_charts_inputs.xlsx`) |
| **Format**    | Excel workbook (multiple sheets) |
| **Rows**      | Varies by sheet |
| **Key Columns** | Import dependency by country/region, production shares by Country Risk Category (CRC), global and U.S. reserves, U.S. production and consumption |
| **Source**    | USGS Mineral Commodity Summaries; USGS trade data |
| **License/Terms** | Publicly available; U.S. Government work product |

### 3b. USGS Mineral Commodity Summary CSVs

| Attribute     | Detail |
|---------------|--------|
| **Files**     | `supply_chain/mcs2023-cadmi_salient.csv`, `mcs2023-galli_salient.csv`, `mcs2023-germa_salient.csv`, `mcs2023-indiu_salient.csv`, `mcs2023-selen_salient.csv`, `mcs2023tellu_salient.csv` |
| **Format**    | CSV |
| **Rows**      | 5 per file (years 2018-2022) |
| **Key Columns** | `DataSource`, `Commodity`, `Year`, U.S. production, imports, exports, consumption, prices, net import reliance (NIR) |
| **Source**    | USGS Mineral Commodity Summaries 2023 |
| **License/Terms** | Publicly available; U.S. Government work product |

**Materials covered:** Cadmium, Gallium, Germanium, Indium, Selenium, Tellurium.

**Citation:**

> U.S. Geological Survey. 2023. "Mineral Commodity Summaries 2023." Reston, VA: U.S. Geological Survey. https://doi.org/10.3133/mcs2023

---

## 4. USGS Input Data (Aggregated)

| Attribute     | Detail |
|---------------|--------|
| **File**      | `input_usgs.csv` |
| **Format**    | CSV |
| **Rows**      | ~113 |
| **Key Columns** | `Commodity`, `Year`, `Variable`, `Unit`, `Value` |
| **Source**    | USGS Mineral Commodity Summaries |
| **License/Terms** | Publicly available; U.S. Government work product |

**Description:** Aggregated U.S. production, consumption, and trade data for materials including Aluminum, and others. Variables include primary and secondary production, consumption, and related metrics. Data spans 2018-2022.

**Citation:**

> U.S. Geological Survey. 2023. "Mineral Commodity Summaries 2023." Reston, VA: U.S. Geological Survey. https://doi.org/10.3133/mcs2023

---

## Data Access

All source data used in this project are publicly available:

- **NREL Standard Scenarios 2024**: Downloadable from https://www.nrel.gov/analysis/standard-scenarios.html
- **USGS Mineral Commodity Summaries**: Available at https://www.usgs.gov/centers/national-minerals-information-center/mineral-commodity-summaries
- **Material intensity values**: Compiled from peer-reviewed publications; see project references for the full source list

No proprietary or access-restricted data are used in this analysis.
