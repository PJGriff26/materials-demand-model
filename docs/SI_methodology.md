# Supporting Information: Methodology

## Overview

This document provides detailed methodology for the materials demand Monte Carlo simulation pipeline used to project material requirements for U.S. electricity infrastructure expansion (2026-2050).

---

## 1. Scenario Data Source

### NREL Standard Scenarios 2024

The pipeline uses **61 scenarios** from the National Renewable Energy Laboratory (NREL) Standard Scenarios 2024 dataset, which provides capacity projections for electricity generation technologies across the United States.

**Key characteristics:**
- **Time horizon**: 2026-2050 (in 3-year intervals)
- **Technologies**: Solar PV (utility, distributed), wind (onshore, offshore), nuclear, hydro, natural gas, coal, battery storage, geothermal, biomass
- **Geographic scope**: National (U.S. total)
- **Units**: Installed capacity in megawatts (MW)

**Note on methodology change**: The original manuscript used 11 independent capacity expansion models from intermodel comparison studies. This updated pipeline uses the NREL Standard Scenarios which provide a broader range of policy and technology futures through a single consistent modeling framework. This change allows for more systematic uncertainty quantification across a larger scenario space.

### Scenario Types

The 61 scenarios span various assumptions about:
- Policy environments (with/without Inflation Reduction Act provisions)
- Technology costs (reference, advanced renewable energy, high natural gas)
- Demand growth trajectories
- Grid flexibility and storage deployment

Rather than classifying scenarios into discrete "Reference" vs "Policy" categories, the pipeline treats all scenarios as equally plausible futures and reports statistics across the full scenario distribution.

---

## 2. Monte Carlo Simulation Approach

### Distribution Fitting

For each material-technology combination, the pipeline fits probability distributions to literature-reported material intensity values.

**Process:**
1. Collect material intensity data points from published literature and environmental product declarations
2. Test candidate distributions: truncated normal, lognormal, gamma, uniform
3. Select best-fit distribution using Kolmogorov-Smirnov goodness-of-fit test
4. Store fitted distribution parameters for Monte Carlo sampling

**Units:**
- Raw intensity data: tonnes per gigawatt (t/GW)
- Converted for calculations: tonnes per megawatt (t/MW)

### Monte Carlo Sampling

**Configuration:**
- **Iterations**: N = 10,000
- **Random seed**: 42 (for reproducibility)
- **Sampling method**: Independent sampling from fitted distributions per iteration

**For each iteration:**
1. Sample material intensity value from fitted distribution
2. Calculate material demand: `Demand = Capacity_Additions × Intensity`
3. Store result

**Output statistics:**
- Mean, standard deviation
- Percentiles: 2.5th (p2), 5th (p5), 25th (p25), 50th/median (p50), 75th (p75), 95th (p95), 97.5th (p97)

### Stock-Flow Accounting

The model uses stock-flow accounting to distinguish between total installed capacity and annual capacity additions.

**Capacity additions calculation:**
```
Additions(t) = Capacity(t) - Capacity(t-1) + Retirements(t)
Retirements(t) = Additions(t - lifetime)
```

**Technology lifetimes:**
| Technology | Lifetime (years) |
|------------|------------------|
| Solar PV | 30 |
| Wind (onshore/offshore) | 25 |
| Nuclear | 60 |
| Hydro | 80 |
| Natural Gas | 40 |
| Coal | 50 |
| Battery Storage | 15 |

---

## 3. Supply Chain Risk Assessment

### Country Risk Classification (CRC)

Material sourcing is assessed using OECD Country Risk Classifications (February 2024), which rate countries on a scale of 0-7 based on political and economic stability.

**CRC categories used:**
- **United States**: Domestic production (CRC weight = 0)
- **OECD**: Other OECD member countries (CRC weight = 0)
- **CRC 1-7**: Non-OECD countries by risk level (CRC weight = 1-7)
- **China**: Special category due to market dominance (CRC weight = 7)
- **Undefined**: Countries without CRC data (CRC weight = 4)

### Import Dependency Calculation

Import dependency represents net import reliance:

```
Import_Dependency = (Imports - Exports) / Consumption
```

For materials where the U.S. is a net exporter, import dependency = 0 (not negative).

### CRC-Weighted Risk Score

For each material, the CRC-weighted risk score is calculated as:

```
CRC_Risk = Σ(Import_Share_i × CRC_Weight_i) / Σ(Import_Share_i)
```

Where `i` indexes import source countries.

### Reserve Adequacy

Reserve coverage ratio compares cumulative projected demand to available reserves:

```
Reserve_Coverage = Reserves / Cumulative_Demand_2026-2050
```

Values > 1 indicate reserves exceed projected demand; values < 1 indicate potential supply constraints.

---

## 4. Data Sources

### Material Intensity Data
- NREL Material Intensity Database
- Published literature on energy technology material requirements
- Environmental Product Declarations (EPDs)

### Capacity Projections
- NREL Standard Scenarios 2024
- URL: https://www.nrel.gov/analysis/standard-scenarios.html

### Supply Chain Data
- USGS Mineral Commodity Summaries 2023
- OECD Country Risk Classifications (February 2024)
- Historical trade data (2018-2022 averages)

### Reserve Data
- USGS Mineral Commodity Summaries 2023
- Global and country-level economic reserve estimates

---

## 5. Key Assumptions and Limitations

### Assumptions

1. **Independence**: Material intensities are assumed independent across technology-material pairs
2. **Static intensities**: Material intensities remain constant over time (no learning curves)
3. **Historical trade patterns**: Import shares based on recent historical data (2018-2022)
4. **No recycling**: Model does not account for material recovery from retired capacity

### Limitations

1. **Technology coverage**: Some emerging technologies (advanced batteries, hydrogen electrolyzers, direct air capture) lack comprehensive material intensity data
2. **Regional variation**: Analysis uses national totals; regional supply chain dynamics not captured
3. **Market dynamics**: Fixed import shares don't account for supply chain shifts in response to demand changes
4. **Retirement uncertainty**: Actual retirement timing depends on economic factors not modeled

---

## 6. Output Files

### Main Outputs

| File | Description |
|------|-------------|
| `material_demand_by_scenario.csv` | Detailed results by scenario, year, material |
| `material_demand_summary.csv` | Aggregated statistics across scenarios |

### Visualization Outputs

| Directory | Contents |
|-----------|----------|
| `outputs/figures/manuscript/` | Publication-quality figures |
| `outputs/figures/clustering/` | Clustering analysis visualizations |
| `outputs/figures/supply_chain/` | Risk assessment figures |

---

## 7. Reproducibility

To reproduce the analysis:

```bash
# Install dependencies
pip install -r requirements.txt

# Run Monte Carlo simulation
python examples/run_simulation.py

# Generate manuscript figures
python visualizations/manuscript_figures.py
python visualizations/manuscript_fig1.py

# Run clustering analysis
cd clustering && python main_analysis.py

# Generate supply chain figures
python clustering/supply_chain_analysis.py
```

**Random seed**: All stochastic processes use `random_state=42` for reproducibility.

---

## References

1. NREL Standard Scenarios 2024. National Renewable Energy Laboratory. https://www.nrel.gov/analysis/standard-scenarios.html

2. USGS Mineral Commodity Summaries 2023. U.S. Geological Survey. https://www.usgs.gov/centers/national-minerals-information-center/mineral-commodity-summaries

3. OECD Country Risk Classifications. Organisation for Economic Co-operation and Development. https://www.oecd.org/trade/topics/export-credits/arrangement-and-sector-understandings/financing-terms-and-conditions/country-risk-classification/

4. ISO/JCGM 101:2008. Evaluation of measurement data — Supplement 1 to the "Guide to the expression of uncertainty in measurement" — Propagation of distributions using a Monte Carlo method.
