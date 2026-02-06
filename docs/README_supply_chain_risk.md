# Supply Chain Risk Analysis Script

## Overview

The `supply_chain_risk_analysis.py` script analyzes supply chain risk by comparing Monte Carlo materials demand projections against USGS production and trade data. It allocates projected demand across Country Risk Categories (CRC) based on US import dependency and source country risk classifications.

## Quick Start

### Basic Usage

```bash
python supply_chain_risk_analysis.py \
  --demand_csv outputs/material_demand_by_scenario.csv \
  --risk_xlsx data/risk_charts_inputs.xlsx
```

This will:
- Analyze all scenarios and materials
- Apply unit conversion (tonnes → thousand metric tonnes)
- Aggregate rare earth elements
- Generate outputs in `./outputs/risk_analysis/`

### Example Use Cases

**1. Focus on specific scenarios with log scale**
```bash
python supply_chain_risk_analysis.py \
  --demand_csv outputs/material_demand_by_scenario.csv \
  --risk_xlsx data/risk_charts_inputs.xlsx \
  --scenarios Mid_Case,Mid_Case_100by2035 \
  --yaxis log \
  --outdir outputs/midcase_analysis
```

**2. Analyze specific materials**
```bash
python supply_chain_risk_analysis.py \
  --demand_csv outputs/material_demand_by_scenario.csv \
  --risk_xlsx data/risk_charts_inputs.xlsx \
  --materials Copper,Nickel,Aluminum \
  --outdir outputs/critical_materials
```

**3. Keep rare earths separate**
```bash
python supply_chain_risk_analysis.py \
  --demand_csv outputs/material_demand_by_scenario.csv \
  --risk_xlsx data/risk_charts_inputs.xlsx \
  --no_aggregate_rare_earths
```

**4. Custom comparison year and unit scale**
```bash
python supply_chain_risk_analysis.py \
  --demand_csv outputs/material_demand_by_scenario.csv \
  --risk_xlsx data/risk_charts_inputs.xlsx \
  --target_year 2040 \
  --unit_scale 1000000  # Convert to million metric tonnes
```

## Command-Line Arguments

### Required Arguments

- `--demand_csv PATH`
  Path to material_demand_by_scenario.csv (Monte Carlo output)

- `--risk_xlsx PATH`
  Path to risk_charts_inputs.xlsx (USGS data)

### Optional Arguments

- `--outdir PATH`
  Output directory (default: `./outputs/risk_analysis`)

- `--unit_scale FLOAT`
  Conversion factor for demand units (default: 1000 = tonnes → kMT)

- `--scenarios LIST`
  Comma-separated scenario names to analyze (default: all)

- `--materials LIST`
  Comma-separated material names to analyze (default: all)

- `--yaxis {linear,log}`
  Y-axis scale for plots (default: linear)

- `--aggregate_rare_earths`
  Aggregate rare earths into single category (default: True)

- `--no_aggregate_rare_earths`
  Keep rare earths separate

- `--target_year YEAR`
  Year for demand vs production comparison table (default: 2035)

- `--tolerance FLOAT`
  CRC share validation tolerance in percentage points (default: 0.1)

## Outputs

All outputs are saved to the specified `--outdir`:

### 1. allocated_demand_by_crc.csv
Tidy-format CSV with demand allocated by CRC category.

**Columns:**
- `scenario`: Scenario name
- `year`: Projection year
- `material`: Material name
- `crc`: Country Risk Category
- `demand_allocated`: Material demand allocated to this CRC (thousand mt)
- `demand_mean_total`: Total demand for this material (thousand mt)
- `share_pct`: Percentage of demand allocated to this CRC

### 2. crc_shares_audit.csv
Validation table showing CRC shares sum to 100% per material.

**Columns:**
- `material`: Material name
- `total_share`: Sum of CRC shares (should be 100%)
- `deviation`: Deviation from 100%
- `valid`: Whether material passed validation

### 3. demand_vs_production_comparison.csv
Comparison of projected demand vs historical USGS production/consumption.

**Columns:**
- `scenario`: Scenario name
- `material`: Material name
- `demand_{year}`: Projected demand for target year (thousand mt)
- `avg_production`: Average US production 2018-2022 (thousand mt)
- `avg_consumption`: Average US consumption 2018-2022 (thousand mt)
- `avg_net_import`: Average US net imports 2018-2022 (thousand mt)
- `demand_to_production_ratio`: Ratio of projected demand to historical production
- `demand_to_consumption_ratio`: Ratio of projected demand to historical consumption

### 4. risk_analysis_stacked_bars_{linear|log}.png
Multi-panel visualization with one subplot per material.

**Features:**
- Stacked bars by CRC category (color-coded by risk level)
- One bar per scenario at each year
- Reference lines for average production (solid), consumption (dashed), and net imports (dotted)
- Shared legend showing all CRC categories
- Publication-quality at 1000 dpi

### 5. risk_analysis_summary.txt
Text summary report with:
- Input file paths and configuration
- Analysis scope (scenarios, materials analyzed)
- CRC validation results
- Top 10 materials by demand
- Materials with demand > 2x historical production
- Materials with highest import reliance

## CRC (Country Risk Categories)

The script allocates demand across the following risk categories:

- **United States**: Domestic production
- **OECD**: OECD countries (low risk)
- **1-7**: Numerical risk ratings from OECD export credit framework
  - 1: Lower risk
  - 7: Higher risk
- **China**: Tracked separately given strategic importance
- **Undefined**: Countries without assigned risk rating

## Methodology

The CRC allocation follows this logic (from `risk_charts.py`):

1. **For each material:**
   - Calculate average US import reliance (2018-2022 USGS data)
   - Get import source countries and their shares of total imports
   - Map source countries to CRC categories

2. **Allocate shares:**
   - **Domestic share** = 100% - import reliance%
   - **Imported shares** = import_shares × (import_reliance% / 100)

3. **Apply to demand projections:**
   - **Allocated demand** = total demand × (CRC share / 100)

4. **Validation:**
   - Verify CRC shares sum to 100% per material (±0.1% tolerance)

## Materials Coverage

**With USGS import data (18 materials):**
Aluminum, Boron, Cement, Chromium, Copper, Lead, Magnesium, Manganese, Molybdenum, Nickel, Niobium, Rare Earths (aggregated), Silicon, Silver, Steel, Tin, Vanadium, Yttrium, Zinc

**Without USGS import data (9 materials - skipped with warnings):**
Cadmium, Fiberglass, Gadium, Gallium, Germanium, Glass, Indium, Selenium, Tellurium

## Troubleshooting

### Error: "Demand CSV not found"
- Verify the path to `material_demand_by_scenario.csv`
- Use absolute paths if relative paths fail
- Example: `--demand_csv /full/path/to/outputs/material_demand_by_scenario.csv`

### Error: "Missing required sheet"
- Ensure `risk_charts_inputs.xlsx` contains all required sheets:
  - import_dependency
  - import_shares
  - crc
  - aggregate

### Error: "Scenarios not found in data"
- Check scenario names match exactly (case-sensitive)
- List available scenarios: `grep "^[^,]*," outputs/material_demand_by_scenario.csv | cut -d',' -f1 | sort -u`

### Warning: "Skipping X materials without USGS import data"
- This is expected for materials not in USGS trade database
- These materials are excluded from CRC allocation but remain in original demand outputs

### CRC share validation failed
- This indicates data inconsistency in USGS import shares
- Check import shares sum to 100% per material in `risk_charts_inputs.xlsx`
- Adjust `--tolerance` if needed (e.g., `--tolerance 0.5`)

## Dependencies

```bash
pip install pandas>=1.5.0 numpy>=1.23.0 matplotlib>=3.6.0 openpyxl>=3.0.0
```

## References

- **USGS Mineral Commodity Summaries**: https://www.usgs.gov/centers/national-minerals-information-center/mineral-commodity-summaries
- **OECD Country Risk Classifications**: https://www.oecd.org/trade/topics/export-credits/arrangement-and-sector-understandings/financing-terms-and-conditions/country-risk-classification/

## Citation

If you use this script in your research, please cite:

```
[Your Research Team]. (2026). Materials Demand Monte Carlo Pipeline with Supply Chain Risk Analysis.
[Repository/Publication Details]
```

## Support

For issues, questions, or contributions, please contact [Your Contact Information].

---

**Last Updated:** January 2026
**Version:** 1.0.0
