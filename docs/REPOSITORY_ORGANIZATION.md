# Repository Organization Summary

**Date**: January 26, 2026
**Original Folder**: `Python/11.30.25`
**New Repository**: `Python/materials_demand_model`

---

## Overview

This document summarizes the reorganization of the materials demand analysis codebase from the working directory (`11.30.25`) into a clean, research-grade repository structure suitable for publication and collaboration.

## What Was Cleaned Up

### Files Removed (Obsolete/Duplicate)

The following files were **not copied** as they were duplicates, demos with " copy" in the name, or superseded by the final versions:

#### Duplicate Demo Files
- `Load/demo_data_ingestion copy.py` - ❌ Duplicate of demo_data_ingestion.py
- `Load/demo_distribution_fitting copy.py` - ❌ Duplicate of demo_distribution_fitting.py

#### Individual Demo Files (Consolidated)
- `Load/demo_data_ingestion.py` - ❌ Functionality integrated into main example
- `Load/demo_distribution_fitting.py` - ❌ Functionality integrated into main example
- `Load/demo_visualizations.py` - ❌ Superseded by demo_viz_2.py
- `Load/demo_viz_2.py` - ❌ Integrated into run_simulation.py
- `Load/test_data_ingestion.py` - ❌ Basic test; replaced by validate_units.py

#### Unused Modules
- `Load/materials_viz_2.py` - ❌ Superseded by materials_visualizations.py
- `Load/example_supply_comparison.py` - ❌ Example for compare_demand_to_supply.py
- `Load/compare_demand_to_supply.py` - ⚠️ Not included (specialized analysis, can add later if needed)

#### Intermediate Output Directories
- `Outputs/` - ❌ Old output location
- `Fit/` - ❌ Old fit results location
- `Monte Carlo Outputs/` - ✅ Contents documented but folder structure not preserved

### Files Kept (Core Functionality)

#### Source Code (in `src/`)
✅ **data_ingestion.py** - Data loading and validation (KEPT - CORE)
✅ **distribution_fitting.py** - Statistical distribution fitting (KEPT - CORE)
✅ **technology_mapping.py** - Technology-material mappings (KEPT - CORE)
✅ **stock_flow_simulation.py** - Monte Carlo simulation engine (KEPT - CORE)
✅ **materials_visualizations.py** - Publication-quality viz (KEPT - CORE)

#### Data Files (in `data/`)
✅ **intensity_data.csv** - Material intensity data
✅ **StdScen24_annual_national.csv** - NREL capacity projections

**Not copied** (but can be added if needed):
- `StdScen24_annual_states.csv` - State-level projections
- `StdScen24_annual_balancingAreas.csv` - Balancing area data
- `StdScen24_transmission_capacities.csv` - Transmission data
- `input_usgs.csv` - USGS supply chain data (for compare_demand_to_supply.py)

#### Examples (in `examples/`)
✅ **run_simulation.py** - Complete workflow (formerly demo_stock_flow_simulation.py)

#### Tests (in `tests/`)
✅ **validate_units.py** - Unit conversion validation

#### Documentation (in `docs/`)
✅ **MONTE_CARLO_ASSESSMENT.md** - Technical quality assessment
✅ **UNIT_FIX_SUMMARY.md** - Unit conversion fix documentation

---

## Current Repository Structure (Updated February 2026)

```
materials_demand_model/
│
├── README.md                     # Comprehensive documentation
├── LICENSE                       # MIT License
├── CHANGELOG.md                  # Version history
├── CONTRIBUTING.md               # Contribution guidelines
├── QUICKSTART.md                 # Quick start guide
├── PIPELINE_DOCUMENTATION.md    # Full pipeline documentation
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation
├── .gitignore                    # Git ignore patterns
│
├── src/                          # Core source code
│   ├── __init__.py              # Package initialization
│   ├── data_ingestion.py        # Data loading and validation
│   ├── distribution_fitting.py  # Statistical distribution fitting
│   ├── technology_mapping.py    # Technology-material mappings
│   ├── stock_flow_simulation.py # Monte Carlo simulation engine
│   └── materials_visualizations.py  # Publication-quality viz
│
├── analysis/                     # Sensitivity analysis module (NEW Feb 2026)
│   ├── __init__.py              # Module exports
│   └── sensitivity_analysis.py  # Variance decomposition, elasticity, Spearman
│
├── clustering/                   # Clustering analysis module
│   ├── __init__.py              # Module exports
│   ├── config.py                # Configuration settings
│   ├── feature_engineering.py   # Scenario/material features
│   ├── preprocessing.py         # Data preprocessing (VIF, scaling)
│   ├── main_analysis.py         # K-means clustering
│   ├── visualization.py         # Clustering visualizations
│   ├── pca_feature_importance.py # PCA analysis
│   ├── sparse_nmf_analysis.py   # Sparse PCA and NMF
│   ├── sparse_pca_story.py      # Sparse PCA interpretation
│   └── supply_chain_analysis.py # Supply chain risk analysis
│
├── visualizations/               # Visualization scripts
│   ├── manuscript_figures.py    # Publication figures
│   ├── manuscript_fig1.py       # Demand curves figure
│   ├── risk_ranking_chart.py    # Supply chain risk charts
│   ├── feature_scatterplots.py  # EDA scatterplots
│   └── compare_figures.py       # Figure validation
│
├── data/                         # Input data
│   ├── intensity_data.csv       # Material intensity data
│   ├── StdScen24_annual_national.csv  # NREL capacity projections
│   └── supply_chain/            # Supply chain risk data
│       └── risk_charts_inputs.xlsx  # USGS/trade data
│
├── examples/                     # Example workflows
│   ├── run_simulation.py        # Main simulation example
│   ├── supply_chain_risk_analysis.py  # Risk analysis example
│   └── diagnostics/             # Debugging utilities
│
├── tests/                        # Testing and validation
│   └── validate_units.py        # Unit conversion validation
│
├── outputs/                      # Generated results
│   ├── data/                    # CSV outputs
│   │   ├── clustering/          # Cluster results
│   │   └── sensitivity/         # Sensitivity analysis results
│   ├── figures/                 # Visualization outputs
│   │   ├── clustering/          # Cluster visualizations
│   │   ├── exploratory/         # EDA figures
│   │   ├── manuscript/          # Publication figures
│   │   ├── sensitivity/         # Sensitivity figures
│   │   └── supply_chain/        # Risk visualizations
│   └── reports/                 # Text reports
│
└── docs/                         # Documentation
    ├── variable_reference.csv   # All variables documented
    ├── visualization_inventory.csv  # All figures documented
    ├── SI_methodology.md        # Supplementary methods
    └── [technical docs]         # Various technical docs
```

---

## Key Improvements

### 1. **Professional Structure**
- Follows Python package conventions (src/, tests/, examples/, docs/)
- Clear separation of source code, data, and outputs
- Standard auxiliary files (LICENSE, CONTRIBUTING, etc.)

### 2. **Proper Python Package**
- `__init__.py` with explicit exports
- `setup.py` for pip installation
- `requirements.txt` for dependencies
- Can be installed with `pip install -e .`

### 3. **Comprehensive Documentation**
- **README.md**: 350+ lines covering installation, usage, methodology, limitations
- **CONTRIBUTING.md**: Guidelines for contributors
- **CHANGELOG.md**: Version history
- **Existing technical docs**: Monte Carlo assessment and unit fix summary

### 4. **Version Control Ready**
- `.gitignore` configured for Python projects
- `.gitkeep` to preserve directory structure
- Clear documentation of what's tracked vs. generated

### 5. **Research-Grade Standards**
- Follows reproducibility best practices
- Clear citation format
- Explicit licensing (MIT)
- Comprehensive methodology documentation

---

## File Mapping Reference

| Original (11.30.25) | New Location | Status | Notes |
|---------------------|--------------|--------|-------|
| `Load/data_ingestion.py` | `src/data_ingestion.py` | ✅ Kept | Core module |
| `Load/distribution_fitting.py` | `src/distribution_fitting.py` | ✅ Kept | Core module |
| `Load/technology_mapping.py` | `src/technology_mapping.py` | ✅ Kept | Core module |
| `Load/stock_flow_simulation.py` | `src/stock_flow_simulation.py` | ✅ Kept | Core module |
| `Load/materials_visualizations.py` | `src/materials_visualizations.py` | ✅ Kept | Core module |
| `Load/demo_stock_flow_simulation.py` | `examples/run_simulation.py` | ✅ Modified | Updated paths |
| `Load/validate_units.py` | `tests/validate_units.py` | ✅ Kept | Validation test |
| `src/intensity_data.csv` | `data/intensity_data.csv` | ✅ Kept | Essential data |
| `src/StdScen24_annual_national.csv` | `data/StdScen24_annual_national.csv` | ✅ Kept | Essential data |
| `MONTE_CARLO_ASSESSMENT.md` | `docs/MONTE_CARLO_ASSESSMENT.md` | ✅ Kept | Technical docs |
| `UNIT_FIX_SUMMARY.md` | `docs/UNIT_FIX_SUMMARY.md` | ✅ Kept | Technical docs |
| `Load/materials_viz_2.py` | — | ❌ Removed | Superseded |
| `Load/demo_data_ingestion.py` | — | ❌ Removed | Consolidated |
| `Load/demo_distribution_fitting.py` | — | ❌ Removed | Consolidated |
| `Load/demo_visualizations.py` | — | ❌ Removed | Consolidated |
| `Load/*copy.py` | — | ❌ Removed | Duplicates |
| `Outputs/`, `Fit/`, `Monte Carlo Outputs/` | — | ❌ Removed | Old outputs |

---

## How to Use the New Repository

### Installation

```bash
cd materials_demand_model
pip install -r requirements.txt
```

Or install as package:
```bash
pip install -e .
```

### Running Simulation

```bash
cd examples
python run_simulation.py
```

### Importing as Package

```python
from src import run_full_simulation

simulation, results = run_full_simulation(
    intensity_path='data/intensity_data.csv',
    capacity_path='data/StdScen24_annual_national.csv',
    n_iterations=10000,
    random_state=42
)
```

---

## Migration Notes

### For Existing Users

If you were using code from `11.30.25`, here's how to migrate:

**Old**:
```python
from stock_flow_simulation import run_full_simulation

results = run_full_simulation(
    intensity_path='Python/11.30.25/src/intensity_data.csv',
    capacity_path='Python/11.30.25/src/StdScen24_annual_national.csv',
    # ...
)
```

**New**:
```python
from src.stock_flow_simulation import run_full_simulation

results = run_full_simulation(
    intensity_path='data/intensity_data.csv',
    capacity_path='data/StdScen24_annual_national.csv',
    # ...
)
```

### Path Updates

- All paths in `examples/run_simulation.py` have been updated to use the new structure
- Data paths now relative to repository root
- Outputs go to `outputs/` directory

---

## What's Next?

### Recommended Additions

1. **Add Jupyter Notebooks**: Create `notebooks/` with example workflows
2. **Expand Tests**: Add pytest-based test suite in `tests/`
3. **CI/CD**: Add GitHub Actions for automated testing
4. **Additional Data**: Add state-level and transmission data if needed
5. **Supply Chain Analysis**: Optionally include `compare_demand_to_supply.py`

### Publishing to GitHub

```bash
cd materials_demand_model
git init
git add .
git commit -m "Initial commit: Research-grade materials demand model v1.0.0"
git remote add origin https://github.com/yourusername/materials_demand_model.git
git push -u origin main
```

### Publishing to PyPI (Optional)

After further testing:
```bash
python setup.py sdist bdist_wheel
twine upload dist/*
```

---

## Verification Checklist

✅ All core modules copied
✅ Essential data files included
✅ Working example with updated paths
✅ Unit validation test included
✅ Technical documentation preserved
✅ README.md comprehensive
✅ LICENSE file added
✅ Contributing guidelines added
✅ Package can be imported
✅ .gitignore configured
✅ requirements.txt complete

---

## Contact

For questions about the repository organization:
- Review this document and README.md
- Check CONTRIBUTING.md for contribution guidelines
- Open an issue if something is unclear

**Repository organized by**: Claude Code (Materials Demand Research Team)
**Date**: January 26, 2026
**Version**: 1.0.0
