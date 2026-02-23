# Materials Demand Model

**A Research-Grade Monte Carlo Simulation Framework for Energy Infrastructure Materials Demand**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides a comprehensive Monte Carlo simulation framework for estimating material demand uncertainty in energy infrastructure deployment. The model uses stock-flow accounting combined with probabilistic material intensity data to project demand for critical materials (copper, steel, aluminum, lithium, etc.) under various energy transition scenarios.

### Key Features

- **Rigorous Uncertainty Quantification**: Monte Carlo sampling (N=10,000) with full percentile reporting
- **Robust Distribution Fitting**: Tests multiple parametric distributions with goodness-of-fit validation
- **Stock-Flow Accounting**: Models capacity additions, retirements, and lifetime effects
- **Flexible Technology Mapping**: Easily configurable mapping from capacity to material intensities
- **Publication-Quality Visualizations**: Automated generation of figures for research papers
- **Research-Grade Implementation**: Follows ISO JCGM 101:2008 and NIST uncertainty standards

## Repository Structure

```
materials_demand_model/
├── run_pipeline.py                   # Full reproducibility pipeline (Steps 1–10)
├── requirements.txt                  # Pinned Python dependencies
├── setup.py                          # Package installation
├── CHANGELOG.md                      # Version history
├── CONTRIBUTING.md                   # Contribution guidelines
├── PIPELINE_DOCUMENTATION.md         # Pipeline step-by-step documentation
├── QUICKSTART.md                     # Quick start guide
│
├── src/                              # Core simulation engine
│   ├── __init__.py
│   ├── data_ingestion.py             # Data loading and validation
│   ├── data_quality.py               # Data quality checks and reporting
│   ├── distribution_fitting.py       # Statistical distribution fitting
│   ├── technology_mapping.py         # Technology-material mappings
│   ├── stock_flow_simulation.py      # Monte Carlo simulation engine
│   └── materials_visualizations.py   # Core visualization tools
│
├── analysis/                         # Sensitivity analysis module
│   ├── __init__.py
│   └── sensitivity_analysis.py       # Variance decomposition, elasticity, Spearman
│
├── clustering/                       # Clustering & dimensionality reduction
│   ├── config.py                     # Paths and configuration
│   ├── feature_engineering.py        # Scenario/material feature calculation
│   ├── preprocessing.py              # VIF filtering, StandardScaler
│   ├── clustering.py                 # ClusterAnalyzer class (K-means)
│   ├── main_analysis.py              # Production pipeline (SPCA → K-means)
│   ├── visualization.py              # Per-cluster visualizations
│   ├── pca_feature_importance.py     # Standard PCA analysis
│   ├── sparse_nmf_analysis.py        # Sparse PCA, NMF, method comparison
│   ├── sparse_pca_story.py           # Sparse PCA interpretation & narratives
│   ├── factor_analysis.py            # Factor Analysis module
│   ├── clustering_comparison.py      # 4-method comparison (VIF/PCA/SPCA/FA)
│   └── supply_chain_analysis.py      # CRC sourcing, reserve adequacy
│
├── visualizations/                   # Manuscript & analysis figures
│   ├── manuscript_figures.py         # Fig. 2, SI capacity/intensity figures
│   ├── manuscript_fig1.py            # Fig. 1 demand curves
│   ├── risk_ranking_chart.py         # Supply chain risk charts
│   ├── feature_scatterplots.py       # EDA scatterplots & correlations
│   ├── compare_figures.py            # Before/after figure comparison
│   └── examples/                     # Example workflows & diagnostics
│       ├── run_simulation.py         # Complete simulation example
│       ├── sensitivity_analysis.py   # Sensitivity analysis example
│       ├── supply_chain_risk_analysis.py  # Risk analysis example
│       └── diagnostics/              # Debugging & inspection scripts
│
├── paper/                            # Manuscript generation scripts
│   ├── generate_manuscript.py        # Main manuscript generator
│   ├── generate_si.py               # Supplementary information generator
│   ├── extract_results.py            # Results extraction for manuscript
│   └── organize_figures.py           # Figure organization for submission
│
├── proposal_figures/                 # Original proposal visualizations
│   ├── figure1_scenario_spaghetti.*  # Scenario ensemble plots
│   ├── figure2_material_correlations.*
│   ├── figure3_material_boxplots.*
│   ├── figure4_risk_scatter.*
│   ├── figure_capacity_additions.*
│   ├── figure_technology_blocks.*
│   └── figure_technology_correlation_blocks.*
│
├── data/                             # Input data (see data/README.md)
│   ├── intensity_data.csv            # Material intensity (t/GW)
│   ├── StdScen24_annual_national.csv # NREL Standard Scenarios 2024
│   ├── input_usgs.csv                # USGS mineral commodity data
│   └── supply_chain/
│       └── risk_charts_inputs.xlsx   # USGS/trade risk data
│
├── tests/                            # Testing and validation
│   ├── conftest.py                   # Pytest configuration
│   ├── test_pipeline.py              # Regression tests
│   └── validate_units.py             # Unit conversion validation
│
├── outputs/                          # Generated results
│   ├── material_demand_by_scenario.csv  # Full MC results by scenario
│   ├── material_demand_summary.csv      # Aggregated summary statistics
│   ├── simulation_report.txt            # Comprehensive text report
│   ├── data/
│   │   ├── clustering/               # Cluster results, features, scores
│   │   │   └── comparison/           # 4-method comparison metrics
│   │   ├── sensitivity/              # Variance decomposition, Spearman
│   │   └── supply_chain/             # CRC allocation, risk scores
│   ├── figures/
│   │   ├── manuscript/               # Publication figures (Fig. 1–4, SI)
│   │   ├── clustering/               # Organized by analysis type
│   │   │   ├── kmeans/               # Elbow, silhouette, biplot, profiles
│   │   │   ├── pca_analysis/         # Scree, loadings, feature importance
│   │   │   ├── dimensionality_reduction/  # SPCA, NMF, method comparison
│   │   │   ├── spca_story/           # Named components, quadrant plots
│   │   │   ├── factor_analysis/      # FA loadings & communalities
│   │   │   └── comparison/           # 4-method clustering comparison
│   │   ├── sensitivity/              # Spearman, variance decomposition
│   │   ├── exploratory/              # EDA scatterplots & correlations
│   │   ├── supply_chain_risk/        # Risk ranking, heatmaps
│   │   └── archive/                  # Historical figures
│   │       ├── figure_validation/    # Before/after comparison PNGs
│   │       └── prototype/            # Early prototype visualizations
│   └── reports/                      # Text reports
│
└── docs/                             # Documentation
    ├── variable_reference.csv        # Master variable documentation
    ├── visualization_inventory.csv   # All visualizations documented
    ├── SI_methodology.md             # Supplementary methods
    ├── README_supply_chain_risk.md   # Risk methodology
    ├── README_VISUALIZATION_TOOLS.md # Visualization tooling guide
    ├── REPOSITORY_ORGANIZATION.md    # Repo organization reference
    ├── manuscript_figure_map.csv     # Figure-to-code mapping
    └── archive/                      # Superseded diagnostic documents
```

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup

```bash
git clone https://github.com/PJGriff26/materials-demand-model.git
cd materials-demand-model
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "from src import MaterialsStockFlowSimulation; print('OK')"
```

## Quick Start

### Reproduce All Results

```bash
python run_pipeline.py                    # Full pipeline (Steps 1–10, ~20 min)
python run_pipeline.py --skip-simulation  # Skip MC if outputs exist (~5 min)
python run_pipeline.py --step 5           # Run only one step
python run_pipeline.py --from 3           # Resume from Step 3
```

### Run Simulation Only

```bash
python visualizations/examples/run_simulation.py
```

This will:
1. Load material intensity and capacity projection data
2. Fit probability distributions to material intensities
3. Run Monte Carlo simulation (10,000 iterations)
4. Generate summary statistics and detailed reports
5. Save results to `outputs/` directory

### Basic Python Usage

```python
from src.stock_flow_simulation import run_full_simulation

# Run simulation
simulation, results = run_full_simulation(
    intensity_path='data/intensity_data.csv',
    capacity_path='data/StdScen24_annual_national.csv',
    n_iterations=10000,
    random_state=42  # For reproducibility
)

# Get statistics
stats = results.get_statistics(percentiles=[2.5, 5, 25, 50, 75, 95, 97.5])

# Save results
stats.to_csv('outputs/material_demand.csv', index=False)

# Print summary
copper_2035 = stats[(stats['material'] == 'Copper') & (stats['year'] == 2035)]
print(f"Copper demand in 2035: {copper_2035['p50'].values[0]/1e6:.2f} million tonnes")
print(f"95% CI: [{copper_2035['p2'].values[0]/1e6:.2f}, {copper_2035['p97'].values[0]/1e6:.2f}]")
```

### Creating Visualizations

```python
from src.materials_visualizations import MaterialsDemandVisualizer

# Load results
results_df = pd.read_csv('outputs/material_demand_by_scenario.csv')

# Create visualizer
viz = MaterialsDemandVisualizer(results_df, output_dir='outputs/figures')

# Generate complete figure suite
viz.generate_figure_suite(
    key_materials=['Copper', 'Aluminum', 'Steel', 'Lithium', 'Silicon'],
    key_year=2035
)
```

## Methodology

### Model Overview

The model implements a **stock-flow accounting framework** with **Monte Carlo uncertainty propagation**:

1. **Input Data**:
   - Material intensity distributions (tonnes of material per MW of capacity)
   - Capacity deployment projections (MW by technology, year, scenario)

2. **Stock-Flow Model**:
   ```
   Stock(t) = Stock(t-1) + Additions(t) - Retirements(t)
   Retirements(t) = Additions(t - lifetime)
   Material Demand(t) = Additions(t) × Material Intensity
   ```

3. **Monte Carlo Sampling**:
   - For each iteration (1 to 10,000):
     - Sample material intensity from fitted distribution
     - Calculate material demand using sampled intensity
     - Store results

4. **Output**:
   - Full probability distributions for material demand
   - Percentiles: 2.5, 5, 25, 50 (median), 75, 95, 97.5
   - Summary statistics by scenario, year, and material

### Uncertainty Quantification Standards

This implementation follows:
- **ISO/JCGM 101:2008** (GUM Supplement 1): Monte Carlo methods for uncertainty propagation
- **NIST Technical Note 1297**: Guidelines for uncertainty evaluation
- **Best practices** from peer-reviewed infrastructure modeling literature

See [`docs/archive/MONTE_CARLO_ASSESSMENT.md`](docs/archive/MONTE_CARLO_ASSESSMENT.md) for detailed technical assessment.

## Data Sources

### Material Intensity Data
- **Source**: NREL Material Intensity Database
- **Format**: CSV with columns `technology`, `Material`, `value`
- **Units**: **tonnes per gigawatt (t/GW)** → automatically converted to t/MW
- **Coverage**: 21 technologies × 31 materials = 169 combinations

### Capacity Projections
- **Source**: NREL Standard Scenarios 2024
- **Scenarios**: 61 scenarios covering range of policy and technology futures
- **Time horizon**: 2026-2050 (3-year intervals)
- **Technologies**: Solar (utility, distributed), wind (onshore, offshore), nuclear, hydro, geothermal, fossil fuels, storage

## Outputs

### Main Output Files

1. **`material_demand_by_scenario.csv`**
   - Detailed results for each scenario, year, and material
   - Columns: `scenario`, `year`, `material`, `mean`, `std`, `p2`, `p5`, `p25`, `p50`, `p75`, `p95`, `p97`
   - ~47,000 rows (61 scenarios × 9 years × 31 materials × statistics)

2. **`material_demand_summary.csv`**
   - Aggregated results across all scenarios
   - Useful for total material requirements
   - ~280 rows (9 years × 31 materials)

3. **`simulation_report.txt`**
   - Comprehensive text report with:
     - Simulation parameters
     - Model description
     - Key results (top materials, temporal trends, scenario comparisons)
     - Data quality notes and limitations

### Expected Output Magnitudes

After unit correction, typical annual demand (2035, aggressive buildout):
- **Copper**: 1-10 million tonnes
- **Steel**: 10-100 million tonnes
- **Aluminum**: 10-100 million tonnes
- **Cement**: 100-500 million tonnes

**Note**: Previous versions had a 1000× error (outputs in trillions). This has been fixed as of January 2026. See [`docs/archive/UNIT_FIX_SUMMARY.md`](docs/archive/UNIT_FIX_SUMMARY.md).

## Customization

### Modifying Technology Mapping

Edit `src/technology_mapping.py` to customize how capacity technologies map to material intensities:

```python
TECHNOLOGY_MAPPING = {
    'upv': {
        'utility-scale solar pv': 0.70,  # 70% crystalline Si
        'CIGS': 0.15,                     # 15% thin film CIGS
        'CdTe': 0.15                      # 15% thin film CdTe
    },
    # ... add your mappings
}
```

### Adjusting Technology Lifetimes

```python
TECHNOLOGY_LIFETIMES = {
    'upv': 30,           # Solar PV: 30 years
    'wind_onshore': 25,  # Wind: 25 years
    # ... modify as needed
}
```

### Running Specific Scenarios

```python
# Run only key scenarios
scenarios_of_interest = ['Mid_Case', 'Mid_Case_No_IRA', 'Mid_Case_100by2035']

simulation, results = run_full_simulation(
    intensity_path='data/intensity_data.csv',
    capacity_path='data/StdScen24_annual_national.csv',
    n_iterations=10000,
    scenarios_to_run=scenarios_of_interest,
    random_state=42
)
```

## Validation and Testing

### Running Tests

```bash
pip install pytest
pytest tests/                         # All tests
pytest tests/ -m "not slow"           # Skip slow simulation tests
python tests/validate_units.py        # Unit conversion validation
```

### Test Coverage
- **Data loading**: Verifies input files load correctly with expected columns
- **Technology mapping**: Validates technology lifetimes and mappings
- **Distribution fitting**: Checks distribution fits for sample materials
- **Simulation regression**: Runs small MC (N=100) to verify output shape, non-negativity, and reproducibility
- **Unit validation**: Confirms t/GW → t/MW conversion is correct

## Known Limitations

1. **Independence Assumption**: Material intensities are assumed independent across technology-material pairs. This may underestimate uncertainty for correlated materials (e.g., foundation materials).

2. **Static Intensities**: Material intensities are assumed constant over time. In reality, learning curves and technological improvements may reduce intensities.

3. **No Recycling**: Model does not account for material recovery from retired capacity.

4. **Retirement Model**: Assumes all baseline capacity was installed in baseline year (no age distribution). This underestimates early-year retirements.

5. **Limited Technologies**: Some emerging technologies (batteries, DAC, electrolyzers) lack material intensity data.

See [`docs/archive/MONTE_CARLO_ASSESSMENT.md`](docs/archive/MONTE_CARLO_ASSESSMENT.md) for detailed discussion of limitations and recommended enhancements.

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make changes with clear commit messages
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## Citation

If you use this model in your research, please cite:

```bibtex
@software{materials_demand_model,
  title = {Materials Demand Model: Monte Carlo Simulation for Energy Infrastructure},
  author = {Griffiths, Paul J.},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/PJGriff26/materials-demand-model}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or collaboration opportunities:
- **Email**: paul.j.griffiths.26@dartmouth.edu
- **Issues**: [GitHub Issues](https://github.com/PJGriff26/materials-demand-model/issues)

## Acknowledgments

- **Data Sources**: NREL Standard Scenarios 2024, NREL Material Intensity Database, USGS Mineral Commodity Summaries 2023
- **Methodology**: Follows ISO/JCGM 101:2008 and NIST uncertainty guidelines
- **Peer Review**: Research-grade implementation validated against published standards

---

**Version**: 2.0.0 (February 2026)
**Status**: Research-Grade - Publication Ready
