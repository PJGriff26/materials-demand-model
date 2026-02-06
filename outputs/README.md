# Outputs Directory Structure

This directory contains all outputs from the materials demand pipeline.

## Directory Layout

```
outputs/
├── data/                           # Data outputs (CSVs, results)
│   ├── clustering/                 # Clustering analysis results
│   │   ├── scenario_features_raw.csv
│   │   ├── material_features_raw.csv
│   │   ├── scenario_clusters.csv
│   │   ├── material_clusters.csv
│   │   ├── scenario_cluster_profiles.csv
│   │   ├── material_cluster_profiles.csv
│   │   ├── stress_matrix.csv
│   │   ├── vif_scenarios.csv
│   │   ├── vif_materials.csv
│   │   └── validation_metrics.txt
│   └── sensitivity/                # Sensitivity analysis results
│       └── technology_contributions.csv
│
├── figures/                        # All visualizations
│   ├── clustering/                 # Clustering visualizations
│   │   ├── elbow_*.png/pdf
│   │   ├── pca_biplot_*.png/pdf
│   │   ├── silhouette_*.png/pdf
│   │   ├── cluster_profiles_*.png/pdf
│   │   ├── feature_sensitivity_*.png/pdf
│   │   ├── stress_matrix_*.png/pdf
│   │   └── demand_spaghetti_*.png/pdf
│   │
│   ├── demand/                     # Material demand visualizations
│   │   └── (demand time series, projections)
│   │
│   ├── exploratory/                # EDA visualizations
│   │   ├── scenario_scatterplot_matrix.png
│   │   ├── material_scatterplot_matrix.png
│   │   ├── scenario_correlation_heatmap.png
│   │   └── material_correlation_heatmap.png
│   │
│   ├── manuscript/                 # Publication-quality figures
│   │   ├── fig2_technology_breakdown.png
│   │   ├── fig2_technology_pies.png
│   │   ├── figs1_capacity_*.png
│   │   ├── figs1_additions_*.png
│   │   ├── figs3_intensity_distributions.png
│   │   └── figs3_intensity_*_by_tech.png
│   │
│   └── supply_chain/               # Supply chain risk visualizations
│       ├── risk_ranking_chart.png
│       ├── risk_component_heatmap.png
│       └── risk_scores_by_material.csv
│
├── reports/                        # Text reports and summaries
│   └── (analysis reports)
│
└── material_demand_by_scenario.csv # Main simulation output
```

## Generating Outputs

### Main Pipeline
```bash
cd Python/materials_demand_model
python main.py
```

### Clustering Analysis
```bash
cd clustering
python main_analysis.py
```

### Visualization Scripts
```bash
cd visualizations
python manuscript_figures.py      # Manuscript-style figures
python feature_scatterplots.py    # Exploratory analysis
python risk_ranking_chart.py      # Supply chain risk charts
```

## Output Formats

- **Figures**: PNG (150-300 DPI) and PDF formats
- **Data**: CSV files with headers
- **Reports**: Plain text (.txt) or Markdown (.md)

## Documentation

For detailed documentation of variables and visualizations, see:
- `docs/variable_reference.csv` - Master list of all variables, calculations, and data sources
- `docs/visualization_inventory.csv` - Inventory of all visualizations with their input variables
