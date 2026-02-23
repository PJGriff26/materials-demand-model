# Outputs Directory Structure

This directory contains all outputs from the materials demand pipeline.

## Directory Layout

```
outputs/
├── material_demand_by_scenario.csv    # Full MC results (scenario × year × material)
├── material_demand_summary.csv        # Aggregated summary across all scenarios
├── simulation_report.txt              # Comprehensive text report
│
├── data/                              # Data outputs (CSVs, results)
│   ├── clustering/                    # Clustering analysis results
│   │   ├── material_features_raw.csv
│   │   ├── scenario_features_raw.csv
│   │   ├── material_clusters.csv
│   │   ├── scenario_clusters.csv
│   │   ├── material_cluster_profiles.csv
│   │   ├── scenario_cluster_profiles.csv
│   │   ├── spca_scores_*.csv          # Sparse PCA scores
│   │   ├── sparse_pca_loadings_*.csv  # Sparse PCA loadings
│   │   ├── nmf_loadings_*.csv         # NMF loadings
│   │   ├── pca_loadings_*.csv         # PCA loadings
│   │   ├── fa_loadings_*.csv          # Factor analysis loadings
│   │   ├── method_comparison_*.csv    # Dimensionality reduction comparison
│   │   ├── stress_matrix.csv
│   │   ├── validation_metrics.txt
│   │   └── comparison/                # 4-method comparison metrics
│   │       ├── comparison_labels_*.csv
│   │       ├── comparison_metrics_*.csv
│   │       ├── comparison_pairwise_ari_*.csv
│   │       └── comparison_summary_report.txt
│   └── sensitivity/                   # Sensitivity analysis results
│       ├── variance_decomposition.csv
│       ├── variance_decomposition_by_year.csv
│       ├── uncertainty_decomposition.csv
│       ├── intensity_elasticity.csv
│       ├── spearman_sensitivity.csv
│       ├── spearman_iteration_data.csv
│       ├── scenario_sensitivity.csv
│       ├── temporal_sensitivity.csv
│       ├── technology_contributions.csv
│       ├── material_correlations.csv
│       └── high_correlation_pairs.csv
│
├── figures/                           # All visualizations
│   ├── manuscript/                    # Publication-quality figures
│   │   ├── fig1_demand_curves.png
│   │   ├── fig1_cumulative_demand.png
│   │   ├── fig2_technology_*.png
│   │   ├── fig3_*supply_risk*.png
│   │   ├── fig4_reserve_adequacy*.png
│   │   ├── figs1_capacity_*.png       # SI capacity figures
│   │   ├── figs1_additions_*.png      # SI additions figures
│   │   ├── figs3_intensity_*.png      # SI intensity figures
│   │   └── si_fig_*.png               # Additional SI figures
│   │
│   ├── clustering/                    # Clustering visualizations
│   │   ├── kmeans/                    # Elbow, silhouette, biplot, profiles
│   │   ├── pca_analysis/              # Scree, loadings, feature importance
│   │   ├── dimensionality_reduction/  # SPCA, NMF, method comparison
│   │   ├── spca_story/                # Named components, quadrant plots
│   │   ├── factor_analysis/           # FA loadings & communalities
│   │   └── comparison/                # 4-method clustering comparison
│   │
│   ├── sensitivity/                   # Sensitivity analysis figures
│   │   ├── spearman_*.png             # Spearman correlation plots
│   │   ├── variance_*.png             # Variance decomposition
│   │   └── tornado_elasticity.png
│   │
│   ├── exploratory/                   # EDA figures
│   │   ├── scenario_scatterplot_matrix.png
│   │   ├── material_scatterplot_matrix.png
│   │   ├── scenario_correlation_heatmap.png
│   │   └── material_correlation_heatmap.png
│   │
│   ├── supply_chain_risk/             # Supply chain risk visualizations
│   │   ├── risk_ranking_chart.png
│   │   ├── risk_component_heatmap.png
│   │   └── risk_analysis_stacked_bars_*.png
│   │
│   └── archive/                       # Historical figures
│       ├── figure_validation/         # Before/after comparison PNGs
│       └── prototype/                 # Early prototype visualizations
│
└── reports/                           # Text reports and summaries
    ├── simulation_report.txt
    ├── sensitivity_analysis_report.txt
    ├── critical_findings.txt
    ├── data_quality_report.txt
    ├── risk_analysis_summary.txt
    └── validation_metrics.txt
```

## Generating Outputs

### Full Pipeline (Recommended)
```bash
python run_pipeline.py                    # Full pipeline (Steps 1–10, ~20 min)
python run_pipeline.py --skip-simulation  # Skip MC if outputs exist (~5 min)
python run_pipeline.py --step 5           # Run only one step
python run_pipeline.py --from 3           # Resume from Step 3
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
python manuscript_fig1.py         # Fig. 1 demand curves
python feature_scatterplots.py    # Exploratory analysis
python risk_ranking_chart.py      # Supply chain risk charts
python compare_figures.py         # Before/after figure comparison
```

## Output Formats

- **Figures**: PNG (300 DPI)
- **Data**: CSV files with headers
- **Reports**: Plain text (.txt)

## Documentation

For detailed documentation of variables and visualizations, see:
- `docs/variable_reference.csv` - Master list of all variables, calculations, and data sources
- `docs/visualization_inventory.csv` - Inventory of all visualizations with their input variables
