# Clustering Analysis for Energy Transition Materials Demand

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify data files exist** (relative to `materials_demand_model/`):
   - `data/StdScen24_annual_national.csv`
   - `data/intensity_data.csv`
   - `data/input_usgs.csv`
   - `outputs/material_demand_by_scenario.csv`

## Running the Analysis

```bash
cd clustering/
python main_analysis.py
```

The pipeline will:
1. Load demand, NREL capacity, and USGS supply-chain data
2. Engineer 12 features per scenario and 12 features per material
3. Log-transform skewed features, check VIF, standardise
4. Run K-means for k=2..10, select k by silhouette score
5. Validate cluster stability via ARI
6. Generate PCA biplots, silhouette plots, profile heatmaps, and stress matrix
7. Export cluster assignments and validation metrics to CSV

### Running individual modules

```bash
python feature_engineering.py   # Test feature construction
python preprocessing.py         # Test preprocessing pipeline
python clustering.py            # Test clustering on synthetic data
```

## Output Structure

```
outputs/clustering/
├── figures/
│   ├── elbow_scenarios.png/pdf
│   ├── elbow_materials.png/pdf
│   ├── pca_biplot_scenarios.png/pdf
│   ├── pca_biplot_materials.png/pdf
│   ├── silhouette_scenarios.png/pdf
│   ├── silhouette_materials.png/pdf
│   ├── cluster_profiles_scenarios.png/pdf
│   ├── cluster_profiles_materials.png/pdf
│   ├── stress_matrix_demand.png/pdf
│   └── feature_sensitivity_*.png/pdf
└── results/
    ├── scenario_clusters.csv
    ├── material_clusters.csv
    ├── scenario_cluster_profiles.csv
    ├── material_cluster_profiles.csv
    ├── scenario_features_raw.csv
    ├── material_features_raw.csv
    ├── stress_matrix.csv
    ├── vif_scenarios.csv
    ├── vif_materials.csv
    └── validation_metrics.txt
```

## Configuration

All parameters are in `config.py`:
- File paths (auto-resolved relative to project root)
- K-means parameters (k range, n_init, tolerance)
- Validation thresholds (silhouette, ARI)
- Figure settings (DPI, format, size)

## Troubleshooting

**"File not found"** — Run from the `clustering/` directory, or edit paths in `config.py`.

**"Module not found"** — `pip install -r requirements.txt`

**High VIF warnings** — The pipeline auto-drops features with VIF > 10. Check `vif_scenarios.csv` / `vif_materials.csv` to see what was removed.
