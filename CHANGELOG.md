# Changelog

All notable changes to the Materials Demand Model will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-26

### Added
- Initial release of research-grade Monte Carlo simulation framework
- Complete stock-flow accounting model for material demand
- Robust distribution fitting with multiple parametric and empirical options
- Technology mapping system for capacity-to-intensity relationships
- Publication-quality visualization suite
- Comprehensive documentation and examples
- Unit validation testing framework

### Fixed
- **CRITICAL**: Unit conversion from t/GW to t/MW (previous versions overestimated by 1000×)
  - See `docs/UNIT_FIX_SUMMARY.md` for details
  - All results from versions prior to 1.0.0 are invalid

### Technical Details
- Monte Carlo implementation follows ISO/JCGM 101:2008 standards
- N=10,000 iterations with full percentile reporting (2.5-97.5)
- Support for 21 technologies and 31 materials
- 61 NREL Standard Scenarios included

### Known Limitations
- Material intensities assumed independent (no correlation structure)
- Static intensities over time (no learning curves)
- No material recycling/recovery modeled
- Simplified retirement model (baseline stock assumed new)

## [2.1.0] - 2026-03-05

### Added
- **Technology consolidation preprocessing** (`src/data_ingestion.py`, Step 1c):
  - CDTE (BOS materials) merged into CdTe during data loading
  - ASIGE (BOS materials) merged into a-Si during data loading
  - Consolidation mapping defined in `TECHNOLOGY_CONSOLIDATION` dict (`src/technology_mapping.py`)
  - CdTe now has 13 materials (was 9), including structural BOS materials (Aluminum, Cement, Copper, Glass, Steel)
  - a-Si now has 14 materials (was 10), including the same BOS materials
  - Raw data (`intensity_data.csv`) unchanged; consolidation applied at preprocessing
- **Data quality module** (`src/data_quality.py`):
  - Known corrections and removals for intensity data outliers
  - IQR-based statistical outlier detection

### Changed
- **UPV sub-technology weights** updated from 70/15/15 to 90/7/3 (c-Si/CdTe/CIGS):
  - Based on peer-reviewed market share data (Fraunhofer ISE 2024, IEA-PVPS 2024, US EIA)
  - c-Si: 90% (was 70%) — reflects ~95-98% global production share
  - CdTe: 7% (was 15%) — reflects ~2-5% global, ~22% US utility-scale
  - CIGS: 3% (was 15%) — reflects <1% global market share
- Post-consolidation, pipeline processes 19 technologies (was 21 in raw data)

### Fixed
- **Missing BOS materials for CdTe**: The CdTe fraction of UPV was previously missing balance-of-system materials (Aluminum, Cement, Copper, Glass, Steel), causing ~15% underestimate of structural material demand from solar PV

## [2.0.0] - 2026-02-08

### Added
- **Sensitivity analysis module** (`analysis/sensitivity_analysis.py`):
  - Variance decomposition (within-scenario MC vs between-scenario)
  - Intensity elasticity analysis (% demand change per 1% intensity change)
  - Spearman rank correlation across Monte Carlo iterations
  - Temporal variance decomposition (by year)
- **4-method clustering comparison** (`clustering/clustering_comparison.py`):
  - Systematic comparison of VIF-Pruned, PCA, Sparse PCA, and Factor Analysis as preprocessing for K-means
  - 7 comparison visualizations (silhouette panels, ARI heatmap, stability bars, biplot gallery, membership matrix, summary table, optimal-k)
  - Conclusion: Sparse PCA is the recommended method (best silhouette, interpretability, consensus)
- **Factor Analysis module** (`clustering/factor_analysis.py`):
  - sklearn FactorAnalysis with communality diagnostics
  - Loadings heatmap and communality bar chart visualizations
- **Sparse PCA interpretation** (`clustering/sparse_pca_story.py`):
  - Named components (e.g., "Demand Scale", "Geopolitical Risk")
  - Risk quadrant plots and material story dashboards
- **Manuscript figures** (`visualizations/manuscript_fig1.py`):
  - Fig. 1 demand curves with median + 95% interval + IQR bands
  - Cumulative demand bar charts with uncertainty
- **Full pipeline runner** (`run_pipeline.py`):
  - Single entry point for complete reproducibility (10 steps)
  - Supports `--step`, `--from`, `--skip-simulation` flags
- **Pytest test suite** (`tests/test_pipeline.py`):
  - Data loading, technology mapping, distribution fitting, and simulation regression tests
- **Data provenance documentation** (`data/README.md`):
  - Source citations, access terms, and column descriptions for all input datasets
- **Supply chain analysis** (`clustering/supply_chain_analysis.py`):
  - CRC-weighted demand sourcing (Fig. 3)
  - Reserve adequacy by CRC category (Fig. 4, global + US)
  - Production shares visualization (SI figure)

### Changed
- **requirements.txt**: Pinned all dependencies to exact versions for reproducibility
- **Figure output organization**: Reorganized ~140 clustering figures into 7 categorical subfolders (kmeans/, pca_analysis/, dimensionality_reduction/, spca_story/, supply_chain/, factor_analysis/, comparison/)
- **Documentation**: Moved 9 superseded diagnostic docs to `docs/archive/`
- **README.md**: Updated to reflect current repository structure and pipeline

### Removed
- Stale VIF-experiment files (20 figures + 2 CSVs from early prototyping)

## [1.0.0] - 2026-01-26

### Added
- Initial release of research-grade Monte Carlo simulation framework
- Complete stock-flow accounting model for material demand
- Robust distribution fitting with multiple parametric and empirical options
- Technology mapping system for capacity-to-intensity relationships
- Publication-quality visualization suite
- Comprehensive documentation and examples
- Unit validation testing framework

### Fixed
- **CRITICAL**: Unit conversion from t/GW to t/MW (previous versions overestimated by 1000x)
  - See `docs/archive/UNIT_FIX_SUMMARY.md` for details
  - All results from versions prior to 1.0.0 are invalid

### Technical Details
- Monte Carlo implementation follows ISO/JCGM 101:2008 standards
- N=10,000 iterations with full percentile reporting (2.5-97.5)
- Support for 21 technologies and 31 materials
- 61 NREL Standard Scenarios included

### Known Limitations
- Material intensities assumed independent (no correlation structure)
- Static intensities over time (no learning curves)
- No material recycling/recovery modeled
- Simplified retirement model (baseline stock assumed new)
