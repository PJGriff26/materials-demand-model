# Analysis module for materials demand model
from .sensitivity_analysis import (
    # Variance decomposition
    compute_variance_decomposition,
    compute_variance_decomposition_by_year,
    # Elasticity analysis
    compute_intensity_elasticity,
    compute_demand_sensitivity_to_intensity,
    # Spearman correlation sensitivity
    compute_spearman_sensitivity,
    compute_spearman_sensitivity_from_simulation,
    run_spearman_from_files,
    # Visualizations
    plot_variance_decomposition,
    plot_tornado_diagram,
    plot_variance_by_year,
    plot_spearman_tornado,
    plot_spearman_heatmap,
    plot_spearman_scatter,
    plot_spearman_summary,
    # Reporting
    generate_sensitivity_report,
)

__all__ = [
    # Variance decomposition
    'compute_variance_decomposition',
    'compute_variance_decomposition_by_year',
    # Elasticity analysis
    'compute_intensity_elasticity',
    'compute_demand_sensitivity_to_intensity',
    # Spearman correlation sensitivity
    'compute_spearman_sensitivity',
    'compute_spearman_sensitivity_from_simulation',
    'run_spearman_from_files',
    # Visualizations
    'plot_variance_decomposition',
    'plot_tornado_diagram',
    'plot_variance_by_year',
    'plot_spearman_tornado',
    'plot_spearman_heatmap',
    'plot_spearman_scatter',
    'plot_spearman_summary',
    # Reporting
    'generate_sensitivity_report',
]
