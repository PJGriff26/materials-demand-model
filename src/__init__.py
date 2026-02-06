"""
Materials Demand Model
======================

A research-grade Monte Carlo simulation framework for estimating material
demand uncertainty in energy infrastructure deployment.

Main Components:
- data_ingestion: Load and validate intensity and capacity data
- distribution_fitting: Fit probability distributions to material intensity data
- technology_mapping: Map capacity technologies to material intensities
- stock_flow_simulation: Monte Carlo simulation with stock-flow accounting
- materials_visualizations: Publication-quality visualization tools

Example Usage:
    from src.stock_flow_simulation import run_full_simulation

    simulation, results = run_full_simulation(
        intensity_path='data/intensity_data.csv',
        capacity_path='data/StdScen24_annual_national.csv',
        n_iterations=10000,
        random_state=42
    )

    # Export results
    stats = results.get_statistics()
    stats.to_csv('outputs/material_demand.csv', index=False)

Author: Materials Demand Research Team
Version: 1.0.0
Date: January 2026
"""

__version__ = '1.0.0'
__author__ = 'Materials Demand Research Team'

from .data_ingestion import (
    MaterialIntensityLoader,
    CapacityProjectionLoader,
    TransmissionCapacityLoader,
    load_all_data
)

from .distribution_fitting import (
    DistributionFitter,
    MaterialIntensityDistribution,
    create_distribution_report
)

from .technology_mapping import (
    TECHNOLOGY_MAPPING,
    TECHNOLOGY_LIFETIMES,
    get_intensity_technologies,
    get_lifetime,
    validate_mapping
)

from .stock_flow_simulation import (
    MaterialsStockFlowSimulation,
    SimulationResult,
    run_full_simulation
)

from .materials_visualizations import (
    MaterialsDemandVisualizer,
    create_spaghetti_plot,
    create_scenario_comparison,
    create_material_comparison
)

__all__ = [
    # Data loading
    'MaterialIntensityLoader',
    'CapacityProjectionLoader',
    'TransmissionCapacityLoader',
    'load_all_data',

    # Distribution fitting
    'DistributionFitter',
    'MaterialIntensityDistribution',
    'create_distribution_report',

    # Technology mapping
    'TECHNOLOGY_MAPPING',
    'TECHNOLOGY_LIFETIMES',
    'get_intensity_technologies',
    'get_lifetime',
    'validate_mapping',

    # Simulation
    'MaterialsStockFlowSimulation',
    'SimulationResult',
    'run_full_simulation',

    # Visualization
    'MaterialsDemandVisualizer',
    'create_spaghetti_plot',
    'create_scenario_comparison',
    'create_material_comparison',
]
