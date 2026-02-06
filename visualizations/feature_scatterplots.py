"""
Feature Scatterplot Matrices

Creates pairwise scatterplot matrices for scenario and material features
to visualize relationships and identify patterns in the data.

Author: Generated for Materials Demand Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_features():
    """Load scenario and material features from clustering output."""
    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / "outputs" / "data" / "clustering"

    scenario_feats = pd.read_csv(results_dir / "scenario_features_raw.csv", index_col=0)
    material_feats = pd.read_csv(results_dir / "material_features_raw.csv", index_col=0)

    return scenario_feats, material_feats


def create_scatterplot_matrix(df, title, output_path=None, figsize_per_var=1.5):
    """
    Create a pairwise scatterplot matrix using seaborn.

    Parameters
    ----------
    df : DataFrame
        Feature matrix (rows=entities, columns=features).
    title : str
        Title for the figure.
    output_path : Path, optional
        If provided, save figure to this path.
    figsize_per_var : float
        Size multiplier per variable (default 1.5 inches per variable).
    """
    n_vars = len(df.columns)
    figsize = (n_vars * figsize_per_var, n_vars * figsize_per_var)

    # Create pairplot
    g = sns.pairplot(
        df.reset_index(drop=True),
        diag_kind="kde",
        plot_kws={
            "alpha": 0.6,
            "s": 30,
            "edgecolor": "white",
            "linewidth": 0.5,
        },
        diag_kws={
            "fill": True,
            "alpha": 0.6,
        },
        corner=False,  # Show full matrix
    )

    # Adjust figure size
    g.figure.set_size_inches(figsize)

    # Add title
    g.figure.suptitle(title, y=1.02, fontsize=16, fontweight="bold")

    # Rotate x-axis labels for readability
    for ax in g.axes.flatten():
        if ax is not None:
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()

    if output_path:
        g.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return g


def create_correlation_heatmap(df, title, output_path=None):
    """
    Create a correlation heatmap as a companion to the scatterplot matrix.

    Parameters
    ----------
    df : DataFrame
        Feature matrix.
    title : str
        Title for the figure.
    output_path : Path, optional
        If provided, save figure to this path.
    """
    n_vars = len(df.columns)
    figsize = (max(12, n_vars * 0.6), max(10, n_vars * 0.5))

    # Calculate correlation matrix
    corr = df.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # Upper triangle mask
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig, ax


def shorten_column_names(df, max_len=15):
    """Shorten column names for better display in plots."""
    name_map = {
        "total_cumulative_demand": "cum_demand",
        "peak_demand": "peak_demand",
        "mean_demand_early": "early_demand",
        "year_of_peak": "peak_year",
        "demand_slope": "slope",
        "temporal_concentration": "temp_conc",
        "mean_cv": "mean_cv",
        "mean_ci_width": "ci_width",
        "solar_fraction_2035": "solar_frac",
        "wind_fraction_2035": "wind_frac",
        "storage_fraction_2035": "storage_frac",
        "n_active_materials": "n_materials",
        "supply_chain_stress": "sc_stress",
        "peak_supply_chain_stress": "peak_sc_stress",
        "mean_n_exceeding_production": "mean_exceed",
        "peak_n_exceeding_production": "peak_exceed",
        "total_import_exposed_demand": "import_demand",
        "domestic_production": "dom_prod",
        "import_dependency": "import_dep",
        "crc_weighted_risk": "crc_risk",
        "mean_capacity_ratio": "mean_cap_ratio",
        "max_capacity_ratio": "max_cap_ratio",
        "exceedance_frequency": "exceed_freq",
        "reserve_depletion_rate": "reserve_depl",
        "domestic_reserves_years": "dom_res_yrs",
        "global_reserves_years": "glob_res_yrs",
        "reserves_high_risk_frac": "res_high_risk",
        "reserves_oecd_frac": "res_oecd",
        "reserves_china_frac": "res_china",
        "import_china_frac": "imp_china",
        "import_high_risk_frac": "imp_high_risk",
        "import_oecd_frac": "imp_oecd",
        "import_hhi": "imp_hhi",
        "scenario_cv": "scenario_cv",
        "demand_volatility": "volatility",
        "n_active_scenarios": "n_scenarios",
    }

    renamed = {}
    for col in df.columns:
        if col in name_map:
            renamed[col] = name_map[col]
        elif len(col) > max_len:
            renamed[col] = col[:max_len]
        else:
            renamed[col] = col

    return df.rename(columns=renamed)


def main():
    """Generate scatterplot matrices for scenario and material features."""
    print("=" * 70)
    print("FEATURE SCATTERPLOT MATRICES")
    print("=" * 70)

    # Setup output directory
    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / "outputs" / "figures" / "exploratory"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    print("\nLoading features...")
    scenario_feats, material_feats = load_features()
    print(f"  Scenario features: {scenario_feats.shape[0]} scenarios x {scenario_feats.shape[1]} features")
    print(f"  Material features: {material_feats.shape[0]} materials x {material_feats.shape[1]} features")

    # Shorten column names for better display
    scenario_feats_short = shorten_column_names(scenario_feats)
    material_feats_short = shorten_column_names(material_feats)

    # --- Scenario Features ---
    print("\n" + "-" * 50)
    print("SCENARIO FEATURES")
    print("-" * 50)

    print(f"\nGenerating scatterplot matrix ({scenario_feats.shape[1]} features)...")
    print("  This may take a moment...")

    create_scatterplot_matrix(
        scenario_feats_short,
        "Scenario Features - Pairwise Scatterplots",
        output_path=output_dir / "scenario_scatterplot_matrix.png",
        figsize_per_var=1.3,
    )

    print("\nGenerating correlation heatmap...")
    create_correlation_heatmap(
        scenario_feats_short,
        "Scenario Features - Correlation Matrix",
        output_path=output_dir / "scenario_correlation_heatmap.png",
    )

    # --- Material Features ---
    print("\n" + "-" * 50)
    print("MATERIAL FEATURES")
    print("-" * 50)

    print(f"\nGenerating scatterplot matrix ({material_feats.shape[1]} features)...")
    print("  This may take a moment (larger feature set)...")

    create_scatterplot_matrix(
        material_feats_short,
        "Material Features - Pairwise Scatterplots",
        output_path=output_dir / "material_scatterplot_matrix.png",
        figsize_per_var=1.2,
    )

    print("\nGenerating correlation heatmap...")
    create_correlation_heatmap(
        material_feats_short,
        "Material Features - Correlation Matrix",
        output_path=output_dir / "material_correlation_heatmap.png",
    )

    # --- Summary Statistics ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find highly correlated pairs
    print("\nHighly correlated scenario feature pairs (|r| > 0.8):")
    scen_corr = scenario_feats.corr()
    for i in range(len(scen_corr.columns)):
        for j in range(i + 1, len(scen_corr.columns)):
            r = scen_corr.iloc[i, j]
            if abs(r) > 0.8:
                print(f"  {scen_corr.columns[i]} <-> {scen_corr.columns[j]}: r={r:.3f}")

    print("\nHighly correlated material feature pairs (|r| > 0.8):")
    mat_corr = material_feats.corr()
    for i in range(len(mat_corr.columns)):
        for j in range(i + 1, len(mat_corr.columns)):
            r = mat_corr.iloc[i, j]
            if abs(r) > 0.8:
                print(f"  {mat_corr.columns[i]} <-> {mat_corr.columns[j]}: r={r:.3f}")

    print(f"\nOutputs saved to: {output_dir}")
    print("  - scenario_scatterplot_matrix.png")
    print("  - scenario_correlation_heatmap.png")
    print("  - material_scatterplot_matrix.png")
    print("  - material_correlation_heatmap.png")

    plt.show()


if __name__ == "__main__":
    main()
