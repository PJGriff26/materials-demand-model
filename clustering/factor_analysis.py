# factor_analysis.py
"""
Factor Analysis for clustering input.

Uses sklearn.decomposition.FactorAnalysis to extract latent factors
from standardized features. Generates scores, loadings, and diagnostic
visualizations.

Key difference from PCA: FA models observed variables as linear combinations
of latent factors plus unique noise, separating shared variance (communality)
from feature-specific variance (uniqueness).

Usage:
    cd clustering/
    python factor_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

from config import (
    FIGURES_FA_DIR, RESULTS_DIR, FIGURE_DPI, FIGURE_FORMAT,
    FIGSIZE_STANDARD, FIGSIZE_WIDE, FA_N_COMPONENTS,
)


def _save(fig, stem):
    """Save figure in all configured formats."""
    for fmt in FIGURE_FORMAT:
        fig.savefig(FIGURES_FA_DIR / f"{stem}.{fmt}", dpi=FIGURE_DPI, bbox_inches="tight")


def preprocess_features(df):
    """
    Preprocess features: drop zero-variance, handle missing/inf, standardize.

    Returns
    -------
    X_scaled : ndarray
        Standardized feature matrix (mean=0, std=1).
    feature_names : list
        Feature column names after cleaning.
    """
    df_clean = df.copy()
    df_clean = df_clean.loc[:, df_clean.std() > 0]
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(df_clean.median())

    feature_names = list(df_clean.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    return X_scaled, feature_names


def run_factor_analysis(X_scaled, feature_names, n_components, random_state=42):
    """
    Fit Factor Analysis model.

    Returns
    -------
    fa : FactorAnalysis
        Fitted model.
    loadings : DataFrame
        Features × factors loading matrix.
    scores : DataFrame
        Samples × factors score matrix.
    """
    fa = FactorAnalysis(
        n_components=n_components,
        max_iter=1000,
        random_state=random_state,
    )
    X_fa = fa.fit_transform(X_scaled)

    col_names = [f"FA{i+1}" for i in range(n_components)]

    loadings = pd.DataFrame(
        fa.components_.T,
        index=feature_names,
        columns=col_names,
    )

    scores = pd.DataFrame(X_fa, columns=col_names)

    return fa, loadings, scores


def calculate_communalities(fa, loadings):
    """
    Compute communalities and uniqueness for each feature.

    Communality = fraction of variance explained by the factor model.
    Uniqueness = fa.noise_variance_ (feature-specific noise).
    """
    communality = (loadings ** 2).sum(axis=1)

    comm_df = pd.DataFrame({
        "feature": loadings.index,
        "communality": communality.values,
        "uniqueness": fa.noise_variance_,
    })
    comm_df = comm_df.sort_values("communality", ascending=False).reset_index(drop=True)
    return comm_df


def calculate_fa_importance(loadings):
    """
    Feature importance: normalized sum of squared loadings across all factors.
    """
    importance = (loadings ** 2).sum(axis=1)
    importance = importance / importance.sum()
    return importance.sort_values(ascending=False)


def plot_fa_loadings_heatmap(loadings, name):
    """Heatmap of factor loadings."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.heatmap(
        loadings, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(f"Factor Analysis Loadings — {name.title()}", fontsize=14)
    ax.set_ylabel("Feature")
    ax.set_xlabel("Factor")
    plt.tight_layout()
    _save(fig, f"fa_loadings_heatmap_{name}")
    plt.close(fig)
    print(f"  Saved fa_loadings_heatmap_{name}")


def plot_communalities(comm_df, name):
    """Horizontal bar chart of communalities with 0.5 threshold."""
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    colors = ["#2ca02c" if c >= 0.5 else "#d62728" for c in comm_df["communality"]]
    ax.barh(range(len(comm_df)), comm_df["communality"], color=colors)
    ax.set_yticks(range(len(comm_df)))
    ax.set_yticklabels(comm_df["feature"], fontsize=9)
    ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1, label="Threshold (0.5)")
    ax.set_xlabel("Communality")
    ax.set_title(f"Factor Analysis Communalities — {name.title()}", fontsize=14)
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, f"fa_communalities_{name}")
    plt.close(fig)
    print(f"  Saved fa_communalities_{name}")


def analyze_dataset(df, name, n_components, entity_index=None):
    """
    Run complete Factor Analysis pipeline on a dataset.

    Parameters
    ----------
    df : DataFrame
        Raw features (rows=entities, columns=features).
    name : str
        Dataset name (e.g. "scenarios", "materials").
    n_components : int
        Number of latent factors.
    entity_index : Index, optional
        Row index for scores (defaults to df.index).

    Returns
    -------
    dict with keys: fa, loadings, scores, importance, communalities
    """
    print(f"\n{'='*60}")
    print(f"FACTOR ANALYSIS: {name.upper()}")
    print(f"{'='*60}")
    print(f"  Input: {df.shape[0]} samples x {df.shape[1]} features")

    X_scaled, feature_names = preprocess_features(df)
    print(f"  After preprocessing: {len(feature_names)} features")

    max_components = min(X_scaled.shape[0] - 1, len(feature_names), n_components)
    print(f"  Using {max_components} factors")

    fa, loadings, scores = run_factor_analysis(
        X_scaled, feature_names, max_components
    )

    if entity_index is not None:
        scores.index = entity_index
    else:
        scores.index = df.index

    comm_df = calculate_communalities(fa, loadings)
    importance = calculate_fa_importance(loadings)

    # Print summary
    n_well_captured = (comm_df["communality"] >= 0.5).sum()
    print(f"\n  Communalities: {n_well_captured}/{len(comm_df)} features >= 0.5")
    print(f"  Mean communality: {comm_df['communality'].mean():.3f}")
    print(f"  Log-likelihood: {fa.score(X_scaled).mean():.3f}")

    print(f"\n  Top 5 features by importance:")
    for feat, imp in importance.head(5).items():
        print(f"    {feat}: {imp:.3f}")

    # Generate visualizations
    print("\n  Generating visualizations...")
    plot_fa_loadings_heatmap(loadings, name)
    plot_communalities(comm_df, name)

    # Save CSVs
    print("\n  Saving CSV outputs...")
    scores.to_csv(RESULTS_DIR / f"fa_scores_{name}.csv")
    print(f"  Saved fa_scores_{name}.csv")

    loadings.to_csv(RESULTS_DIR / f"fa_loadings_{name}.csv")
    print(f"  Saved fa_loadings_{name}.csv")

    comm_df.to_csv(RESULTS_DIR / f"fa_communalities_{name}.csv", index=False)
    print(f"  Saved fa_communalities_{name}.csv")

    return {
        "fa": fa,
        "loadings": loadings,
        "scores": scores,
        "importance": importance,
        "communalities": comm_df,
    }


def main():
    """Run Factor Analysis on both datasets."""
    print("=" * 70)
    print("FACTOR ANALYSIS")
    print("=" * 70)

    # Load features
    scenario_path = RESULTS_DIR / "scenario_features_raw.csv"
    material_path = RESULTS_DIR / "material_features_raw.csv"

    if not scenario_path.exists() or not material_path.exists():
        raise FileNotFoundError(
            "Raw feature files not found. Run main_analysis.py first.\n"
            f"  Expected: {scenario_path}\n"
            f"  Expected: {material_path}"
        )

    scenario_feats = pd.read_csv(scenario_path, index_col=0)
    material_feats = pd.read_csv(material_path, index_col=0)
    print(f"\nLoaded features:")
    print(f"  Scenarios: {scenario_feats.shape}")
    print(f"  Materials: {material_feats.shape}")

    # Analyze scenarios
    scenario_results = analyze_dataset(
        scenario_feats, "scenarios",
        n_components=FA_N_COMPONENTS["scenarios"],
    )

    # Analyze materials
    material_results = analyze_dataset(
        material_feats, "materials",
        n_components=FA_N_COMPONENTS["materials"],
    )

    print("\n" + "=" * 70)
    print("FACTOR ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to:")
    print(f"  Figures: {FIGURES_FA_DIR}")
    print(f"  Data: {RESULTS_DIR}")

    return scenario_results, material_results


if __name__ == "__main__":
    main()
