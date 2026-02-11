# pca_feature_importance.py
"""
PCA Feature Importance Analysis

Performs comprehensive PCA to identify which features contribute most to
variance in scenario and material feature sets. Generates visualizations
and CSV outputs for feature importance ranking.

Usage:
    python pca_feature_importance.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from config import (
    FIGURES_PCA_DIR, RESULTS_DIR, FIGURE_DPI, FIGURE_FORMAT,
    FIGSIZE_STANDARD, FIGSIZE_WIDE
)


def _save(fig, stem):
    """Save figure in all configured formats."""
    for fmt in FIGURE_FORMAT:
        fig.savefig(FIGURES_PCA_DIR / f"{stem}.{fmt}", dpi=FIGURE_DPI, bbox_inches="tight")


def load_features():
    """Load raw features from clustering outputs."""
    scenario_path = RESULTS_DIR / "scenario_features_raw.csv"
    material_path = RESULTS_DIR / "material_features_raw.csv"

    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario features not found: {scenario_path}")
    if not material_path.exists():
        raise FileNotFoundError(f"Material features not found: {material_path}")

    scenario_feats = pd.read_csv(scenario_path, index_col=0)
    material_feats = pd.read_csv(material_path, index_col=0)

    return scenario_feats, material_feats


def preprocess_features(df):
    """
    Preprocess features for PCA: handle missing values and standardize.

    Returns
    -------
    X_scaled : ndarray
        Standardized feature matrix
    feature_names : list
        Feature column names
    scaler : StandardScaler
        Fitted scaler for reference
    """
    # Drop columns with zero variance
    df_clean = df.copy()
    df_clean = df_clean.loc[:, df_clean.std() > 0]

    # Handle missing/infinite values
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(df_clean.median())

    feature_names = list(df_clean.columns)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    return X_scaled, feature_names, scaler


def run_full_pca(X_scaled, feature_names):
    """
    Run PCA with all components.

    Returns
    -------
    pca : PCA
        Fitted PCA object
    loadings : DataFrame
        Feature loadings matrix (features x components)
    """
    n_components = min(X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    # Create loadings DataFrame
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

    return pca, loadings


def calculate_feature_importance(pca, loadings):
    """
    Calculate feature importance using variance-weighted loadings.

    importance_i = sum_j(|loading_ij|^2 * explained_variance_j)

    This measures how much each feature contributes to the total
    explained variance across all principal components.
    """
    # Squared loadings weighted by explained variance
    explained_var = pca.explained_variance_ratio_
    weighted_loadings_sq = (loadings ** 2) * explained_var

    # Sum across components
    importance = weighted_loadings_sq.sum(axis=1)

    # Normalize to sum to 1
    importance = importance / importance.sum()

    importance_df = pd.DataFrame({
        'feature': importance.index,
        'importance': importance.values,
        'importance_pct': importance.values * 100
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    importance_df['cumulative_pct'] = importance_df['importance_pct'].cumsum()
    importance_df['rank'] = range(1, len(importance_df) + 1)

    return importance_df


def plot_scree(pca, name):
    """
    Scree plot showing explained variance by component.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    n_components = len(pca.explained_variance_ratio_)
    components = range(1, n_components + 1)
    var_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(var_ratio)

    # Individual variance
    ax1.bar(components, var_ratio * 100, color='#4292c6', alpha=0.8)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance (%)')
    ax1.set_title(f'Variance by Component — {name}')
    ax1.set_xticks(components)
    ax1.grid(axis='y', alpha=0.3)

    # Cumulative variance
    ax2.plot(components, cumulative * 100, 'o-', color='#2c7bb6', linewidth=2)
    ax2.axhline(80, color='gray', linestyle='--', alpha=0.7, label='80%')
    ax2.axhline(90, color='gray', linestyle=':', alpha=0.7, label='90%')
    ax2.axhline(95, color='gray', linestyle='-.', alpha=0.7, label='95%')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance (%)')
    ax2.set_title(f'Cumulative Variance — {name}')
    ax2.set_xticks(components)
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 105)

    # Mark thresholds
    for threshold in [0.80, 0.90, 0.95]:
        n_for_threshold = np.argmax(cumulative >= threshold) + 1
        if cumulative[n_for_threshold - 1] >= threshold:
            ax2.axvline(n_for_threshold, color='red', alpha=0.3, linestyle=':')
            ax2.annotate(
                f'{n_for_threshold} PCs\n({threshold:.0%})',
                xy=(n_for_threshold, threshold * 100),
                xytext=(n_for_threshold + 0.5, threshold * 100 - 5),
                fontsize=8, color='red'
            )

    fig.tight_layout()
    _save(fig, f"scree_plot_{name}")
    plt.close(fig)
    print(f"  Saved scree_plot_{name}")


def plot_loadings_heatmap(loadings, name, n_pcs=None):
    """
    Heatmap of feature loadings on principal components.
    """
    if n_pcs is None:
        n_pcs = min(10, loadings.shape[1])

    loadings_subset = loadings.iloc[:, :n_pcs]

    # Sort features by absolute loading on PC1
    sort_order = loadings_subset['PC1'].abs().sort_values(ascending=False).index
    loadings_sorted = loadings_subset.loc[sort_order]

    fig_height = max(8, len(loadings_sorted) * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    sns.heatmap(
        loadings_sorted,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Loading'}
    )

    ax.set_title(f'PCA Loadings — {name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Feature')

    fig.tight_layout()
    _save(fig, f"loadings_heatmap_{name}")
    plt.close(fig)
    print(f"  Saved loadings_heatmap_{name}")


def plot_feature_importance(importance_df, name):
    """
    Horizontal bar chart of feature importance.
    """
    fig, ax = plt.subplots(figsize=(10, max(6, len(importance_df) * 0.35)))

    # Sort for display (least important at top, most important at bottom)
    df_sorted = importance_df.sort_values('importance', ascending=True)

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(df_sorted)))

    bars = ax.barh(df_sorted['feature'], df_sorted['importance_pct'], color=colors)

    ax.set_xlabel('Feature Importance (%)')
    ax.set_title(f'Feature Importance (Variance-Weighted) — {name}', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add percentage labels on bars
    for bar, pct in zip(bars, df_sorted['importance_pct']):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f'{pct:.1f}%',
            va='center',
            fontsize=8
        )

    ax.set_xlim(0, df_sorted['importance_pct'].max() * 1.15)

    fig.tight_layout()
    _save(fig, f"feature_importance_{name}")
    plt.close(fig)
    print(f"  Saved feature_importance_{name}")


def plot_biplot(X_scaled, pca, feature_names, name, pc_x=1, pc_y=2, entity_names=None):
    """
    PCA biplot showing samples and loading vectors.
    """
    # Transform data
    X_pca = pca.transform(X_scaled)

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    # Plot samples
    ax.scatter(
        X_pca[:, pc_x - 1],
        X_pca[:, pc_y - 1],
        alpha=0.6,
        edgecolors='k',
        linewidth=0.4,
        s=60,
        c='#4292c6'
    )

    # Add entity labels if provided
    if entity_names is not None:
        for idx, txt in enumerate(entity_names):
            ax.annotate(
                txt,
                (X_pca[idx, pc_x - 1], X_pca[idx, pc_y - 1]),
                fontsize=6,
                alpha=0.6,
                textcoords='offset points',
                xytext=(4, 4)
            )

    # Loading vectors
    loadings = pca.components_.T
    scale = max(np.abs(X_pca[:, [pc_x - 1, pc_y - 1]]).max(axis=0)) * 0.8

    for j, feat in enumerate(feature_names):
        ax.annotate(
            '',
            xy=(loadings[j, pc_x - 1] * scale, loadings[j, pc_y - 1] * scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.2, alpha=0.7)
        )
        ax.text(
            loadings[j, pc_x - 1] * scale * 1.1,
            loadings[j, pc_y - 1] * scale * 1.1,
            feat,
            fontsize=7,
            color='red',
            ha='center',
            alpha=0.8
        )

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC{pc_x} ({ev[pc_x - 1]:.1%} variance)')
    ax.set_ylabel(f'PC{pc_y} ({ev[pc_y - 1]:.1%} variance)')
    ax.set_title(f'PCA Biplot (PC{pc_x} vs PC{pc_y}) — {name}')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    _save(fig, f"biplot_pc{pc_x}{pc_y}_{name}")
    plt.close(fig)
    print(f"  Saved biplot_pc{pc_x}{pc_y}_{name}")


def create_explained_variance_csv(pca, name):
    """Save explained variance to CSV."""
    n_components = len(pca.explained_variance_ratio_)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    df = pd.DataFrame({
        'component': [f'PC{i+1}' for i in range(n_components)],
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': cumulative,
        'explained_variance_pct': pca.explained_variance_ratio_ * 100,
        'cumulative_variance_pct': cumulative * 100
    })

    output_path = RESULTS_DIR / f"pca_explained_variance_{name}.csv"
    df.to_csv(output_path, index=False)
    print(f"  Saved {output_path.name}")

    return df


def analyze_dataset(df, name, show_entity_labels=True):
    """
    Run complete PCA analysis on a feature dataset.
    """
    print(f"\n{'='*60}")
    print(f"PCA Analysis: {name.upper()}")
    print(f"{'='*60}")
    print(f"  Input: {df.shape[0]} samples × {df.shape[1]} features")

    # Preprocess
    X_scaled, feature_names, scaler = preprocess_features(df)
    print(f"  After preprocessing: {X_scaled.shape[1]} features")

    # Run PCA
    pca, loadings = run_full_pca(X_scaled, feature_names)
    print(f"  PCA components: {pca.n_components_}")

    # Summary stats
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    for threshold in [0.80, 0.90, 0.95]:
        n_for_threshold = np.argmax(cumulative >= threshold) + 1
        print(f"  Components for {threshold:.0%} variance: {n_for_threshold}")

    # Calculate feature importance
    importance_df = calculate_feature_importance(pca, loadings)

    print("\n  Top 5 features by importance:")
    for _, row in importance_df.head(5).iterrows():
        print(f"    {row['rank']}. {row['feature']}: {row['importance_pct']:.1f}%")

    # Generate visualizations
    print("\n  Generating visualizations...")
    plot_scree(pca, name)
    plot_loadings_heatmap(loadings, name)
    plot_feature_importance(importance_df, name)

    entity_names = list(df.index) if show_entity_labels else None
    plot_biplot(X_scaled, pca, feature_names, name, pc_x=1, pc_y=2, entity_names=entity_names)

    if pca.n_components_ >= 3:
        plot_biplot(X_scaled, pca, feature_names, name, pc_x=2, pc_y=3, entity_names=entity_names)

    # Save CSVs
    print("\n  Saving CSV outputs...")
    create_explained_variance_csv(pca, name)

    loadings.to_csv(RESULTS_DIR / f"pca_loadings_{name}.csv")
    print(f"  Saved pca_loadings_{name}.csv")

    importance_df.to_csv(RESULTS_DIR / f"feature_importance_{name}.csv", index=False)
    print(f"  Saved feature_importance_{name}.csv")

    return pca, loadings, importance_df


def main():
    """Run PCA feature importance analysis on both datasets."""
    print("=" * 70)
    print("PCA FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    # Load features
    print("\nLoading feature data...")
    scenario_feats, material_feats = load_features()
    print(f"  Scenarios: {scenario_feats.shape}")
    print(f"  Materials: {material_feats.shape}")

    # Analyze scenarios
    scenario_pca, scenario_loadings, scenario_importance = analyze_dataset(
        scenario_feats, "scenarios", show_entity_labels=True
    )

    # Analyze materials
    material_pca, material_loadings, material_importance = analyze_dataset(
        material_feats, "materials", show_entity_labels=True
    )

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to:")
    print(f"  Figures: {FIGURES_PCA_DIR}")
    print(f"  Data: {RESULTS_DIR}")
    print("\nKey outputs:")
    print("  - scree_plot_*.png: Explained variance by component")
    print("  - loadings_heatmap_*.png: Feature contributions to PCs")
    print("  - feature_importance_*.png: Ranked feature importance")
    print("  - feature_importance_*.csv: Importance scores for analysis")

    return {
        'scenarios': (scenario_pca, scenario_loadings, scenario_importance),
        'materials': (material_pca, material_loadings, material_importance)
    }


if __name__ == "__main__":
    main()
