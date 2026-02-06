# sparse_nmf_analysis.py
"""
Sparse PCA and NMF Feature Importance Analysis

Compares three dimensionality reduction methods:
1. Standard PCA - baseline, all features contribute
2. Sparse PCA - L1 regularization, identifies essential features
3. NMF - non-negative, parts-based decomposition

Generates comparison visualizations and CSV outputs.

Usage:
    python sparse_nmf_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, SparsePCA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import spearmanr
from pathlib import Path

from config import (
    FIGURES_DIR, RESULTS_DIR, FIGURE_DPI, FIGURE_FORMAT,
    FIGSIZE_STANDARD, FIGSIZE_WIDE
)


def _save(fig, stem):
    """Save figure in all configured formats."""
    for fmt in FIGURE_FORMAT:
        fig.savefig(FIGURES_DIR / f"{stem}.{fmt}", dpi=FIGURE_DPI, bbox_inches="tight")


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
    Preprocess features: handle missing values and standardize.

    Returns
    -------
    X_scaled : ndarray
        Standardized feature matrix (mean=0, std=1)
    X_nonneg : ndarray
        Non-negative version for NMF (min-shifted)
    feature_names : list
        Feature column names
    """
    # Drop columns with zero variance
    df_clean = df.copy()
    df_clean = df_clean.loc[:, df_clean.std() > 0]

    # Handle missing/infinite values
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(df_clean.median())

    feature_names = list(df_clean.columns)

    # Standardize for PCA and Sparse PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    # Create non-negative version for NMF (min-shift)
    X_nonneg = X_scaled - X_scaled.min(axis=0)

    return X_scaled, X_nonneg, feature_names


# ============================================================================
# STANDARD PCA
# ============================================================================

def run_pca(X_scaled, feature_names, n_components=5):
    """
    Run standard PCA.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    return pca, loadings, X_pca


def calculate_pca_importance(pca, loadings):
    """
    PCA importance: variance-weighted squared loadings.
    """
    explained_var = pca.explained_variance_ratio_
    weighted_loadings_sq = (loadings ** 2) * explained_var
    importance = weighted_loadings_sq.sum(axis=1)
    importance = importance / importance.sum()

    return importance.sort_values(ascending=False)


# ============================================================================
# SPARSE PCA
# ============================================================================

def run_sparse_pca(X_scaled, feature_names, n_components=5, alpha=1.0):
    """
    Run Sparse PCA with L1 regularization.

    Parameters
    ----------
    alpha : float
        Sparsity controlling parameter. Higher = sparser loadings.
    """
    print(f"    Running Sparse PCA (alpha={alpha})...")
    spca = SparsePCA(
        n_components=n_components,
        alpha=alpha,
        random_state=42,
        max_iter=500
    )
    X_spca = spca.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        spca.components_.T,
        index=feature_names,
        columns=[f"SPC{i+1}" for i in range(n_components)]
    )

    return spca, loadings, X_spca


def calculate_sparse_importance(loadings):
    """
    Sparse PCA importance: sum of squared loadings.
    Also count non-zero loadings per feature.
    """
    # Sum of squared loadings
    importance = (loadings ** 2).sum(axis=1)
    importance = importance / importance.sum()

    # Count non-zero loadings (threshold for numerical zeros)
    threshold = 1e-6
    nonzero_counts = (loadings.abs() > threshold).sum(axis=1)

    return importance.sort_values(ascending=False), nonzero_counts


# ============================================================================
# NMF
# ============================================================================

def run_nmf(X_nonneg, feature_names, n_components=5):
    """
    Run NMF on non-negative data.
    """
    print(f"    Running NMF (n_components={n_components})...")
    nmf = NMF(
        n_components=n_components,
        init='nndsvd',
        random_state=42,
        max_iter=500
    )
    W = nmf.fit_transform(X_nonneg)  # Sample weights
    H = nmf.components_               # Feature loadings

    loadings = pd.DataFrame(
        H.T,
        index=feature_names,
        columns=[f"NMF{i+1}" for i in range(n_components)]
    )

    return nmf, loadings, W


def calculate_nmf_importance(loadings):
    """
    NMF importance: sum of component weights per feature.
    All values are non-negative.
    """
    importance = loadings.sum(axis=1)
    importance = importance / importance.sum()

    return importance.sort_values(ascending=False)


# ============================================================================
# COMPARISON
# ============================================================================

def compare_methods(pca_imp, spca_imp, nmf_imp, spca_nonzero, feature_names):
    """
    Create comparison DataFrame of all methods.
    """
    comparison = pd.DataFrame({
        'feature': feature_names
    })

    # Add PCA results
    comparison['pca_importance'] = comparison['feature'].map(pca_imp)
    comparison['pca_rank'] = comparison['pca_importance'].rank(ascending=False).astype(int)

    # Add Sparse PCA results
    comparison['spca_importance'] = comparison['feature'].map(spca_imp)
    comparison['spca_rank'] = comparison['spca_importance'].rank(ascending=False).astype(int)
    comparison['spca_nonzero'] = comparison['feature'].map(spca_nonzero)

    # Add NMF results
    comparison['nmf_importance'] = comparison['feature'].map(nmf_imp)
    comparison['nmf_rank'] = comparison['nmf_importance'].rank(ascending=False).astype(int)

    # Sort by average rank
    comparison['avg_rank'] = (
        comparison['pca_rank'] + comparison['spca_rank'] + comparison['nmf_rank']
    ) / 3
    comparison = comparison.sort_values('avg_rank')

    return comparison


def calculate_rank_correlations(comparison):
    """
    Calculate Spearman rank correlations between methods.
    """
    pca_ranks = comparison['pca_rank'].values
    spca_ranks = comparison['spca_rank'].values
    nmf_ranks = comparison['nmf_rank'].values

    corr_pca_spca, _ = spearmanr(pca_ranks, spca_ranks)
    corr_pca_nmf, _ = spearmanr(pca_ranks, nmf_ranks)
    corr_spca_nmf, _ = spearmanr(spca_ranks, nmf_ranks)

    return {
        'PCA-SparsePCA': corr_pca_spca,
        'PCA-NMF': corr_pca_nmf,
        'SparsePCA-NMF': corr_spca_nmf
    }


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_sparse_loadings_heatmap(loadings, name):
    """
    Heatmap of Sparse PCA loadings, highlighting zeros.
    """
    fig_height = max(8, len(loadings) * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Sort by first component loading magnitude
    sort_order = loadings.iloc[:, 0].abs().sort_values(ascending=False).index
    loadings_sorted = loadings.loc[sort_order]

    # Custom colormap that emphasizes zeros
    sns.heatmap(
        loadings_sorted,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Loading'},
        mask=(loadings_sorted.abs() < 1e-6)  # Highlight zeros
    )

    # Overlay zeros
    for i in range(loadings_sorted.shape[0]):
        for j in range(loadings_sorted.shape[1]):
            if abs(loadings_sorted.iloc[i, j]) < 1e-6:
                ax.text(j + 0.5, i + 0.5, '0', ha='center', va='center',
                       fontsize=8, color='gray')

    ax.set_title(f'Sparse PCA Loadings — {name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Sparse Component')
    ax.set_ylabel('Feature')

    fig.tight_layout()
    _save(fig, f"sparse_loadings_heatmap_{name}")
    plt.close(fig)
    print(f"  Saved sparse_loadings_heatmap_{name}")


def plot_nmf_components_heatmap(loadings, name):
    """
    Heatmap of NMF components (all non-negative).
    """
    fig_height = max(8, len(loadings) * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Sort by total contribution
    sort_order = loadings.sum(axis=1).sort_values(ascending=False).index
    loadings_sorted = loadings.loc[sort_order]

    sns.heatmap(
        loadings_sorted,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Weight'}
    )

    ax.set_title(f'NMF Components — {name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('NMF Component')
    ax.set_ylabel('Feature')

    fig.tight_layout()
    _save(fig, f"nmf_components_heatmap_{name}")
    plt.close(fig)
    print(f"  Saved nmf_components_heatmap_{name}")


def plot_method_comparison(comparison, name):
    """
    Side-by-side comparison of feature importance across methods.
    """
    n_features = len(comparison)
    fig, axes = plt.subplots(1, 3, figsize=(15, max(6, n_features * 0.3)))

    methods = [
        ('pca_importance', 'PCA', '#4292c6'),
        ('spca_importance', 'Sparse PCA', '#41ab5d'),
        ('nmf_importance', 'NMF', '#ef6548')
    ]

    for ax, (col, title, color) in zip(axes, methods):
        data = comparison.sort_values(col, ascending=True)

        bars = ax.barh(data['feature'], data[col] * 100, color=color, alpha=0.8)
        ax.set_xlabel('Importance (%)')
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add percentage labels
        for bar, val in zip(bars, data[col] * 100):
            if val > 1:
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                       f'{val:.1f}%', va='center', fontsize=7)

    fig.suptitle(f'Feature Importance Comparison — {name}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, f"method_comparison_{name}")
    plt.close(fig)
    print(f"  Saved method_comparison_{name}")


def plot_rank_agreement(comparison, correlations, name):
    """
    Visualize rank agreement between methods.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Bump chart showing rank changes
    features = comparison['feature'].values
    y_positions = range(len(features))

    ax1.plot([0, 1, 2], [comparison['pca_rank'].values,
                          comparison['spca_rank'].values,
                          comparison['nmf_rank'].values], 'o-', alpha=0.5)

    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(['PCA', 'Sparse PCA', 'NMF'])
    ax1.set_ylabel('Rank (1 = most important)')
    ax1.set_title('Rank Changes Across Methods')
    ax1.invert_yaxis()
    ax1.grid(axis='y', alpha=0.3)

    # Correlation heatmap
    corr_matrix = pd.DataFrame({
        'PCA': [1.0, correlations['PCA-SparsePCA'], correlations['PCA-NMF']],
        'Sparse PCA': [correlations['PCA-SparsePCA'], 1.0, correlations['SparsePCA-NMF']],
        'NMF': [correlations['PCA-NMF'], correlations['SparsePCA-NMF'], 1.0]
    }, index=['PCA', 'Sparse PCA', 'NMF'])

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax2,
        cbar_kws={'label': 'Spearman ρ'}
    )
    ax2.set_title('Rank Correlation Between Methods')

    fig.suptitle(f'Method Agreement — {name}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, f"rank_agreement_{name}")
    plt.close(fig)
    print(f"  Saved rank_agreement_{name}")


def plot_sparsity_profile(loadings, name):
    """
    Show sparsity pattern: which features have non-zero loadings.
    """
    threshold = 1e-6
    nonzero_mask = loadings.abs() > threshold

    fig, ax = plt.subplots(figsize=(10, max(6, len(loadings) * 0.3)))

    # Sort by number of non-zero loadings
    sort_order = nonzero_mask.sum(axis=1).sort_values(ascending=True).index
    nonzero_sorted = nonzero_mask.loc[sort_order]

    sns.heatmap(
        nonzero_sorted.astype(int),
        cmap=['white', '#41ab5d'],
        cbar=False,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(f'Sparsity Pattern (Sparse PCA) — {name}', fontweight='bold')
    ax.set_xlabel('Component')
    ax.set_ylabel('Feature')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='gray', label='Zero'),
        Patch(facecolor='#41ab5d', edgecolor='gray', label='Non-zero')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    fig.tight_layout()
    _save(fig, f"sparsity_profile_{name}")
    plt.close(fig)
    print(f"  Saved sparsity_profile_{name}")


# ============================================================================
# INTERPRETABLE VISUALIZATIONS
# ============================================================================

def plot_consensus_importance(comparison, name):
    """
    Single horizontal bar chart showing features ranked by consensus importance.
    Color-codes based on method agreement.
    """
    fig, ax = plt.subplots(figsize=(12, max(8, len(comparison) * 0.4)))

    # Sort by average rank (best at bottom for horizontal bars)
    df = comparison.sort_values('avg_rank', ascending=False).copy()

    # Calculate rank spread (disagreement measure)
    df['rank_spread'] = df[['pca_rank', 'spca_rank', 'nmf_rank']].max(axis=1) - \
                        df[['pca_rank', 'spca_rank', 'nmf_rank']].min(axis=1)

    # Color by agreement: green=good agreement, yellow=moderate, red=poor
    colors = []
    for spread in df['rank_spread']:
        if spread <= 3:
            colors.append('#2ca25f')  # Strong agreement (green)
        elif spread <= 7:
            colors.append('#feb24c')  # Moderate agreement (yellow)
        else:
            colors.append('#de2d26')  # Poor agreement (red)

    # Calculate mean importance across methods
    df['mean_importance'] = (df['pca_importance'] + df['spca_importance'] + df['nmf_importance']) / 3

    bars = ax.barh(df['feature'], df['mean_importance'] * 100, color=colors, edgecolor='black', linewidth=0.5)

    # Add rank labels on bars
    for i, (bar, row) in enumerate(zip(bars, df.itertuples())):
        rank_text = f"#{int(row.avg_rank):.0f}"
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                rank_text, va='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('Mean Importance (%)', fontsize=11)
    ax.set_title(f'Feature Importance Consensus — {name.title()}', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Legend for agreement colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca25f', edgecolor='black', label='Strong agreement (±3 ranks)'),
        Patch(facecolor='#feb24c', edgecolor='black', label='Moderate (±4-7 ranks)'),
        Patch(facecolor='#de2d26', edgecolor='black', label='Methods disagree (>7 ranks)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # Add subtle gridlines
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, f"consensus_importance_{name}")
    plt.close(fig)
    print(f"  Saved consensus_importance_{name}")


def plot_feature_tiers(comparison, name):
    """
    Classify features into importance tiers and visualize as grouped categories.
    """
    n_features = len(comparison)

    # Define tiers based on average rank
    tier_cutoffs = [n_features // 3, 2 * n_features // 3]

    df = comparison.copy()
    df['tier'] = pd.cut(
        df['avg_rank'],
        bins=[0, tier_cutoffs[0], tier_cutoffs[1], n_features + 1],
        labels=['High Importance', 'Medium Importance', 'Low Importance']
    )

    # Create figure with tier breakdown
    fig, axes = plt.subplots(1, 3, figsize=(16, max(6, n_features * 0.15)))

    tier_colors = {
        'High Importance': '#1a9850',
        'Medium Importance': '#fee08b',
        'Low Importance': '#d73027'
    }

    for ax, tier in zip(axes, ['High Importance', 'Medium Importance', 'Low Importance']):
        tier_df = df[df['tier'] == tier].sort_values('avg_rank')

        if len(tier_df) == 0:
            ax.text(0.5, 0.5, 'No features', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(tier)
            continue

        y_pos = range(len(tier_df))

        # Plot bars for each method
        width = 0.25
        x = np.arange(len(tier_df))

        ax.barh(x - width, tier_df['pca_importance'] * 100, width,
                label='PCA', color='#4292c6', alpha=0.8)
        ax.barh(x, tier_df['spca_importance'] * 100, width,
                label='Sparse PCA', color='#41ab5d', alpha=0.8)
        ax.barh(x + width, tier_df['nmf_importance'] * 100, width,
                label='NMF', color='#ef6548', alpha=0.8)

        ax.set_yticks(x)
        ax.set_yticklabels(tier_df['feature'], fontsize=9)
        ax.set_xlabel('Importance (%)')
        ax.set_title(tier, fontsize=12, fontweight='bold',
                    color=tier_colors[tier], bbox=dict(boxstyle='round', facecolor='white', edgecolor=tier_colors[tier]))
        ax.grid(axis='x', alpha=0.3)

        if ax == axes[0]:
            ax.legend(loc='lower right', fontsize=8)

    fig.suptitle(f'Feature Importance by Tier — {name.title()}', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, f"feature_tiers_{name}")
    plt.close(fig)
    print(f"  Saved feature_tiers_{name}")


def plot_method_agreement_dots(comparison, name):
    """
    Dot plot showing each method's rank for each feature with connecting lines.
    Makes disagreement visually obvious.
    """
    fig, ax = plt.subplots(figsize=(10, max(8, len(comparison) * 0.35)))

    # Sort by average rank
    df = comparison.sort_values('avg_rank').reset_index(drop=True)
    y_positions = range(len(df))

    # Plot connecting lines (show spread)
    for i, row in df.iterrows():
        ranks = [row['pca_rank'], row['spca_rank'], row['nmf_rank']]
        ax.plot([min(ranks), max(ranks)], [i, i], 'gray', linewidth=1.5, alpha=0.4, zorder=1)

    # Plot dots for each method
    ax.scatter(df['pca_rank'], y_positions, c='#4292c6', s=80, label='PCA',
              edgecolors='black', linewidth=0.5, zorder=2)
    ax.scatter(df['spca_rank'], y_positions, c='#41ab5d', s=80, label='Sparse PCA',
              marker='s', edgecolors='black', linewidth=0.5, zorder=2)
    ax.scatter(df['nmf_rank'], y_positions, c='#ef6548', s=80, label='NMF',
              marker='^', edgecolors='black', linewidth=0.5, zorder=2)

    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df['feature'])
    ax.set_xlabel('Feature Rank (1 = most important)', fontsize=11)
    ax.set_title(f'Method Agreement by Feature — {name.title()}', fontsize=14, fontweight='bold')

    # Reverse x-axis so rank 1 is on the left
    ax.invert_xaxis()

    ax.legend(loc='lower left', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # Add vertical line at median
    median_rank = len(df) / 2
    ax.axvline(median_rank, color='gray', linestyle='--', alpha=0.5, label='Median')

    fig.tight_layout()
    _save(fig, f"method_agreement_dots_{name}")
    plt.close(fig)
    print(f"  Saved method_agreement_dots_{name}")


def plot_top_features_summary(comparison, name, top_n=10):
    """
    Simple summary showing top N features with agreement indicators.
    Publication-ready format.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get top N by average rank
    top_df = comparison.nsmallest(top_n, 'avg_rank').copy()

    # Calculate agreement score (inverse of rank spread, normalized)
    max_spread = len(comparison) - 1
    top_df['agreement_score'] = 1 - (
        (top_df[['pca_rank', 'spca_rank', 'nmf_rank']].max(axis=1) -
         top_df[['pca_rank', 'spca_rank', 'nmf_rank']].min(axis=1)) / max_spread
    )

    # Sort for display
    top_df = top_df.sort_values('avg_rank')

    y_pos = range(len(top_df))

    # Create horizontal bars
    bars = ax.barh(y_pos, top_df['agreement_score'], color='#3182bd', edgecolor='black', alpha=0.8)

    # Add feature names and rank info
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df['feature'], fontsize=11)

    # Add text annotations showing individual method ranks
    for i, row in enumerate(top_df.itertuples()):
        rank_text = f"PCA:{int(row.pca_rank)} | SpPCA:{int(row.spca_rank)} | NMF:{int(row.nmf_rank)}"
        ax.text(row.agreement_score + 0.02, i, rank_text, va='center', fontsize=8, color='gray')

    ax.set_xlabel('Agreement Score (1.0 = perfect agreement)', fontsize=11)
    ax.set_xlim(0, 1.4)
    ax.set_title(f'Top {top_n} Most Important Features — {name.title()}', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    _save(fig, f"top_features_summary_{name}")
    plt.close(fig)
    print(f"  Saved top_features_summary_{name}")


def plot_summary_dashboard(comparison, correlations, name, pca_loadings, spca_loadings, nmf_loadings):
    """
    Single-page dashboard combining key insights for stakeholders.
    """
    fig = plt.figure(figsize=(16, 12))

    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # ── Panel 1: Top 10 Features (large, left side) ──
    ax1 = fig.add_subplot(gs[0:2, 0:2])

    top_10 = comparison.nsmallest(10, 'avg_rank').sort_values('avg_rank')
    y_pos = range(len(top_10))
    mean_imp = (top_10['pca_importance'] + top_10['spca_importance'] + top_10['nmf_importance']) / 3

    # Color by tier
    colors = ['#1a9850' if i < 4 else '#fee08b' if i < 7 else '#fdae61' for i in range(len(top_10))]
    bars = ax1.barh(y_pos, mean_imp * 100, color=colors, edgecolor='black')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_10['feature'], fontsize=10)
    ax1.set_xlabel('Mean Importance (%)')
    ax1.set_title('Top 10 Features by Consensus', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # ── Panel 2: Method Correlation Matrix ──
    ax2 = fig.add_subplot(gs[0, 2])

    corr_matrix = pd.DataFrame({
        'PCA': [1.0, correlations['PCA-SparsePCA'], correlations['PCA-NMF']],
        'SPCA': [correlations['PCA-SparsePCA'], 1.0, correlations['SparsePCA-NMF']],
        'NMF': [correlations['PCA-NMF'], correlations['SparsePCA-NMF'], 1.0]
    }, index=['PCA', 'SPCA', 'NMF'])

    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', vmin=-1, vmax=1,
                center=0, ax=ax2, cbar_kws={'label': 'ρ', 'shrink': 0.8}, square=True)
    ax2.set_title('Method Agreement\n(Spearman ρ)', fontsize=11, fontweight='bold')

    # ── Panel 3: Key Statistics ──
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis('off')

    n_features = len(comparison)
    n_high_tier = (comparison['avg_rank'] <= n_features/3).sum()

    # Count features where methods strongly agree (spread <= 3)
    comparison_temp = comparison.copy()
    comparison_temp['spread'] = comparison_temp[['pca_rank', 'spca_rank', 'nmf_rank']].max(axis=1) - \
                                comparison_temp[['pca_rank', 'spca_rank', 'nmf_rank']].min(axis=1)
    n_strong_agree = (comparison_temp['spread'] <= 3).sum()

    stats_text = f"""Summary Statistics
─────────────────────
Total features: {n_features}

Strong agreement: {n_strong_agree}
(rank spread ≤ 3)

Top tier features: {n_high_tier}
(avg rank ≤ {n_features//3})

Mean correlation: {np.mean(list(correlations.values())):.2f}"""

    ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    # ── Panel 4: High Disagreement Features ──
    ax4 = fig.add_subplot(gs[2, 0])

    high_disagree = comparison_temp.nlargest(5, 'spread')[['feature', 'pca_rank', 'spca_rank', 'nmf_rank']]
    y_pos = range(len(high_disagree))

    for i, row in enumerate(high_disagree.itertuples()):
        ax4.plot([row.pca_rank, row.spca_rank, row.nmf_rank], [i, i, i], 'o-', markersize=8)
        ax4.text(max(row.pca_rank, row.spca_rank, row.nmf_rank) + 0.5, i,
                row.feature, va='center', fontsize=9)

    ax4.set_xlabel('Rank')
    ax4.set_yticks([])
    ax4.set_title('Features with\nHighest Disagreement', fontsize=11, fontweight='bold')
    ax4.invert_xaxis()

    # ── Panel 5: Method Insights ──
    ax5 = fig.add_subplot(gs[2, 1:])
    ax5.axis('off')

    # Find unique top features per method
    pca_top3 = comparison.nsmallest(3, 'pca_rank')['feature'].tolist()
    spca_top3 = comparison.nsmallest(3, 'spca_rank')['feature'].tolist()
    nmf_top3 = comparison.nsmallest(3, 'nmf_rank')['feature'].tolist()

    insights_text = f"""Method-Specific Top Features
─────────────────────────────

PCA emphasizes:         {', '.join(pca_top3)}
Sparse PCA selects:     {', '.join(spca_top3)}
NMF identifies:         {', '.join(nmf_top3)}

Interpretation Guide:
• PCA: Features driving overall variance
• Sparse PCA: Essential features (others can be dropped)
• NMF: Features that co-occur in additive "parts" """

    ax5.text(0.05, 0.95, insights_text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f7f7f7', edgecolor='gray'))

    fig.suptitle(f'Dimensionality Reduction Summary — {name.title()}',
                fontsize=16, fontweight='bold', y=0.98)

    _save(fig, f"summary_dashboard_{name}")
    plt.close(fig)
    print(f"  Saved summary_dashboard_{name}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_dataset(df, name, n_components=5, spca_alpha=1.0):
    """
    Run complete analysis on a dataset with all three methods.
    """
    print(f"\n{'='*60}")
    print(f"DIMENSIONALITY REDUCTION COMPARISON: {name.upper()}")
    print(f"{'='*60}")
    print(f"  Input: {df.shape[0]} samples × {df.shape[1]} features")

    # Preprocess
    X_scaled, X_nonneg, feature_names = preprocess_features(df)
    print(f"  After preprocessing: {len(feature_names)} features")

    # Adjust n_components if needed
    max_components = min(X_scaled.shape[0] - 1, len(feature_names), n_components)
    print(f"  Using {max_components} components")

    # Run all three methods
    print("\n  Running analyses...")

    # PCA
    pca, pca_loadings, X_pca = run_pca(X_scaled, feature_names, max_components)
    pca_importance = calculate_pca_importance(pca, pca_loadings)

    # Sparse PCA
    spca, spca_loadings, X_spca = run_sparse_pca(
        X_scaled, feature_names, max_components, spca_alpha
    )
    spca_importance, spca_nonzero = calculate_sparse_importance(spca_loadings)

    # NMF
    nmf, nmf_loadings, W_nmf = run_nmf(X_nonneg, feature_names, max_components)
    nmf_importance = calculate_nmf_importance(nmf_loadings)

    # Compare methods
    comparison = compare_methods(
        pca_importance, spca_importance, nmf_importance,
        spca_nonzero, feature_names
    )
    correlations = calculate_rank_correlations(comparison)

    # Print summary
    print("\n  Sparsity summary (Sparse PCA):")
    n_active = (spca_nonzero > 0).sum()
    print(f"    Features with non-zero loadings: {n_active}/{len(feature_names)}")

    print("\n  Rank correlations between methods:")
    for pair, corr in correlations.items():
        print(f"    {pair}: ρ = {corr:.3f}")

    print("\n  Top 5 features by each method:")
    print("    PCA:", list(pca_importance.head(5).index))
    print("    Sparse PCA:", list(spca_importance.head(5).index))
    print("    NMF:", list(nmf_importance.head(5).index))

    # Generate visualizations
    print("\n  Generating visualizations...")

    # Technical visualizations
    plot_sparse_loadings_heatmap(spca_loadings, name)
    plot_nmf_components_heatmap(nmf_loadings, name)
    plot_method_comparison(comparison, name)
    plot_rank_agreement(comparison, correlations, name)
    plot_sparsity_profile(spca_loadings, name)

    # Interpretable visualizations
    print("\n  Generating interpretable visualizations...")
    plot_consensus_importance(comparison, name)
    plot_feature_tiers(comparison, name)
    plot_method_agreement_dots(comparison, name)
    plot_top_features_summary(comparison, name, top_n=min(10, len(comparison)))
    plot_summary_dashboard(comparison, correlations, name, pca_loadings, spca_loadings, nmf_loadings)

    # Save comparison CSV
    print("\n  Saving CSV outputs...")
    comparison.to_csv(RESULTS_DIR / f"method_comparison_{name}.csv", index=False)
    print(f"  Saved method_comparison_{name}.csv")

    spca_loadings.to_csv(RESULTS_DIR / f"sparse_pca_loadings_{name}.csv")
    print(f"  Saved sparse_pca_loadings_{name}.csv")

    nmf_loadings.to_csv(RESULTS_DIR / f"nmf_loadings_{name}.csv")
    print(f"  Saved nmf_loadings_{name}.csv")

    return {
        'pca': (pca, pca_loadings, pca_importance),
        'spca': (spca, spca_loadings, spca_importance, spca_nonzero),
        'nmf': (nmf, nmf_loadings, nmf_importance),
        'comparison': comparison,
        'correlations': correlations
    }


def main():
    """Run dimensionality reduction comparison on both datasets."""
    print("=" * 70)
    print("SPARSE PCA AND NMF ANALYSIS")
    print("=" * 70)

    # Load features
    print("\nLoading feature data...")
    scenario_feats, material_feats = load_features()
    print(f"  Scenarios: {scenario_feats.shape}")
    print(f"  Materials: {material_feats.shape}")

    # Analyze scenarios (n/p ratio is acceptable)
    scenario_results = analyze_dataset(
        scenario_feats,
        "scenarios",
        n_components=5,
        spca_alpha=1.0
    )

    # Analyze materials (HDLSS - use higher sparsity)
    material_results = analyze_dataset(
        material_feats,
        "materials",
        n_components=5,
        spca_alpha=2.0  # Higher alpha for sparser solution
    )

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to:")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Data: {RESULTS_DIR}")
    print("\nKey findings:")

    # Scenarios
    scen_active = (scenario_results['spca'][3] > 0).sum()
    print(f"\n  SCENARIOS:")
    print(f"    Sparse PCA selected {scen_active}/{len(scenario_feats.columns)} features")

    # Materials
    mat_active = (material_results['spca'][3] > 0).sum()
    print(f"\n  MATERIALS:")
    print(f"    Sparse PCA selected {mat_active}/{len(material_feats.columns)} features")
    print(f"    (Important for HDLSS data with n/p = {len(material_feats)}/{len(material_feats.columns)} = {len(material_feats)/len(material_feats.columns):.2f})")

    return scenario_results, material_results


if __name__ == "__main__":
    main()
