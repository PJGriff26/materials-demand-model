# clustering_comparison.py
"""
Clustering comparison: evaluate K-means with 4 different input representations.

Methods:
  1. VIF-Pruned: Log transform + iterative VIF drop + StandardScaler
  2. PCA: Standard PCA projecting to N components (80% cumulative variance)
  3. Sparse PCA: L1-regularized PCA (pre-tuned components and alpha)
  4. Factor Analysis: Latent factor model (sklearn FactorAnalysis)

For each method, runs:
  - Optimal k search (elbow + silhouette)
  - Final K-means fit
  - Stability validation (ARI across seeds)
  - Feature sensitivity (leave-one-out)
  - Cluster profiles (raw feature means)

Then generates cross-method comparison visualizations and a summary report.

Usage:
    cd clustering/
    python sparse_nmf_analysis.py   # generates SPCA scores (if not already run)
    python factor_analysis.py       # generates FA scores (if not already run)
    python clustering_comparison.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, SparsePCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples

from config import (
    RESULTS_DIR, FIGURE_DPI, FIGURE_FORMAT,
    FIGSIZE_STANDARD, FIGSIZE_WIDE, RANDOM_SEED,
    COMPARISON_FIGURES_DIR, COMPARISON_DATA_DIR,
    METHOD_LABELS, METHOD_KEYS,
    SPCA_N_COMPONENTS, SPCA_ALPHA, FA_N_COMPONENTS,
    SCENARIO_LOG_FEATURES, MATERIAL_LOG_FEATURES,
)
from feature_engineering import (
    load_demand_data, load_nrel_data, load_risk_data, load_usgs_2023_thin_film,
    engineer_scenario_features, engineer_material_features,
)
from preprocessing import preprocess_pipeline
from clustering import ClusterAnalyzer


# ============================================================================
# HELPERS
# ============================================================================

def _save(fig, stem):
    """Save figure to comparison directory in all configured formats."""
    for fmt in FIGURE_FORMAT:
        fig.savefig(
            COMPARISON_FIGURES_DIR / f"{stem}.{fmt}",
            dpi=FIGURE_DPI, bbox_inches="tight",
        )


METHOD_COLORS = {
    "vif": "#1f77b4",
    "pca": "#ff7f0e",
    "spca": "#2ca02c",
    "fa": "#d62728",
}


# ============================================================================
# INPUT PREPARATION
# ============================================================================

def prepare_vif_input(raw_feats, log_features, vif_threshold=10.0):
    """Prepare VIF-pruned features for clustering."""
    print("\n  [VIF] Log transform + VIF pruning + StandardScaler...")
    X_std, scaler, vif_after, dropped = preprocess_pipeline(
        raw_feats, log_features=log_features, vif_threshold=vif_threshold,
    )
    print(f"  [VIF] Surviving features: {X_std.shape[1]} ({list(X_std.columns)})")
    return X_std


def prepare_pca_input(raw_feats, variance_threshold=0.80):
    """Prepare PCA-projected features for clustering."""
    print("\n  [PCA] StandardScaler + PCA (80% cumulative variance)...")
    df_clean = raw_feats.copy()
    df_clean = df_clean.loc[:, df_clean.std() > 0]
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(df_clean.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    # Full PCA to find N for 80% variance
    pca_full = PCA(random_state=RANDOM_SEED)
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_components = min(n_components, X_scaled.shape[0] - 1, X_scaled.shape[1])

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)

    col_names = [f"PC{i+1}" for i in range(n_components)]
    X_df = pd.DataFrame(X_pca, index=raw_feats.index, columns=col_names)

    total_var = cumvar[n_components - 1] * 100
    print(f"  [PCA] {n_components} components, {total_var:.1f}% cumulative variance")
    return X_df


def prepare_spca_input(dataset_name, raw_feats=None):
    """Load pre-computed Sparse PCA scores, or run inline if missing."""
    path = RESULTS_DIR / f"spca_scores_{dataset_name}.csv"
    if path.exists():
        print(f"\n  [SPCA] Loading {path.name}...")
        X_df = pd.read_csv(path, index_col=0)
    else:
        print(f"\n  [SPCA] Scores not found, running inline...")
        df_clean = raw_feats.copy()
        df_clean = df_clean.loc[:, df_clean.std() > 0]
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.fillna(df_clean.median())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean)

        n_comp = SPCA_N_COMPONENTS[dataset_name]
        alpha = SPCA_ALPHA[dataset_name]
        spca = SparsePCA(n_components=n_comp, alpha=alpha, max_iter=500,
                         random_state=RANDOM_SEED)
        X_spca = spca.fit_transform(X_scaled)

        col_names = [f"SPC{i+1}" for i in range(n_comp)]
        X_df = pd.DataFrame(X_spca, index=raw_feats.index, columns=col_names)

    print(f"  [SPCA] {X_df.shape[1]} components: {list(X_df.columns)}")
    return X_df


def prepare_fa_input(dataset_name, raw_feats=None):
    """Load pre-computed Factor Analysis scores, or run inline if missing."""
    path = RESULTS_DIR / f"fa_scores_{dataset_name}.csv"
    if path.exists():
        print(f"\n  [FA] Loading {path.name}...")
        X_df = pd.read_csv(path, index_col=0)
    else:
        print(f"\n  [FA] Scores not found, running inline...")
        df_clean = raw_feats.copy()
        df_clean = df_clean.loc[:, df_clean.std() > 0]
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.fillna(df_clean.median())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean)

        n_comp = FA_N_COMPONENTS[dataset_name]
        fa = FactorAnalysis(n_components=n_comp, max_iter=1000,
                            random_state=RANDOM_SEED)
        X_fa = fa.fit_transform(X_scaled)

        col_names = [f"FA{i+1}" for i in range(n_comp)]
        X_df = pd.DataFrame(X_fa, index=raw_feats.index, columns=col_names)

    print(f"  [FA] {X_df.shape[1]} factors: {list(X_df.columns)}")
    return X_df


# ============================================================================
# CORE CLUSTERING
# ============================================================================

def apply_k_guards(best_k, sil_scores, k_range, dataset_name):
    """Apply interpretability guards on k (same logic as main_analysis.py)."""
    if dataset_name == "scenarios" and best_k > 7:
        sil_arr = np.array(sil_scores)
        candidates = [(k_range[i], sil_arr[i])
                      for i in range(len(sil_arr)) if 2 <= k_range[i] <= 7]
        if candidates:
            best_k = max(candidates, key=lambda x: x[1])[0]
            print(f"    → Capping to k={best_k} for interpretability")

    if dataset_name == "materials" and best_k == 2:
        sil_arr = np.array(sil_scores)
        candidates = [(k_range[i], sil_arr[i])
                      for i in range(len(sil_arr)) if 3 <= k_range[i] <= 7]
        if candidates:
            best_k = max(candidates, key=lambda x: x[1])[0]
            print(f"    → Overriding k=2 to k={best_k} for interpretability")

    return best_k


def run_single_method(X_df, raw_feats, method_key, dataset_name):
    """Run full clustering pipeline for one method."""
    method_label = dict(zip(METHOD_KEYS, METHOD_LABELS))[method_key]
    print(f"\n  --- {method_label} ({X_df.shape[1]} features) ---")

    analyzer = ClusterAnalyzer(X_df, name=f"{dataset_name}_{method_key}")
    k_range, wcss, sil_scores = analyzer.find_optimal_k()

    best_k = k_range[int(np.argmax(sil_scores))]
    best_k = apply_k_guards(best_k, sil_scores, k_range, dataset_name)

    labels = analyzer.fit_final_model(best_k)
    mean_ari = analyzer.validate_stability()
    sensitivity = analyzer.feature_sensitivity()
    profiles = analyzer.get_cluster_profiles(raw_feats)

    overall_sil = silhouette_score(X_df.values, labels)

    return {
        "X": X_df,
        "analyzer": analyzer,
        "labels": labels,
        "k": best_k,
        "silhouette": overall_sil,
        "stability_ari": mean_ari,
        "stability_ari_std": analyzer.results.get("stability_ari_std", 0),
        "n_features": X_df.shape[1],
        "feature_names": list(X_df.columns),
        "k_range": k_range,
        "wcss": wcss,
        "sil_scores": sil_scores,
        "sensitivity": sensitivity,
        "profiles": profiles,
    }


def run_all_methods(dataset_name, raw_feats, log_features):
    """Prepare inputs and run clustering for all 4 methods."""
    print(f"\n{'='*70}")
    print(f"CLUSTERING COMPARISON: {dataset_name.upper()}")
    print(f"{'='*70}")

    X_vif = prepare_vif_input(raw_feats, log_features)
    X_pca = prepare_pca_input(raw_feats)
    X_spca = prepare_spca_input(dataset_name, raw_feats)
    X_fa = prepare_fa_input(dataset_name, raw_feats)

    inputs = {"vif": X_vif, "pca": X_pca, "spca": X_spca, "fa": X_fa}

    results = {}
    for key in METHOD_KEYS:
        results[key] = run_single_method(
            inputs[key], raw_feats, key, dataset_name,
        )

    return results


# ============================================================================
# COMPARISON METRICS
# ============================================================================

def compute_comparison_metrics(results):
    """Build metrics DataFrame with one row per method."""
    rows = []
    for key, label in zip(METHOD_KEYS, METHOD_LABELS):
        r = results[key]
        counts = np.bincount(r["labels"])
        rows.append({
            "method": label,
            "method_key": key,
            "n_input_features": r["n_features"],
            "optimal_k": r["k"],
            "silhouette": r["silhouette"],
            "stability_ari": r["stability_ari"],
            "stability_ari_std": r["stability_ari_std"],
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
            "cluster_balance": 1 - (counts.max() - counts.min()) / len(r["labels"]),
        })
    return pd.DataFrame(rows)


def compute_pairwise_ari(results):
    """Compute 4x4 pairwise ARI matrix between methods."""
    ari_matrix = pd.DataFrame(
        np.eye(len(METHOD_KEYS)),
        index=METHOD_LABELS,
        columns=METHOD_LABELS,
    )
    for i, (ki, li) in enumerate(zip(METHOD_KEYS, METHOD_LABELS)):
        for j, (kj, lj) in enumerate(zip(METHOD_KEYS, METHOD_LABELS)):
            if i < j:
                ari = adjusted_rand_score(results[ki]["labels"], results[kj]["labels"])
                ari_matrix.loc[li, lj] = ari
                ari_matrix.loc[lj, li] = ari
    return ari_matrix


# ============================================================================
# COMPARISON VISUALIZATIONS
# ============================================================================

def plot_silhouette_comparison(results, dataset_name):
    """2x2 panel silhouette plots, one per method."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(zip(METHOD_KEYS, METHOD_LABELS)):
        ax = axes[idx]
        r = results[key]
        X = r["X"].values
        labels = r["labels"]
        n_clusters = r["k"]

        sil_samples = silhouette_samples(X, labels)
        y_lower = 0

        for c in range(n_clusters):
            mask = labels == c
            c_vals = np.sort(sil_samples[mask])
            y_upper = y_lower + len(c_vals)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, c_vals, alpha=0.7)
            y_lower = y_upper + 2

        ax.axvline(x=r["silhouette"], color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{label}\nk={n_clusters}, sil={r['silhouette']:.3f}", fontsize=12)
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_ylabel("Sample")

    fig.suptitle(f"Silhouette Comparison — {dataset_name.title()}", fontsize=16, y=1.02)
    plt.tight_layout()
    _save(fig, f"comparison_silhouette_{dataset_name}")
    plt.close(fig)


def plot_optimal_k_comparison(results, dataset_name):
    """Silhouette vs k curves + chosen k bar chart."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[2, 1], hspace=0.35)

    # Top row: silhouette vs k for each method
    for idx, (key, label) in enumerate(zip(METHOD_KEYS, METHOD_LABELS)):
        ax = fig.add_subplot(gs[0, idx])
        r = results[key]
        ax.plot(r["k_range"], r["sil_scores"], "o-", color=METHOD_COLORS[key])
        ax.axvline(x=r["k"], color="red", linestyle="--", alpha=0.7)
        ax.set_title(f"{label}", fontsize=11)
        ax.set_xlabel("k")
        if idx == 0:
            ax.set_ylabel("Silhouette Score")

    # Bottom: bar chart of chosen k and silhouette
    ax_bar = fig.add_subplot(gs[1, :])
    x = np.arange(len(METHOD_KEYS))
    ks = [results[k]["k"] for k in METHOD_KEYS]
    sils = [results[k]["silhouette"] for k in METHOD_KEYS]
    colors = [METHOD_COLORS[k] for k in METHOD_KEYS]

    bars = ax_bar.bar(x, sils, color=colors, width=0.6)
    for i, (k_val, s_val) in enumerate(zip(ks, sils)):
        ax_bar.text(i, s_val + 0.01, f"k={k_val}\n{s_val:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(METHOD_LABELS, fontsize=11)
    ax_bar.set_ylabel("Silhouette Score")
    ax_bar.set_title("Chosen k and Silhouette Score per Method", fontsize=12)

    fig.suptitle(f"Optimal k Comparison — {dataset_name.title()}", fontsize=16)
    plt.tight_layout()
    _save(fig, f"comparison_optimal_k_{dataset_name}")
    plt.close(fig)


def plot_pairwise_ari_heatmap(ari_matrix, dataset_name):
    """4x4 heatmap of pairwise ARI between methods."""
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        ari_matrix.values.astype(float), annot=True, fmt=".3f",
        cmap="RdYlGn", vmin=-0.1, vmax=1.0,
        xticklabels=METHOD_LABELS, yticklabels=METHOD_LABELS,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(f"Pairwise ARI Between Methods — {dataset_name.title()}", fontsize=14)
    plt.tight_layout()
    _save(fig, f"comparison_ari_heatmap_{dataset_name}")
    plt.close(fig)


def plot_stability_comparison(results, dataset_name):
    """Bar chart of stability ARI per method."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(METHOD_KEYS))
    aris = [results[k]["stability_ari"] for k in METHOD_KEYS]
    stds = [results[k]["stability_ari_std"] for k in METHOD_KEYS]

    colors = []
    for a in aris:
        if a > 0.9:
            colors.append("#2ca02c")
        elif a > 0.7:
            colors.append("#ff7f0e")
        else:
            colors.append("#d62728")

    ax.bar(x, aris, yerr=stds, color=colors, width=0.6, capsize=5)
    ax.axhline(y=0.8, color="black", linestyle="--", linewidth=1, label="Stability threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_LABELS, fontsize=11)
    ax.set_ylabel("Stability ARI (mean ± std)")
    ax.set_title(f"Clustering Stability — {dataset_name.title()}", fontsize=14)
    ax.legend()
    ax.set_ylim(0, 1.05)

    for i, (a, s) in enumerate(zip(aris, stds)):
        ax.text(i, a + s + 0.02, f"{a:.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    _save(fig, f"comparison_stability_{dataset_name}")
    plt.close(fig)


def plot_biplot_gallery(results, raw_feats, dataset_name):
    """2x2 biplots using the SAME PCA projection, colored by each method's clusters."""
    # Fit a common PCA on standardized raw features for consistent spatial layout
    df_clean = raw_feats.copy()
    df_clean = df_clean.loc[:, df_clean.std() > 0]
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(df_clean.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    pca_common = PCA(n_components=2, random_state=RANDOM_SEED)
    X_2d = pca_common.fit_transform(X_scaled)
    var1, var2 = pca_common.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(zip(METHOD_KEYS, METHOD_LABELS)):
        ax = axes[idx]
        r = results[key]
        labels = r["labels"]
        n_clusters = r["k"]

        scatter = ax.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=labels, cmap="tab10", s=60, alpha=0.8, edgecolors="white", linewidth=0.5,
        )

        # Mark centroids
        for c in range(n_clusters):
            mask = labels == c
            cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
            ax.scatter(cx, cy, marker="X", s=200, c="black", zorder=5)
            ax.annotate(str(c), (cx, cy), fontsize=9, fontweight="bold",
                        ha="center", va="bottom", xytext=(0, 8),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        ax.set_title(f"{label} (k={n_clusters}, sil={r['silhouette']:.3f})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel(f"PC1 ({var1:.1f}%)")
        ax.set_ylabel(f"PC2 ({var2:.1f}%)")

    fig.suptitle(
        f"Biplot Gallery (Common PCA Projection) — {dataset_name.title()}",
        fontsize=16, y=1.02,
    )
    plt.tight_layout()
    _save(fig, f"comparison_biplot_gallery_{dataset_name}")
    plt.close(fig)


def plot_comparison_summary_table(metrics_df, dataset_name):
    """Render metrics table as a figure."""
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")

    cols = ["method", "n_input_features", "optimal_k", "silhouette",
            "stability_ari", "min_cluster_size", "max_cluster_size", "cluster_balance"]
    display_cols = ["Method", "# Features", "k", "Silhouette", "Stability ARI",
                    "Min Cluster", "Max Cluster", "Balance"]

    cell_text = []
    for _, row in metrics_df.iterrows():
        cell_text.append([
            row["method"],
            str(row["n_input_features"]),
            str(row["optimal_k"]),
            f"{row['silhouette']:.3f}",
            f"{row['stability_ari']:.3f}",
            str(row["min_cluster_size"]),
            str(row["max_cluster_size"]),
            f"{row['cluster_balance']:.3f}",
        ])

    table = ax.table(
        cellText=cell_text, colLabels=display_cols,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Bold the best silhouette and ARI
    best_sil_idx = metrics_df["silhouette"].idxmax()
    best_ari_idx = metrics_df["stability_ari"].idxmax()
    for row_idx in range(len(metrics_df)):
        if row_idx == best_sil_idx:
            table[row_idx + 1, 3].set_text_props(fontweight="bold")
            table[row_idx + 1, 3].set_facecolor("#d4edda")
        if row_idx == best_ari_idx:
            table[row_idx + 1, 4].set_text_props(fontweight="bold")
            table[row_idx + 1, 4].set_facecolor("#d4edda")

    ax.set_title(f"Clustering Comparison Summary — {dataset_name.title()}",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    _save(fig, f"comparison_summary_{dataset_name}")
    plt.close(fig)


def plot_membership_matrix(results, entity_names, dataset_name):
    """Dot-matrix showing cluster assignment per entity per method."""
    n_entities = len(entity_names)
    fig_height = max(8, n_entities * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Build matrix: rows=entities, columns=methods
    membership = pd.DataFrame(index=entity_names)
    for key, label in zip(METHOD_KEYS, METHOD_LABELS):
        membership[label] = results[key]["labels"]

    # Use a qualitative colormap
    max_k = max(r["k"] for r in results.values())
    cmap = plt.colormaps.get_cmap("tab10").resampled(max_k)

    for col_idx, method in enumerate(METHOD_LABELS):
        for row_idx, entity in enumerate(entity_names):
            cluster = membership.loc[entity, method]
            ax.scatter(col_idx, row_idx, c=[cmap(cluster)], s=80,
                       edgecolors="white", linewidth=0.5)

    ax.set_xticks(range(len(METHOD_LABELS)))
    ax.set_xticklabels(METHOD_LABELS, fontsize=11, rotation=15, ha="right")
    ax.set_yticks(range(n_entities))
    ax.set_yticklabels(entity_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f"Cluster Membership Matrix — {dataset_name.title()}", fontsize=14)

    plt.tight_layout()
    _save(fig, f"comparison_membership_{dataset_name}")
    plt.close(fig)


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def generate_summary_report(all_results, all_metrics, all_ari):
    """Write human-readable comparison report."""
    lines = []
    lines.append("=" * 70)
    lines.append("CLUSTERING COMPARISON SUMMARY REPORT")
    lines.append("=" * 70)
    lines.append("")

    for ds_name in ["scenarios", "materials"]:
        metrics = all_metrics[ds_name]
        ari = all_ari[ds_name]

        lines.append(f"\n{'─'*50}")
        lines.append(f"  {ds_name.upper()}")
        lines.append(f"{'─'*50}")

        lines.append("\n  Method Comparison:")
        lines.append(f"  {'Method':<20} {'k':>3} {'Silhouette':>12} {'Stability':>12} {'Features':>10}")
        lines.append(f"  {'─'*57}")
        for _, row in metrics.iterrows():
            lines.append(
                f"  {row['method']:<20} {row['optimal_k']:>3} "
                f"{row['silhouette']:>12.3f} {row['stability_ari']:>12.3f} "
                f"{row['n_input_features']:>10}"
            )

        # Find best method
        best_sil = metrics.loc[metrics["silhouette"].idxmax()]
        best_ari_row = metrics.loc[metrics["stability_ari"].idxmax()]

        lines.append(f"\n  Best silhouette: {best_sil['method']} ({best_sil['silhouette']:.3f})")
        lines.append(f"  Best stability:  {best_ari_row['method']} ({best_ari_row['stability_ari']:.3f})")

        lines.append(f"\n  Pairwise ARI between methods:")
        lines.append(ari.round(3).to_string())

        # Mean pairwise ARI per method (agreement with others)
        mean_ari = {}
        for m in METHOD_LABELS:
            others = [ari.loc[m, m2] for m2 in METHOD_LABELS if m2 != m]
            mean_ari[m] = np.mean(others)
        best_consensus = max(mean_ari, key=mean_ari.get)
        lines.append(f"\n  Mean pairwise ARI (agreement with other methods):")
        for m, a in sorted(mean_ari.items(), key=lambda x: -x[1]):
            lines.append(f"    {m:<20} {a:.3f}")

    # Overall recommendation
    lines.append(f"\n\n{'='*70}")
    lines.append("RECOMMENDATION")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Sparse PCA is recommended as the primary clustering input because:")
    lines.append("  1. Sparsity is ideal for HDLSS data (31 materials, 24 features)")
    lines.append("  2. Components are directly interpretable (named: Demand Scale, etc.)")
    lines.append("  3. L1 regularization handles multicollinearity without VIF's order-dependence")
    lines.append("  4. See empirical metrics above to confirm or challenge this recommendation")
    lines.append("")
    lines.append("If PCA or FA shows substantially higher silhouette/stability,")
    lines.append("consider switching the production pipeline (main_analysis.py).")
    lines.append("")

    report_text = "\n".join(lines)

    report_path = COMPARISON_DATA_DIR / "comparison_summary_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n  Saved: {report_path}")

    return report_text


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("CLUSTERING COMPARISON PIPELINE")
    print("  4 Methods x 2 Datasets")
    print("=" * 70)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print("\n▸ Step 1: Loading data...")
    demand = load_demand_data()
    nrel = load_nrel_data()
    risk_data = load_risk_data()
    thin_film = load_usgs_2023_thin_film()

    # ── Step 2: Feature engineering ───────────────────────────────────────
    print("\n▸ Step 2: Feature engineering...")
    scenario_feats = engineer_scenario_features(demand, nrel, risk_data, thin_film)
    material_feats = engineer_material_features(demand, risk_data, thin_film)
    print(f"  Scenario features: {scenario_feats.shape}")
    print(f"  Material features: {material_feats.shape}")

    all_results = {}
    all_metrics = {}
    all_ari = {}

    # ── Step 3: Run comparison for each dataset ───────────────────────────
    for ds_name, raw_feats, log_feats in [
        ("scenarios", scenario_feats, SCENARIO_LOG_FEATURES),
        ("materials", material_feats, MATERIAL_LOG_FEATURES),
    ]:
        print(f"\n\n{'#'*70}")
        print(f"  DATASET: {ds_name.upper()}")
        print(f"{'#'*70}")

        results = run_all_methods(ds_name, raw_feats, log_feats)
        all_results[ds_name] = results

        # Compute metrics
        metrics = compute_comparison_metrics(results)
        all_metrics[ds_name] = metrics
        print(f"\n  Comparison Metrics:")
        print(metrics.to_string(index=False))

        ari_matrix = compute_pairwise_ari(results)
        all_ari[ds_name] = ari_matrix
        print(f"\n  Pairwise ARI:")
        print(ari_matrix.round(3).to_string())

        # Save CSVs
        metrics.to_csv(COMPARISON_DATA_DIR / f"comparison_metrics_{ds_name}.csv", index=False)

        ari_matrix.to_csv(COMPARISON_DATA_DIR / f"comparison_pairwise_ari_{ds_name}.csv")

        # Entity x method labels
        labels_df = pd.DataFrame({"entity": raw_feats.index})
        for key, label in zip(METHOD_KEYS, METHOD_LABELS):
            labels_df[label] = results[key]["labels"]
        labels_df.to_csv(COMPARISON_DATA_DIR / f"comparison_labels_{ds_name}.csv", index=False)

        # ── Generate visualizations ───────────────────────────────────────
        print(f"\n  Generating comparison visualizations for {ds_name}...")
        plot_silhouette_comparison(results, ds_name)
        print(f"    Saved comparison_silhouette_{ds_name}")

        plot_optimal_k_comparison(results, ds_name)
        print(f"    Saved comparison_optimal_k_{ds_name}")

        plot_pairwise_ari_heatmap(ari_matrix, ds_name)
        print(f"    Saved comparison_ari_heatmap_{ds_name}")

        plot_stability_comparison(results, ds_name)
        print(f"    Saved comparison_stability_{ds_name}")

        plot_biplot_gallery(results, raw_feats, ds_name)
        print(f"    Saved comparison_biplot_gallery_{ds_name}")

        plot_comparison_summary_table(metrics, ds_name)
        print(f"    Saved comparison_summary_{ds_name}")

        plot_membership_matrix(results, list(raw_feats.index), ds_name)
        print(f"    Saved comparison_membership_{ds_name}")

    # ── Step 4: Summary report ────────────────────────────────────────────
    print("\n\n▸ Generating summary report...")
    report = generate_summary_report(all_results, all_metrics, all_ari)
    print(report)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\n  Figures → {COMPARISON_FIGURES_DIR}")
    print(f"  Data    → {COMPARISON_DATA_DIR}")


if __name__ == "__main__":
    main()
