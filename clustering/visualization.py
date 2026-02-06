# visualization.py
"""
Publication-quality figures for clustering analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from config import FIGURES_DIR, FIGURE_DPI, FIGURE_FORMAT, FIGSIZE_STANDARD, FIGSIZE_WIDE


def _save(fig, stem):
    """Save figure in all configured formats."""
    for fmt in FIGURE_FORMAT:
        fig.savefig(FIGURES_DIR / f"{stem}.{fmt}", dpi=FIGURE_DPI, bbox_inches="tight")


# ── Elbow / silhouette sweep ──────────────────────────────────────────────────

def plot_elbow(k_range, wcss, sil_scores, name):
    """Elbow plot (WCSS) and silhouette score vs k."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    ax1.plot(k_range, wcss, "o-", color="#2c7bb6")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Within-cluster sum of squares")
    ax1.set_title(f"Elbow method — {name}")
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_range, sil_scores, "o-", color="#d7191c")
    ax2.axhline(0.5, color="gray", ls="--", lw=0.8, label="0.5 threshold")
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Mean silhouette score")
    ax2.set_title(f"Silhouette analysis — {name}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, f"elbow_{name}")
    plt.close(fig)
    print(f"  Saved elbow_{name}")


# ── PCA biplot ────────────────────────────────────────────────────────────────

def plot_pca_biplot(X, labels, feature_names, name, entity_names=None,
                    cluster_names=None, raw_features=None,
                    show_entity_names=True):
    """
    2-D PCA projection with cluster coloring and loading vectors.

    Parameters
    ----------
    X : array-like
        Standardised feature matrix.
    labels : array-like
        Cluster labels.
    feature_names : list[str]
        Names of features (for loading vectors).
    name : str
        Dataset name (used in title and filename).
    entity_names : list[str], optional
        Per-point labels (scenario or material names).
    cluster_names : dict, optional
        {cluster_id: "descriptive name"} for legend labels.
    raw_features : DataFrame, optional
        Raw (un-standardised) features for median table.
    show_entity_names : bool
        If False, suppress individual point labels.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    loadings = pca.components_.T  # (n_features, 2)

    # Use taller figure if we have a median table (plot on top, table below)
    if raw_features is not None:
        fig, (ax, ax_table) = plt.subplots(
            2, 1, figsize=(14, 14),
            gridspec_kw={"height_ratios": [3, 1.2]},
        )
    else:
        fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
        ax_table = None

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10", len(unique_labels))

    for i, cl in enumerate(unique_labels):
        mask = labels == cl
        cl_label = (
            f"{cluster_names[cl]} (n={mask.sum()})"
            if cluster_names and cl in cluster_names
            else f"Cluster {cl} (n={mask.sum()})"
        )
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=[cmap(i)], label=cl_label,
            alpha=0.7, edgecolors="k", linewidth=0.4, s=60,
        )
        # Centroid
        cx, cy = X_pca[mask].mean(axis=0)
        ax.scatter(cx, cy, c=[cmap(i)], marker="X", s=200, edgecolors="k", linewidth=1.5)

    # Label points if names provided and enabled
    if show_entity_names and entity_names is not None:
        for idx, txt in enumerate(entity_names):
            ax.annotate(
                txt, (X_pca[idx, 0], X_pca[idx, 1]),
                fontsize=6, alpha=0.6,
                textcoords="offset points", xytext=(4, 4),
            )

    # Loading vectors
    scale = max(np.abs(X_pca).max(axis=0)) * 0.8
    for j, feat in enumerate(feature_names):
        ax.annotate(
            "", xy=(loadings[j, 0] * scale, loadings[j, 1] * scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
        )
        ax.text(
            loadings[j, 0] * scale * 1.08,
            loadings[j, 1] * scale * 1.08,
            feat, fontsize=7, color="gray", ha="center",
        )

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} variance)")
    ax.set_title(f"PCA biplot — {name}")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.2)

    # Median table — only show the features that survived VIF pruning
    if ax_table is not None and raw_features is not None:
        ax_table.axis("off")
        rf = raw_features.copy()
        rf["cluster"] = labels
        # Restrict to features used in clustering
        surviving = [f for f in feature_names if f in rf.columns]
        medians = rf.groupby("cluster")[surviving].median()

        # Short display names for readability
        col_short = {
            "total_cumulative_demand": "Cumul.\ndemand",
            "peak_demand": "Peak\ndemand",
            "mean_demand_early": "Early\ndemand",
            "year_of_peak": "Peak\nyear",
            "demand_slope": "Demand\nslope",
            "temporal_concentration": "Temporal\nconc.",
            "mean_cv": "Mean\nCV",
            "mean_ci_width": "CI\nwidth",
            "solar_fraction_2035": "Solar\nfrac.",
            "wind_fraction_2035": "Wind\nfrac.",
            "storage_fraction_2035": "Storage\nfrac.",
            "n_active_materials": "Active\nmats.",
        }
        col_headers = [col_short.get(c, c) for c in medians.columns]

        # Build display names for rows
        row_labels = []
        for cl in medians.index:
            if cluster_names and cl in cluster_names:
                row_labels.append(cluster_names[cl])
            else:
                row_labels.append(f"Cluster {cl}")

        # Columns that should display as plain integers (e.g., years)
        int_cols = {"year_of_peak", "n_active_materials"}

        # Format values
        def _fmt(v, col_name):
            if col_name in int_cols:
                return f"{int(v)}"
            if abs(v) >= 1e6:
                return f"{v/1e6:.1f}M"
            elif abs(v) >= 1e3:
                return f"{v/1e3:.1f}k"
            else:
                return f"{v:.2f}"

        raw_cols = list(medians.columns)
        cell_text = [[_fmt(medians.iloc[r, c], raw_cols[c]) for c in range(len(col_headers))]
                     for r in range(len(medians))]

        # Color rows by cluster
        row_colors = [cmap(i) for i in range(len(unique_labels))]

        table = ax_table.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_headers,
            rowColours=row_colors,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.6)

        # Style header and row label cells
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_text_props(fontweight="bold", fontsize=6.5)
                cell.set_height(0.12)
            if c == -1:
                cell.set_text_props(fontweight="bold", fontsize=7)

        ax_table.set_title("Cluster medians (raw features)", fontsize=10,
                           fontweight="bold", pad=10)

    fig.tight_layout()
    _save(fig, f"pca_biplot_{name}")
    plt.close(fig)
    print(f"  Saved pca_biplot_{name}")


def plot_pca_biplot_centroid_labels(X, labels, feature_names, name, entity_names=None,
                                     cluster_names=None, raw_features=None):
    """
    2-D PCA projection with cluster coloring, centroid labels only (no individual point labels).

    This version is designed for materials clustering where individual labels would be cluttered.
    Instead, interpretable labels are placed at cluster centroids.

    Parameters
    ----------
    X : array-like
        Standardised feature matrix.
    labels : array-like
        Cluster labels.
    feature_names : list[str]
        Names of features (for loading vectors).
    name : str
        Dataset name (used in title and filename).
    entity_names : list[str], optional
        Per-point names (for tooltip-style reference, not displayed).
    cluster_names : dict, optional
        {cluster_id: "descriptive name"} for centroid labels.
    raw_features : DataFrame, optional
        Raw (un-standardised) features for median table.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    loadings = pca.components_.T  # (n_features, 2)

    # Use taller figure if we have a median table
    if raw_features is not None:
        fig, (ax, ax_table) = plt.subplots(
            2, 1, figsize=(14, 14),
            gridspec_kw={"height_ratios": [3, 1.2]},
        )
    else:
        fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
        ax_table = None

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10", len(unique_labels))

    # Store centroids for labeling
    centroids = {}

    for i, cl in enumerate(unique_labels):
        mask = labels == cl
        cl_label = (
            f"{cluster_names[cl]} (n={mask.sum()})"
            if cluster_names and cl in cluster_names
            else f"Cluster {cl} (n={mask.sum()})"
        )
        # Plot individual points (no labels)
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=[cmap(i)], label=cl_label,
            alpha=0.7, edgecolors="k", linewidth=0.4, s=60,
        )
        # Compute and plot centroid
        cx, cy = X_pca[mask].mean(axis=0)
        centroids[cl] = (cx, cy, cmap(i))
        ax.scatter(cx, cy, c=[cmap(i)], marker="X", s=250, edgecolors="k", linewidth=2, zorder=10)

    # Add interpretable labels at centroids
    for cl, (cx, cy, color) in centroids.items():
        if cluster_names and cl in cluster_names:
            label_text = cluster_names[cl]
        else:
            label_text = f"Cluster {cl}"

        # Add label with white background for readability
        ax.annotate(
            label_text,
            (cx, cy),
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="bottom",
            textcoords="offset points",
            xytext=(0, 12),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.9),
        )

    # Loading vectors
    scale = max(np.abs(X_pca).max(axis=0)) * 0.8
    for j, feat in enumerate(feature_names):
        ax.annotate(
            "", xy=(loadings[j, 0] * scale, loadings[j, 1] * scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
        )
        ax.text(
            loadings[j, 0] * scale * 1.08,
            loadings[j, 1] * scale * 1.08,
            feat, fontsize=7, color="gray", ha="center",
        )

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} variance)")
    ax.set_title(f"PCA biplot — {name}")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.2)

    # Median table — only show the features that survived VIF pruning
    if ax_table is not None and raw_features is not None:
        ax_table.axis("off")
        rf = raw_features.copy()
        rf["cluster"] = labels
        # Restrict to features used in clustering
        surviving = [f for f in feature_names if f in rf.columns]
        medians = rf.groupby("cluster")[surviving].median()

        # Short display names for readability
        col_short = {
            "mean_demand": "Mean\ndemand",
            "peak_demand": "Peak\ndemand",
            "demand_growth": "Demand\ngrowth",
            "domestic_production": "Dom.\nprod.",
            "import_reliance": "Import\nreliance",
            "mean_capacity_ratio": "Mean\ncap.ratio",
            "max_capacity_ratio": "Max\ncap.ratio",
            "domestic_reserves_years": "Dom.\nreserves",
            "global_reserves_years": "Global\nreserves",
            "recycling_rate": "Recycling\nrate",
            "n_scenarios_active": "Active\nscenarios",
        }
        col_headers = [col_short.get(c, c) for c in medians.columns]

        # Build display names for rows
        row_labels = []
        for cl in medians.index:
            if cluster_names and cl in cluster_names:
                row_labels.append(cluster_names[cl])
            else:
                row_labels.append(f"Cluster {cl}")

        # Format values
        def _fmt(v, col_name):
            if pd.isna(v):
                return "—"
            if abs(v) >= 1e6:
                return f"{v/1e6:.1f}M"
            elif abs(v) >= 1e3:
                return f"{v/1e3:.1f}k"
            elif abs(v) < 0.01 and v != 0:
                return f"{v:.2e}"
            else:
                return f"{v:.2f}"

        raw_cols = list(medians.columns)
        cell_text = [[_fmt(medians.iloc[r, c], raw_cols[c]) for c in range(len(col_headers))]
                     for r in range(len(medians))]

        # Color rows by cluster
        row_colors = [cmap(i) for i in range(len(unique_labels))]

        table = ax_table.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_headers,
            rowColours=row_colors,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.6)

        # Style header and row label cells
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_text_props(fontweight="bold", fontsize=6.5)
                cell.set_height(0.12)
            if c == -1:
                cell.set_text_props(fontweight="bold", fontsize=7)

        ax_table.set_title("Cluster medians (raw features)", fontsize=10,
                           fontweight="bold", pad=10)

    fig.tight_layout()
    _save(fig, f"pca_biplot_{name}_centroid_labels")
    plt.close(fig)
    print(f"  Saved pca_biplot_{name}_centroid_labels")


# ── Silhouette plot ───────────────────────────────────────────────────────────

def plot_silhouette(X, labels, name):
    """Detailed per-sample silhouette plot."""
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    sil_avg = silhouette_score(X, labels)
    sample_sil = silhouette_samples(X, labels)

    y_lower = 10
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10", len(unique_labels))

    for i, cl in enumerate(unique_labels):
        vals = np.sort(sample_sil[labels == cl])
        y_upper = y_lower + len(vals)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper), 0, vals,
            facecolor=cmap(i), alpha=0.7, label=f"Cluster {cl}",
        )
        ax.text(-0.05, y_lower + 0.5 * len(vals), str(cl), fontsize=9)
        y_lower = y_upper + 10

    ax.axvline(sil_avg, color="red", ls="--", lw=1.2, label=f"Mean ({sil_avg:.3f})")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Sample index (sorted within cluster)")
    ax.set_title(f"Silhouette analysis — {name}")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    _save(fig, f"silhouette_{name}")
    plt.close(fig)
    print(f"  Saved silhouette_{name}")


# ── Cluster profile heatmap ──────────────────────────────────────────────────

def plot_cluster_profiles(profiles, name):
    """
    Heatmap of standardised cluster centroids.

    Parameters
    ----------
    profiles : DataFrame
        Rows = clusters, columns = features (raw or standardised means).
    """
    # Standardise across clusters for colour comparability
    # Drop zero-variance columns and replace inf/NaN to avoid sklearn errors
    profiles = profiles.replace([np.inf, -np.inf], np.nan).fillna(0)
    keep = profiles.std() > 0
    profiles_clean = profiles.loc[:, keep]
    if profiles_clean.empty:
        print(f"  Skipping cluster_profiles_{name} — no variable features")
        return
    from sklearn.preprocessing import StandardScaler
    arr = StandardScaler().fit_transform(profiles_clean)
    df_std = pd.DataFrame(arr, index=profiles_clean.index, columns=profiles_clean.columns)

    fig, ax = plt.subplots(figsize=(max(8, len(profiles.columns) * 0.7), 6))
    sns.heatmap(
        df_std, annot=True, fmt=".2f", cmap="RdYlBu_r",
        linewidths=0.5, ax=ax, cbar_kws={"label": "z-score"},
    )
    ax.set_title(f"Cluster profiles — {name}")
    ax.set_ylabel("Cluster")

    fig.tight_layout()
    _save(fig, f"cluster_profiles_{name}")
    plt.close(fig)
    print(f"  Saved cluster_profiles_{name}")


# ── Stress matrix heatmap ────────────────────────────────────────────────────

def plot_stress_matrix(stress_matrix, name="stress"):
    """
    Heatmap: scenario-cluster rows × material-cluster columns.
    Values = mean capacity-ratio stress.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    sns.heatmap(
        stress_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Mean capacity-stress ratio"},
        xticklabels=[f"Mat-{i}" for i in range(stress_matrix.shape[1])],
        yticklabels=[f"Scen-{i}" for i in range(stress_matrix.shape[0])],
    )
    ax.set_title("Scenario × Material cluster stress matrix")
    ax.set_xlabel("Material cluster")
    ax.set_ylabel("Scenario cluster")

    fig.tight_layout()
    _save(fig, f"stress_matrix_{name}")
    plt.close(fig)
    print(f"  Saved stress_matrix_{name}")


# ── Feature sensitivity bar chart ─────────────────────────────────────────────

def plot_feature_sensitivity(sensitivity_df, name):
    """Bar chart of ARI when each feature is dropped."""
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    sens = sensitivity_df.sort_values("ARI_vs_full")
    ax.barh(sens["feature_dropped"], sens["ARI_vs_full"], color="#4292c6")
    ax.axvline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("ARI vs. full model")
    ax.set_title(f"Feature sensitivity — {name}")
    ax.grid(True, alpha=0.2, axis="x")

    fig.tight_layout()
    _save(fig, f"feature_sensitivity_{name}")
    plt.close(fig)
    print(f"  Saved feature_sensitivity_{name}")


# ── Demand spaghetti plot by cluster ─────────────────────────────────────────

def plot_demand_spaghetti(demand, labels, index, name,
                          cluster_names=None):
    """
    Spaghetti plot of total demand over time, one line per scenario (or
    material), color-coded by cluster assignment.

    Parameters
    ----------
    demand : DataFrame
        Raw demand data with columns: scenario, year, material, mean.
    labels : array-like
        Cluster labels aligned with *index*.
    index : array-like
        Entity names (scenario names or material names) matching labels.
    name : str
        "scenarios" or "materials" — controls grouping logic and filename.
    cluster_names : dict, optional
        {cluster_id: "descriptive name"}.
    """
    label_map = pd.Series(labels, index=index, name="cluster")
    unique_labels = np.sort(np.unique(labels))
    cmap = plt.get_cmap("tab10", len(unique_labels))

    # Compute total demand per entity per year
    if name == "scenarios":
        group_col = "scenario"
    else:
        group_col = "material"

    totals = (
        demand.groupby([group_col, "year"])["mean"]
        .sum()
        .reset_index()
    )
    totals["cluster"] = totals[group_col].map(label_map)
    totals = totals.dropna(subset=["cluster"])
    totals["cluster"] = totals["cluster"].astype(int)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Plot individual lines (thin, semi-transparent)
    for entity in index:
        cl = label_map.get(entity)
        if cl is None:
            continue
        ent_data = totals[totals[group_col] == entity].sort_values("year")
        ax.plot(
            ent_data["year"], ent_data["mean"],
            color=cmap(int(cl)), alpha=0.35, linewidth=0.8,
        )

    # Plot cluster medians (thick)
    for cl in unique_labels:
        cl_data = totals[totals["cluster"] == cl]
        median_line = cl_data.groupby("year")["mean"].median().sort_index()
        cl_label = (
            cluster_names[cl] if cluster_names and cl in cluster_names
            else f"Cluster {cl}"
        )
        ax.plot(
            median_line.index, median_line.values,
            color=cmap(int(cl)), linewidth=2.5, label=f"{cl_label} (n={int((labels == cl).sum())})",
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Total demand (tonnes)")
    ax.set_title(f"Total demand trajectories by cluster — {name}")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.2)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))

    fig.tight_layout()
    _save(fig, f"demand_spaghetti_{name}")
    plt.close(fig)
    print(f"  Saved demand_spaghetti_{name}")


if __name__ == "__main__":
    print("Visualization module loaded. Run via main_analysis.py.")
