# sparse_pca_story.py
"""
Sparse PCA Story Visualizations

Creates interpretable visualizations that tell a meaningful story about
materials and scenarios using Sparse PCA components.

Usage:
    python sparse_pca_story.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from config import (
    FIGURES_SPCA_STORY_DIR, RESULTS_DIR, FIGURE_DPI, FIGURE_FORMAT,
    FIGSIZE_STANDARD, FIGSIZE_WIDE
)


def _save(fig, stem):
    """Save figure in all configured formats."""
    for fmt in FIGURE_FORMAT:
        fig.savefig(FIGURES_SPCA_STORY_DIR / f"{stem}.{fmt}", dpi=FIGURE_DPI, bbox_inches="tight")


# ============================================================================
# COMPONENT INTERPRETATION
# ============================================================================

# Human-readable names for Sparse PCA components based on loadings analysis
MATERIAL_COMPONENT_NAMES = {
    'SPC1': 'Demand Scale',
    'SPC2': 'Geopolitical Risk',
    'SPC3': 'Reserve Geography',
    'SPC4': 'Import Concentration',
    'SPC5': 'Capacity Stress'
}

MATERIAL_COMPONENT_DESCRIPTIONS = {
    'SPC1': 'High demand magnitude\n(peak, mean, volatility)',
    'SPC2': 'Supply from risky regions\nvs. OECD sources',
    'SPC3': 'Reserves in China/\nhigh-risk countries',
    'SPC4': 'Concentrated imports\nfrom few suppliers',
    'SPC5': 'Demand vs. production\ncapacity ratio'
}

SCENARIO_COMPONENT_NAMES = {
    'SPC1': 'Demand Scale & Wind',
    'SPC2': 'Demand Uncertainty',
    'SPC3': 'Supply Chain Stress',
    'SPC4': 'Solar & Storage Mix',
}

SCENARIO_COMPONENT_DESCRIPTIONS = {
    'SPC1': 'Peak demand, early build-out,\nwind deployment fraction',
    'SPC2': 'High variability\nacross simulations',
    'SPC3': 'Production capacity\nexceedance risk',
    'SPC4': 'Solar + storage\ndeployment share',
}


def load_data():
    """Load features and loadings."""
    # Raw features
    scenario_feats = pd.read_csv(RESULTS_DIR / "scenario_features_raw.csv", index_col=0)
    material_feats = pd.read_csv(RESULTS_DIR / "material_features_raw.csv", index_col=0)

    # Sparse PCA loadings
    spca_loadings_mat = pd.read_csv(RESULTS_DIR / "sparse_pca_loadings_materials.csv", index_col=0)
    spca_loadings_scen = pd.read_csv(RESULTS_DIR / "sparse_pca_loadings_scenarios.csv", index_col=0)

    return scenario_feats, material_feats, spca_loadings_scen, spca_loadings_mat


def preprocess_and_transform(df, loadings):
    """Preprocess features and transform using Sparse PCA loadings."""
    # Clean data
    df_clean = df.copy()
    df_clean = df_clean.loc[:, df_clean.std() > 0]
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(df_clean.median())

    # Keep only features that exist in loadings
    common_features = [f for f in loadings.index if f in df_clean.columns]
    df_clean = df_clean[common_features]
    loadings = loadings.loc[common_features]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    # Transform using loadings (project onto components)
    X_transformed = X_scaled @ loadings.values

    scores = pd.DataFrame(
        X_transformed,
        index=df.index,
        columns=loadings.columns
    )

    return scores, loadings


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_component_interpretation(loadings, component_names, component_descriptions, name):
    """
    Show what each Sparse PCA component captures.
    """
    n_components = len(loadings.columns)
    fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 8))

    for i, (ax, comp) in enumerate(zip(axes, loadings.columns)):
        # Get loadings for this component
        comp_loadings = loadings[comp].copy()

        # Filter to non-zero loadings
        nonzero = comp_loadings[comp_loadings.abs() > 1e-6].sort_values()

        if len(nonzero) == 0:
            ax.text(0.5, 0.5, 'No active\nfeatures', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{component_names.get(comp, comp)}', fontweight='bold')
            continue

        # Color by sign
        colors = ['#d73027' if v < 0 else '#1a9850' for v in nonzero.values]

        bars = ax.barh(range(len(nonzero)), nonzero.values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(nonzero)))
        ax.set_yticklabels(nonzero.index, fontsize=9)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Loading')
        ax.set_title(f'{component_names.get(comp, comp)}', fontsize=12, fontweight='bold')

        # Add description
        ax.text(0.5, -0.15, component_descriptions.get(comp, ''),
               ha='center', va='top', transform=ax.transAxes,
               fontsize=9, style='italic', color='gray')

    fig.suptitle(f'Sparse PCA Component Interpretation — {name.title()}',
                fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, f"spca_components_{name}")
    plt.close(fig)
    print(f"  Saved spca_components_{name}")


def plot_entity_profiles(scores, component_names, name, top_n=15):
    """
    Show how entities (materials/scenarios) score on each component.
    Heatmap format with interpretable component names.
    """
    # Rename columns to interpretable names
    scores_named = scores.copy()
    scores_named.columns = [component_names.get(c, c) for c in scores_named.columns]

    # Select subset if too many entities
    if len(scores_named) > top_n:
        # Select most extreme entities (highest absolute scores)
        extremeness = scores_named.abs().sum(axis=1)
        top_entities = extremeness.nlargest(top_n).index
        scores_named = scores_named.loc[top_entities]

    fig, ax = plt.subplots(figsize=(10, max(6, len(scores_named) * 0.4)))

    # Sort by first component
    scores_named = scores_named.sort_values(scores_named.columns[0], ascending=False)

    sns.heatmap(
        scores_named,
        annot=True,
        fmt='.1f',
        cmap='RdBu_r',
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Component Score (z-scaled)'}
    )

    ax.set_title(f'{name.title()} Profiles on Sparse PCA Components', fontsize=14, fontweight='bold')
    ax.set_xlabel('Component')
    ax.set_ylabel(name.title().rstrip('s'))

    fig.tight_layout()
    _save(fig, f"spca_profiles_{name}")
    plt.close(fig)
    print(f"  Saved spca_profiles_{name}")


def plot_biplot(scores, loadings, component_names, name, pc_x=0, pc_y=1):
    """
    Biplot showing entities and loading vectors in Sparse PCA space.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    comp_x = loadings.columns[pc_x]
    comp_y = loadings.columns[pc_y]

    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None

    # Plot entities
    ax.scatter(
        scores.iloc[:, pc_x],
        scores.iloc[:, pc_y],
        s=100,
        c='#3182bd',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

    # Label entities — only outliers for dense plots, adjustText for repulsion
    n_entities = len(scores)
    xs = scores.iloc[:, pc_x].values
    ys = scores.iloc[:, pc_y].values

    if n_entities > 25:
        x_med, y_med = np.median(xs), np.median(ys)
        dist = np.sqrt((xs - x_med) ** 2 + (ys - y_med) ** 2)
        q75 = np.percentile(dist, 75)
        iqr = q75 - np.percentile(dist, 25)
        threshold = q75 + 1.0 * iqr
        label_mask = dist > threshold
    else:
        label_mask = np.ones(n_entities, dtype=bool)

    texts = []
    for i, idx in enumerate(scores.index):
        if label_mask[i]:
            t = ax.text(xs[i], ys[i], idx, fontsize=7, alpha=0.8)
            texts.append(t)

    if adjust_text is not None and texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.4, lw=0.5))

    # Plot loading vectors (only non-zero) — label top 8 by magnitude
    max_loading_labels = 8
    scale = max(scores.iloc[:, pc_x].abs().max(), scores.iloc[:, pc_y].abs().max()) * 0.8

    # Compute magnitudes for non-zero loadings
    nonzero_feats = []
    for feat in loadings.index:
        lx = loadings.loc[feat, comp_x]
        ly = loadings.loc[feat, comp_y]
        if abs(lx) > 1e-6 or abs(ly) > 1e-6:
            nonzero_feats.append((feat, lx, ly, np.sqrt(lx**2 + ly**2)))

    if nonzero_feats:
        mags = [m for _, _, _, m in nonzero_feats]
        mag_threshold = sorted(mags)[-min(max_loading_labels, len(mags))]

        for feat, lx, ly, mag in nonzero_feats:
            is_top = mag >= mag_threshold
            ax.annotate(
                '',
                xy=(lx * scale, ly * scale),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#e31a1c', lw=1.5,
                                alpha=0.8 if is_top else 0.2)
            )
            if is_top:
                ax.text(
                    lx * scale * 1.1,
                    ly * scale * 1.1,
                    feat,
                    fontsize=9,
                    color='#e31a1c',
                    fontweight='bold',
                    ha='center'
                )

    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel(f'{component_names.get(comp_x, comp_x)}', fontsize=12)
    ax.set_ylabel(f'{component_names.get(comp_y, comp_y)}', fontsize=12)
    ax.set_title(f'Sparse PCA Biplot — {name.title()}', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.2)

    fig.tight_layout()
    _save(fig, f"spca_biplot_{name}")
    plt.close(fig)
    print(f"  Saved spca_biplot_{name}")


def plot_risk_quadrant_materials(scores, name='materials'):
    """
    Create a quadrant plot for materials showing supply risk vs demand pressure.
    Uses most interpretable components.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use SPC1 (Demand Scale) and SPC2 (Geopolitical Risk)
    x = scores['SPC1']  # Demand Scale
    y = scores['SPC2']  # Geopolitical Risk

    # Color by SPC5 (Capacity Stress) if available
    if 'SPC5' in scores.columns:
        colors = scores['SPC5']
        cmap = 'YlOrRd'
    else:
        colors = '#3182bd'
        cmap = None

    scatter = ax.scatter(x, y, s=150, c=colors, cmap=cmap, alpha=0.8,
                        edgecolors='black', linewidth=0.5)

    if cmap:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Capacity Stress', fontsize=10)

    # Label all materials
    for idx in scores.index:
        ax.annotate(
            idx,
            (x[idx], y[idx]),
            fontsize=9,
            textcoords='offset points',
            xytext=(5, 5),
            fontweight='bold' if abs(x[idx]) > 1 or abs(y[idx]) > 1 else 'normal'
        )

    # Add quadrant lines
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    # Add quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.text(xlim[1] * 0.7, ylim[1] * 0.85, 'HIGH RISK\nHigh demand + High geopolitical risk',
           ha='center', va='center', fontsize=10, color='#d73027',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.text(xlim[0] * 0.7, ylim[1] * 0.85, 'WATCH LIST\nLow demand but risky supply',
           ha='center', va='center', fontsize=10, color='#fc8d59',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.text(xlim[1] * 0.7, ylim[0] * 0.85, 'SECURE DEMAND\nHigh demand, stable supply',
           ha='center', va='center', fontsize=10, color='#91bfdb',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.text(xlim[0] * 0.7, ylim[0] * 0.85, 'LOW PRIORITY\nLow demand, stable supply',
           ha='center', va='center', fontsize=10, color='#4575b4',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Demand Scale →', fontsize=12, fontweight='bold')
    ax.set_ylabel('Geopolitical Supply Risk →', fontsize=12, fontweight='bold')
    ax.set_title('Material Risk-Demand Quadrant Analysis\n(Sparse PCA)', fontsize=14, fontweight='bold')

    fig.tight_layout()
    _save(fig, "spca_risk_quadrant_materials")
    plt.close(fig)
    print(f"  Saved spca_risk_quadrant_materials")


def plot_scenario_landscape(scores, name='scenarios'):
    """
    Create a landscape plot for scenarios showing uncertainty vs scale.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Use SPC1 (Demand Scale & Wind) and SPC2 (Demand Uncertainty)
    x = scores['SPC1']  # Demand Scale & Wind
    y = scores['SPC2']  # Demand Uncertainty

    # Color by SPC3 (Supply Chain Stress)
    colors = scores['SPC3']

    scatter = ax.scatter(x, y, s=150, c=colors, cmap='RdYlGn_r', alpha=0.8,
                        edgecolors='black', linewidth=0.5)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Supply Chain Stress', fontsize=10)

    # Label scenarios (abbreviated)
    for idx in scores.index:
        label = idx[:20] + '...' if len(idx) > 20 else idx
        ax.annotate(
            label,
            (x[idx], y[idx]),
            fontsize=7,
            textcoords='offset points',
            xytext=(5, 5),
            alpha=0.8
        )

    # Add quadrant lines
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    # Add quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.text(xlim[1] * 0.7, ylim[1] * 0.85, 'HIGH STAKES\nLarge scale + High uncertainty',
           ha='center', va='center', fontsize=10, color='#d73027',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.text(xlim[0] * 0.7, ylim[1] * 0.85, 'VOLATILE\nSmaller scale but uncertain',
           ha='center', va='center', fontsize=10, color='#fc8d59',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.text(xlim[1] * 0.7, ylim[0] * 0.85, 'PREDICTABLE SCALE\nLarge but consistent',
           ha='center', va='center', fontsize=10, color='#91bfdb',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.text(xlim[0] * 0.7, ylim[0] * 0.85, 'BASELINE\nSmaller, predictable',
           ha='center', va='center', fontsize=10, color='#4575b4',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Demand Scale & Wind →', fontsize=12, fontweight='bold')
    ax.set_ylabel('Demand Uncertainty →', fontsize=12, fontweight='bold')
    ax.set_title('Scenario Landscape Analysis\n(Sparse PCA)', fontsize=14, fontweight='bold')

    fig.tight_layout()
    _save(fig, "spca_landscape_scenarios")
    plt.close(fig)
    print(f"  Saved spca_landscape_scenarios")


def plot_material_summary_story(scores, loadings, component_names):
    """
    Create a single summary figure telling the material story.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # ── Panel 1: Risk Quadrant ──
    ax1 = fig.add_subplot(gs[0, 0:2])

    x = scores['SPC1']
    y = scores['SPC2']
    colors = scores['SPC5'] if 'SPC5' in scores.columns else '#3182bd'

    scatter = ax1.scatter(x, y, s=120, c=colors, cmap='YlOrRd', alpha=0.8,
                         edgecolors='black', linewidth=0.5)

    # Label top materials
    extremeness = scores.abs().sum(axis=1)
    top_materials = extremeness.nlargest(12).index
    for idx in top_materials:
        ax1.annotate(idx, (x[idx], y[idx]), fontsize=8, textcoords='offset points', xytext=(3, 3))

    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Demand Scale →')
    ax1.set_ylabel('Geopolitical Risk →')
    ax1.set_title('Material Risk-Demand Positioning', fontweight='bold')

    if isinstance(colors, pd.Series):
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Capacity Stress')

    # ── Panel 2: Component Legend ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')

    legend_text = """COMPONENT INTERPRETATION
─────────────────────────

SPC1: Demand Scale
  → High: Large absolute demand
  → Materials: Steel, Cement, Aluminum

SPC2: Geopolitical Risk
  → High: Sources from risky regions
  → Materials: Rare earths, Yttrium

SPC3: Reserve Geography
  → High: Reserves in China/risky areas
  → Materials: Neodymium, Dysprosium

SPC4: Import Concentration
  → High: Few dominant suppliers
  → Materials: Boron, Yttrium

SPC5: Capacity Stress
  → High: Demand near production limits
  → Materials: Neodymium, Dysprosium"""

    ax2.text(0, 0.95, legend_text, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f7f7f7', edgecolor='gray'))

    # ── Panel 3: Top materials by each component ──
    ax3 = fig.add_subplot(gs[1, 0])

    # Get top 3 materials for each component
    top_by_comp = {}
    for comp in scores.columns:
        top_by_comp[component_names.get(comp, comp)] = scores[comp].nlargest(3).index.tolist()

    y_pos = 0
    for comp_name, materials in top_by_comp.items():
        ax3.text(0, y_pos, f"{comp_name}:", fontweight='bold', fontsize=10)
        ax3.text(0.4, y_pos, ', '.join(materials), fontsize=9)
        y_pos -= 0.15

    ax3.set_xlim(0, 1)
    ax3.set_ylim(y_pos - 0.1, 0.1)
    ax3.axis('off')
    ax3.set_title('Highest Scoring Materials by Component', fontweight='bold')

    # ── Panel 4: Critical materials summary ──
    ax4 = fig.add_subplot(gs[1, 1:])

    # Define "critical" as high on multiple risk dimensions
    risk_score = scores['SPC2'] + scores.get('SPC3', 0) + scores.get('SPC4', 0)
    critical = risk_score.nlargest(8)

    bars = ax4.barh(range(len(critical)), critical.values, color='#d73027', alpha=0.8, edgecolor='black')
    ax4.set_yticks(range(len(critical)))
    ax4.set_yticklabels(critical.index)
    ax4.set_xlabel('Composite Risk Score (SPC2 + SPC3 + SPC4)')
    ax4.set_title('Materials with Highest Supply Chain Risk Exposure', fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)

    fig.suptitle('Material Supply Chain Risk Analysis — Sparse PCA Summary',
                fontsize=16, fontweight='bold', y=0.98)

    _save(fig, "spca_material_story")
    plt.close(fig)
    print(f"  Saved spca_material_story")


def main():
    """Generate all Sparse PCA story visualizations."""
    print("=" * 70)
    print("SPARSE PCA STORY VISUALIZATIONS")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    scenario_feats, material_feats, spca_loadings_scen, spca_loadings_mat = load_data()

    # Transform data
    print("\nTransforming data...")
    scores_mat, loadings_mat = preprocess_and_transform(material_feats, spca_loadings_mat)
    scores_scen, loadings_scen = preprocess_and_transform(scenario_feats, spca_loadings_scen)

    print(f"  Materials: {scores_mat.shape}")
    print(f"  Scenarios: {scores_scen.shape}")

    # Generate visualizations
    print("\n" + "=" * 40)
    print("MATERIALS")
    print("=" * 40)

    plot_component_interpretation(
        loadings_mat, MATERIAL_COMPONENT_NAMES, MATERIAL_COMPONENT_DESCRIPTIONS, "materials"
    )
    plot_entity_profiles(scores_mat, MATERIAL_COMPONENT_NAMES, "materials", top_n=20)
    plot_biplot(scores_mat, loadings_mat, MATERIAL_COMPONENT_NAMES, "materials")
    plot_risk_quadrant_materials(scores_mat)
    plot_material_summary_story(scores_mat, loadings_mat, MATERIAL_COMPONENT_NAMES)

    print("\n" + "=" * 40)
    print("SCENARIOS")
    print("=" * 40)

    plot_component_interpretation(
        loadings_scen, SCENARIO_COMPONENT_NAMES, SCENARIO_COMPONENT_DESCRIPTIONS, "scenarios"
    )
    plot_entity_profiles(scores_scen, SCENARIO_COMPONENT_NAMES, "scenarios", top_n=20)
    plot_biplot(scores_scen, loadings_scen, SCENARIO_COMPONENT_NAMES, "scenarios")
    plot_scenario_landscape(scores_scen)

    # Save scores for reference
    print("\nSaving transformed scores...")
    scores_mat.to_csv(RESULTS_DIR / "spca_scores_materials.csv")
    scores_scen.to_csv(RESULTS_DIR / "spca_scores_scenarios.csv")
    print(f"  Saved spca_scores_materials.csv")
    print(f"  Saved spca_scores_scenarios.csv")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {FIGURES_SPCA_STORY_DIR}")


if __name__ == "__main__":
    main()
