# main_analysis.py
"""
Main script: run the complete clustering analysis pipeline.

Clusters scenarios and materials using pre-computed Sparse PCA scores
(from sparse_nmf_analysis.py) as input to K-means.

Usage:
    cd clustering/
    python sparse_nmf_analysis.py   # must run first to generate SPCA scores
    python main_analysis.py
"""

import pandas as pd
import numpy as np

from config import RESULTS_DIR, FIGURES_KMEANS_DIR
from feature_engineering import (
    load_demand_data, load_nrel_data, load_risk_data, load_usgs_2023_thin_film,
    engineer_scenario_features, engineer_material_features,
)
from clustering import ClusterAnalyzer
from visualization import (
    plot_elbow, plot_pca_biplot, plot_pca_biplot_centroid_labels, plot_silhouette,
    plot_cluster_profiles, plot_stress_matrix, plot_feature_sensitivity,
    plot_demand_spaghetti,
)


def main():
    print("=" * 80)
    print("CLUSTERING ANALYSIS — ENERGY TRANSITION SCENARIOS & MATERIALS")
    print("  (Using Sparse PCA scores as clustering input)")
    print("=" * 80)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print("\n▸ Step 1: Loading data...")
    demand = load_demand_data()
    nrel = load_nrel_data()
    risk_data = load_risk_data()
    thin_film = load_usgs_2023_thin_film()
    print(f"  demand: {demand.shape[0]:,} rows  "
          f"({demand['scenario'].nunique()} scenarios, "
          f"{demand['material'].nunique()} materials)")
    print(f"  risk data: {'loaded' if risk_data else 'NOT FOUND'}")
    print(f"  thin-film USGS 2023: {len(thin_film)} materials")

    # ── Step 2: Feature engineering ───────────────────────────────────────
    print("\n▸ Step 2: Feature engineering...")

    scenario_feats = engineer_scenario_features(demand, nrel, risk_data, thin_film)
    print(f"  Scenario features: {scenario_feats.shape}")

    material_feats = engineer_material_features(demand, risk_data, thin_film)
    print(f"  Material features: {material_feats.shape}")

    # Save raw features (needed for cluster profile interpretation)
    scenario_feats.to_csv(RESULTS_DIR / "scenario_features_raw.csv")
    material_feats.to_csv(RESULTS_DIR / "material_features_raw.csv")

    # ── Step 3: Load Sparse PCA scores ────────────────────────────────────
    print("\n▸ Step 3: Loading Sparse PCA scores...")

    spca_mat_path = RESULTS_DIR / "spca_scores_materials.csv"
    spca_scen_path = RESULTS_DIR / "spca_scores_scenarios.csv"

    if not spca_mat_path.exists() or not spca_scen_path.exists():
        raise FileNotFoundError(
            "SPCA score files not found. Run sparse_nmf_analysis.py first.\n"
            f"  Expected: {spca_mat_path}\n"
            f"  Expected: {spca_scen_path}"
        )

    X_mat = pd.read_csv(spca_mat_path, index_col=0)
    X_scen = pd.read_csv(spca_scen_path, index_col=0)

    # Apply interpretable labels (derived from SPCA loadings inspection)
    SCEN_COMPONENT_LABELS = {
        "SPC1": "Demand Scale & Wind",
        "SPC2": "Demand Uncertainty",
        "SPC3": "Supply Chain Stress",
        "SPC4": "Solar & Storage Mix",
    }
    MAT_COMPONENT_LABELS = {
        "SPC1": "Demand Scale",
        "SPC2": "Geopolitical Risk",
        "SPC3": "Reserve Geography",
        "SPC4": "Import Concentration",
        "SPC5": "Capacity Stress",
    }
    X_scen.columns = [SCEN_COMPONENT_LABELS.get(c, c) for c in X_scen.columns]
    X_mat.columns = [MAT_COMPONENT_LABELS.get(c, c) for c in X_mat.columns]

    print(f"  Material SPCA scores: {X_mat.shape}  ({list(X_mat.columns)})")
    print(f"  Scenario SPCA scores: {X_scen.shape}  ({list(X_scen.columns)})")

    # ── Step 4: Scenario clustering ───────────────────────────────────────
    print("\n▸ Step 4: Clustering scenarios (on SPCA scores)...")

    analyzer_scen = ClusterAnalyzer(X_scen, name="scenarios")
    k_range_s, wcss_s, sil_s = analyzer_scen.find_optimal_k()
    plot_elbow(k_range_s, wcss_s, sil_s, "scenarios")

    best_k_scen = k_range_s[int(np.argmax(sil_s))]
    print(f"\n  → Silhouette-optimal k={best_k_scen}")

    # Cap scenario k at 7 for interpretability (61 scenarios / 7 ≈ 9 per cluster).
    if best_k_scen > 7:
        sil_arr = np.array(sil_s)
        candidates = [(k_range_s[i], sil_arr[i])
                      for i in range(len(sil_arr)) if 2 <= k_range_s[i] <= 7]
        if candidates:
            best_k_scen = max(candidates, key=lambda x: x[1])[0]
            best_sil = dict(candidates)[best_k_scen]
            print(f"  → Capping to k={best_k_scen} (silhouette={best_sil:.3f})"
                  f" for interpretability")

    scen_labels = analyzer_scen.fit_final_model(best_k_scen)
    analyzer_scen.validate_stability()
    sens_scen = analyzer_scen.feature_sensitivity()

    scen_cluster_names = {
        0: "Volatile / peak-and-decline",
        1: "Aggressive decarbonization",
        2: "Moderate / sustained growth",
    }

    plot_pca_biplot_centroid_labels(
        X_scen.values, scen_labels, list(X_scen.columns),
        "scenarios", entity_names=list(X_scen.index),
        cluster_names=scen_cluster_names,
        raw_features=scenario_feats,
    )
    plot_silhouette(X_scen.values, scen_labels, "scenarios")
    plot_feature_sensitivity(sens_scen, "scenarios")

    plot_demand_spaghetti(
        demand, scen_labels, X_scen.index, "scenarios",
        cluster_names=scen_cluster_names,
    )

    # ── Step 5: Material clustering ───────────────────────────────────────
    print("\n▸ Step 5: Clustering materials (on SPCA scores)...")
    analyzer_mat = ClusterAnalyzer(X_mat, name="materials")
    k_range_m, wcss_m, sil_m = analyzer_mat.find_optimal_k()
    plot_elbow(k_range_m, wcss_m, sil_m, "materials")

    best_k_mat = k_range_m[int(np.argmax(sil_m))]
    print(f"\n  → Silhouette-optimal k={best_k_mat} for materials")

    # k=2 is a trivial volume split for materials (Steel/Cement vs rest).
    # Select the best k in 3-7 range for interpretable groupings.
    if best_k_mat == 2:
        sil_arr = np.array(sil_m)
        candidates = [(k_range_m[i], sil_arr[i])
                      for i in range(len(sil_arr)) if 3 <= k_range_m[i] <= 7]
        if candidates:
            best_k_mat = max(candidates, key=lambda x: x[1])[0]
            best_sil = dict(candidates)[best_k_mat]
            print(f"  → Overriding to k={best_k_mat} (silhouette={best_sil:.3f})"
                  f" for interpretability")

    mat_labels = analyzer_mat.fit_final_model(best_k_mat)
    analyzer_mat.validate_stability()
    sens_mat = analyzer_mat.feature_sensitivity()

    mat_cluster_names = {
        0: "Base metals (OECD-sourced)",
        1: "Bulk industrial",
        2: "Import-dependent specialty",
        3: "Low-volume, concentrated-source",
        4: "Neodymium (REE outlier)",
        5: "REE permanent magnet elements",
    }

    # Standard biplot with individual material labels
    plot_pca_biplot(
        X_mat.values, mat_labels, list(X_mat.columns),
        "materials", entity_names=list(X_mat.index),
        cluster_names=mat_cluster_names,
        raw_features=material_feats,
    )

    # Clean biplot with only centroid labels (no individual material names)
    plot_pca_biplot_centroid_labels(
        X_mat.values, mat_labels, list(X_mat.columns),
        "materials", entity_names=list(X_mat.index),
        cluster_names=mat_cluster_names,
        raw_features=material_feats,
    )

    plot_silhouette(X_mat.values, mat_labels, "materials")
    plot_feature_sensitivity(sens_mat, "materials")

    # ── Step 6: Cluster profiles ──────────────────────────────────────────
    print("\n▸ Step 6: Cluster interpretation...")

    scen_profiles = analyzer_scen.get_cluster_profiles(scenario_feats)
    mat_profiles = analyzer_mat.get_cluster_profiles(material_feats)

    print("\nScenario cluster profiles (raw means):")
    print(scen_profiles.round(2).to_string())

    print("\nMaterial cluster profiles (raw means):")
    print(mat_profiles.round(2).to_string())

    plot_cluster_profiles(scen_profiles, "scenarios")
    plot_cluster_profiles(mat_profiles, "materials")

    # ── Step 7: Stress matrix ─────────────────────────────────────────────
    print("\n▸ Step 7: Scenario × Material stress matrix...")

    # Build stress matrix: for each (scen_cluster, mat_cluster), average
    # the mean demand across constituent scenarios and materials.
    demand_merged = demand.copy()
    scen_map = pd.Series(scen_labels, index=X_scen.index, name="scen_cluster")
    mat_map = pd.Series(mat_labels, index=X_mat.index, name="mat_cluster")
    demand_merged = demand_merged.merge(
        scen_map.reset_index().rename(columns={"index": "scenario"}),
        on="scenario", how="left",
    )
    demand_merged = demand_merged.merge(
        mat_map.reset_index().rename(columns={"index": "material"}),
        on="material", how="left",
    )
    stress = (
        demand_merged
        .groupby(["scen_cluster", "mat_cluster"])["mean"]
        .sum()
        .unstack(fill_value=0)
    )
    plot_stress_matrix(stress.values, "demand")

    # ── Step 8: Export ────────────────────────────────────────────────────
    print("\n▸ Step 8: Exporting results...")

    scen_results = pd.DataFrame({
        "scenario": X_scen.index,
        "cluster": scen_labels,
        "silhouette": analyzer_scen.silhouette_per_sample,
    })
    scen_results.to_csv(RESULTS_DIR / "scenario_clusters.csv", index=False)

    mat_results = pd.DataFrame({
        "material": X_mat.index,
        "cluster": mat_labels,
        "silhouette": analyzer_mat.silhouette_per_sample,
    })
    mat_results.to_csv(RESULTS_DIR / "material_clusters.csv", index=False)

    scen_profiles.to_csv(RESULTS_DIR / "scenario_cluster_profiles.csv")
    mat_profiles.to_csv(RESULTS_DIR / "material_cluster_profiles.csv")
    stress.to_csv(RESULTS_DIR / "stress_matrix.csv")

    # Validation summary
    scen_sil = analyzer_scen.results.get('silhouette_scores', [0])
    mat_sil = analyzer_mat.results.get('silhouette_scores', [0])
    with open(RESULTS_DIR / "validation_metrics.txt", "w") as f:
        f.write(f"Scenario clustering (Sparse PCA-based): k={best_k_scen}\n")
        f.write(f"  Silhouette: {scen_sil[best_k_scen - 2]:.3f}\n")
        f.write(f"  Stability ARI: {analyzer_scen.results.get('stability_ari_mean', 0):.3f}\n")
        f.write(f"  Input features: {list(X_scen.columns)}\n")
        f.write(f"  Method: K-means on {len(X_scen.columns)} Sparse PCA components (alpha=1.0)\n\n")
        f.write(f"Material clustering (Sparse PCA-based): k={best_k_mat}\n")
        f.write(f"  Silhouette: {mat_sil[best_k_mat - 2]:.3f}\n")
        f.write(f"  Stability ARI: {analyzer_mat.results.get('stability_ari_mean', 0):.3f}\n")
        f.write(f"  Input features: {list(X_mat.columns)}\n")
        f.write(f"  Method: K-means on {len(X_mat.columns)} Sparse PCA components (alpha=2.0)\n")

    print(f"\n  Results → {RESULTS_DIR}")
    print(f"  Figures → {FIGURES_KMEANS_DIR}")
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
