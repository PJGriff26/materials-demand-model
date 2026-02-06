# main_analysis.py
"""
Main script: run the complete clustering analysis pipeline.

Usage:
    cd clustering/
    python main_analysis.py
"""

import pandas as pd
import numpy as np

from config import RESULTS_DIR, FIGURES_DIR
from feature_engineering import (
    load_demand_data, load_nrel_data, load_risk_data, load_usgs_2023_thin_film,
    engineer_scenario_features, engineer_material_features,
)
from preprocessing import preprocess_pipeline
from clustering import ClusterAnalyzer
from visualization import (
    plot_elbow, plot_pca_biplot, plot_pca_biplot_centroid_labels, plot_silhouette,
    plot_cluster_profiles, plot_stress_matrix, plot_feature_sensitivity,
    plot_demand_spaghetti,
)


def main():
    print("=" * 80)
    print("CLUSTERING ANALYSIS — ENERGY TRANSITION SCENARIOS & MATERIALS")
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

    # Save raw features
    scenario_feats.to_csv(RESULTS_DIR / "scenario_features_raw.csv")
    material_feats.to_csv(RESULTS_DIR / "material_features_raw.csv")

    # ── Step 3: Preprocessing ─────────────────────────────────────────────
    # Note: Scenario preprocessing is done in Step 4 with two configurations
    scen_log_feats = [
        "total_cumulative_demand", "peak_demand", "mean_demand_early",
        "total_import_exposed_demand",
    ]

    print("\n▸ Step 3: Preprocessing materials...")
    mat_log_feats = [
        "mean_demand", "peak_demand", "domestic_production",
        "mean_capacity_ratio", "max_capacity_ratio",
        "domestic_reserves_years", "global_reserves_years",
    ]
    X_mat, scaler_mat, vif_mat, dropped_mat = preprocess_pipeline(
        material_feats, log_features=mat_log_feats,
    )
    print(f"  Final material feature set: {list(X_mat.columns)}")

    # ── Step 4: Scenario clustering (two configurations) ─────────────────
    print("\n▸ Step 4: Clustering scenarios...")

    # ────────────────────────────────────────────────────────────────────
    # Configuration A: VIF=50, enforced k=4
    # ────────────────────────────────────────────────────────────────────
    print("\n  ── Config A: VIF=50, k=4 ──")
    X_scen_v50, _, vif_scen_v50, dropped_scen_v50 = preprocess_pipeline(
        scenario_feats, log_features=scen_log_feats, vif_threshold=50.0,
    )
    print(f"    Features (VIF≤50): {list(X_scen_v50.columns)}")

    analyzer_scen_v50 = ClusterAnalyzer(X_scen_v50, name="scenarios_vif50")
    k_range_v50, wcss_v50, sil_v50 = analyzer_scen_v50.find_optimal_k()
    plot_elbow(k_range_v50, wcss_v50, sil_v50, "scenarios_vif50_k4")

    best_k_v50 = 4
    print(f"    → Using k={best_k_v50}")
    scen_labels_v50 = analyzer_scen_v50.fit_final_model(best_k_v50)
    analyzer_scen_v50.validate_stability()
    sens_scen_v50 = analyzer_scen_v50.feature_sensitivity()

    # Cluster interpretations based on feature profiles:
    # - Cluster 0: High stress + front-loaded (Con_NG, DAC/Low_Demand_CO2e_100by2035)
    # - Cluster 1: Baseline majority (n=35) - gradual, moderate stress
    # - Cluster 2: All 100by2035 targets - steep decline, high storage needs
    # - Cluster 3: Growing demand (Con_RE, High_H2, No_IRA) - only cluster with positive slope
    scen_cluster_names_v50 = {
        0: "High-stress front-loaded\n(rapid early deployment)",
        1: "Baseline steady-state\n(gradual transition)",
        2: "Deep decarbonization\n(100% by 2035 targets)",
        3: "Sustained growth\n(conservative/no policy)",
    }

    # Centroid-labeled biplot for VIF=50, k=4
    plot_pca_biplot_centroid_labels(
        X_scen_v50.values, scen_labels_v50, list(X_scen_v50.columns),
        "scenarios_vif50_k4", entity_names=list(X_scen_v50.index),
        cluster_names=scen_cluster_names_v50,
        raw_features=scenario_feats,
    )
    plot_silhouette(X_scen_v50.values, scen_labels_v50, "scenarios_vif50_k4")
    plot_feature_sensitivity(sens_scen_v50, "scenarios_vif50_k4")

    # ────────────────────────────────────────────────────────────────────
    # Configuration B: VIF=10, silhouette-optimal k
    # ────────────────────────────────────────────────────────────────────
    print("\n  ── Config B: VIF=10, silhouette-optimal k ──")
    X_scen_v10, _, vif_scen_v10, dropped_scen_v10 = preprocess_pipeline(
        scenario_feats, log_features=scen_log_feats, vif_threshold=10.0,
    )
    print(f"    Features (VIF≤10): {list(X_scen_v10.columns)}")

    analyzer_scen_v10 = ClusterAnalyzer(X_scen_v10, name="scenarios_vif10")
    k_range_v10, wcss_v10, sil_v10 = analyzer_scen_v10.find_optimal_k()
    plot_elbow(k_range_v10, wcss_v10, sil_v10, "scenarios_vif10_optimal")

    # Use silhouette-optimal k
    best_k_v10 = k_range_v10[int(np.argmax(sil_v10))]
    print(f"    → Silhouette-optimal k={best_k_v10}")
    scen_labels_v10 = analyzer_scen_v10.fit_final_model(best_k_v10)
    analyzer_scen_v10.validate_stability()
    sens_scen_v10 = analyzer_scen_v10.feature_sensitivity()

    # Cluster interpretations based on demand_slope × peak_supply_chain_stress:
    # - Cluster 0: Steep decline, moderate stress (100% clean targets)
    # - Cluster 1: Nearly flat, elevated stress (constrained RE scenarios)
    # - Cluster 2: Baseline majority (n=25) - flat, moderate stress
    # - Cluster 3: Strong growth, low stress (H2 economy, 2050 targets)
    # - Cluster 4: Steep decline, HIGHEST stress (aggressive + constraints)
    # - Cluster 5: Moderate decline, lowest stress (balanced advanced scenarios)
    scen_cluster_names_v10 = {
        0: "Rapid decarbonization\n(100% clean targets)",
        1: "Constrained transition\n(limited RE, elevated stress)",
        2: "Baseline steady-state",
        3: "Sustained H2 growth\n(long-term expansion)",
        4: "High-stress rapid decline\n(aggressive + constraints)",
        5: "Low-stress balanced\n(advanced technologies)",
    }

    # Centroid-labeled biplot for VIF=10, optimal k
    plot_pca_biplot_centroid_labels(
        X_scen_v10.values, scen_labels_v10, list(X_scen_v10.columns),
        "scenarios_vif10_optimal", entity_names=list(X_scen_v10.index),
        cluster_names=scen_cluster_names_v10,
        raw_features=scenario_feats,
    )
    plot_silhouette(X_scen_v10.values, scen_labels_v10, "scenarios_vif10_optimal")
    plot_feature_sensitivity(sens_scen_v10, "scenarios_vif10_optimal")

    # ────────────────────────────────────────────────────────────────────
    # Use Config A (VIF=50, k=4) as the primary for downstream analysis
    # ────────────────────────────────────────────────────────────────────
    X_scen = X_scen_v50
    scen_labels = scen_labels_v50
    analyzer_scen = analyzer_scen_v50
    vif_scen = vif_scen_v50
    dropped_scen = dropped_scen_v50
    best_k_scen = best_k_v50
    scen_cluster_names = scen_cluster_names_v50

    # Also generate spaghetti plot for primary config
    plot_demand_spaghetti(
        demand, scen_labels, X_scen.index, "scenarios",
        cluster_names=scen_cluster_names,
    )

    # ── Step 5: Material clustering ───────────────────────────────────────
    print("\n▸ Step 5: Clustering materials...")
    analyzer_mat = ClusterAnalyzer(X_mat, name="materials")
    k_range_m, wcss_m, sil_m = analyzer_mat.find_optimal_k()
    plot_elbow(k_range_m, wcss_m, sil_m, "materials")

    #best_k_mat = k_range_m[int(np.argmax(sil_m))]
    best_k_mat = 4
    print(f"\n  → Using k={best_k_mat} for materials")
    mat_labels = analyzer_mat.fit_final_model(best_k_mat)
    analyzer_mat.validate_stability()
    sens_mat = analyzer_mat.feature_sensitivity()

    # Material cluster names based on cluster profiles
    # These will be refined after inspecting cluster characteristics
    mat_cluster_names = {
        0: "Bulk construction\n(Steel, Cement, Glass)",
        1: "High-risk critical\n(REEs, Te, V)",
        2: "Moderate-risk specialty\n(Cd, Ga, Se, In)",
        3: "Established supply\n(Cu, Al, Si, Ni)",
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

    vif_scen.to_csv(RESULTS_DIR / "vif_scenarios.csv", index=False)
    vif_mat.to_csv(RESULTS_DIR / "vif_materials.csv", index=False)

    # Validation summary
    with open(RESULTS_DIR / "validation_metrics.txt", "w") as f:
        f.write(f"Scenario clustering: k={best_k_scen}\n")
        f.write(f"  Silhouette: {analyzer_scen.results.get('silhouette_scores', [0])[-1]:.3f}\n")
        f.write(f"  Stability ARI: {analyzer_scen.results.get('stability_ari_mean', 0):.3f}\n")
        f.write(f"  Features used: {list(X_scen.columns)}\n")
        f.write(f"  Features dropped (VIF): {dropped_scen}\n\n")
        f.write(f"Material clustering: k={best_k_mat}\n")
        f.write(f"  Silhouette: {analyzer_mat.results.get('silhouette_scores', [0])[-1]:.3f}\n")
        f.write(f"  Stability ARI: {analyzer_mat.results.get('stability_ari_mean', 0):.3f}\n")
        f.write(f"  Features used: {list(X_mat.columns)}\n")
        f.write(f"  Features dropped (VIF): {dropped_mat}\n")

    print(f"\n  Results → {RESULTS_DIR}")
    print(f"  Figures → {FIGURES_DIR}")
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
