# clustering.py
"""
K-means clustering with optimal k selection and validation.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from config import (
    K_RANGE, N_INIT, MAX_ITER, TOLERANCE, RANDOM_SEED,
    N_STABILITY_RUNS, SILHOUETTE_THRESHOLD,
)


class ClusterAnalyzer:
    """Comprehensive K-means clustering with validation."""

    def __init__(self, data, name="dataset"):
        """
        Parameters
        ----------
        data : DataFrame
            Standardised feature matrix (rows=entities, columns=features).
        name : str
            Label used in print statements and filenames.
        """
        self.data = data
        self.name = name
        self.results = {}
        self.kmeans = None
        self.labels = None
        self.optimal_k = None
        self.silhouette_per_sample = None

    # ── Optimal k search ──────────────────────────────────────────────────

    def find_optimal_k(self, k_range=None):
        """
        Run K-means for each k and record WCSS and silhouette.
        """
        if k_range is None:
            k_range = K_RANGE
        k_range = list(k_range)

        wcss = []
        sil_scores = []

        for k in k_range:
            km = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=N_INIT,
                max_iter=MAX_ITER,
                tol=TOLERANCE,
                random_state=RANDOM_SEED,
            )
            km.fit(self.data)
            wcss.append(km.inertia_)
            sil = silhouette_score(self.data, km.labels_)
            sil_scores.append(sil)
            print(f"  k={k:2d}  WCSS={km.inertia_:10.2f}  Silhouette={sil:.3f}")

        self.results["k_range"] = k_range
        self.results["wcss"] = wcss
        self.results["silhouette_scores"] = sil_scores

        # Recommend k with highest silhouette
        best_idx = int(np.argmax(sil_scores))
        recommended_k = k_range[best_idx]
        print(f"\n  Recommended k={recommended_k} (silhouette={sil_scores[best_idx]:.3f})")

        return k_range, wcss, sil_scores

    # ── Fit final model ───────────────────────────────────────────────────

    def fit_final_model(self, optimal_k):
        """Fit final K-means with chosen k."""
        self.optimal_k = optimal_k
        self.kmeans = KMeans(
            n_clusters=optimal_k,
            init="k-means++",
            n_init=N_INIT,
            max_iter=MAX_ITER,
            tol=TOLERANCE,
            random_state=RANDOM_SEED,
        )
        self.labels = self.kmeans.fit_predict(self.data)
        self.silhouette_per_sample = silhouette_samples(self.data, self.labels)

        overall_sil = silhouette_score(self.data, self.labels)
        print(f"\n  Final model: k={optimal_k}, silhouette={overall_sil:.3f}")

        # Per-cluster summary
        for c in range(optimal_k):
            mask = self.labels == c
            n = mask.sum()
            mean_sil = self.silhouette_per_sample[mask].mean()
            print(f"    Cluster {c}: n={n}, mean silhouette={mean_sil:.3f}")

        return self.labels

    # ── Stability validation ──────────────────────────────────────────────

    def validate_stability(self, n_runs=None):
        """
        Run K-means with different random seeds and measure pairwise ARI.
        """
        if n_runs is None:
            n_runs = N_STABILITY_RUNS

        all_labels = []
        for i in range(n_runs):
            km = KMeans(
                n_clusters=self.optimal_k,
                init="k-means++",
                n_init=1,
                random_state=i,
            )
            all_labels.append(km.fit_predict(self.data))

        ari_scores = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                ari_scores.append(adjusted_rand_score(all_labels[i], all_labels[j]))

        mean_ari = np.mean(ari_scores)
        std_ari = np.std(ari_scores)
        print(f"\n  Stability ({n_runs} runs): ARI = {mean_ari:.3f} ± {std_ari:.3f}")
        if mean_ari > 0.9:
            print("    → Very stable")
        elif mean_ari > 0.7:
            print("    → Moderately stable")
        else:
            print("    → Unstable — consider different k or features")

        self.results["stability_ari_mean"] = mean_ari
        self.results["stability_ari_std"] = std_ari
        return mean_ari

    # ── Feature sensitivity ───────────────────────────────────────────────

    def feature_sensitivity(self):
        """
        Leave-one-feature-out: re-cluster and measure ARI vs full model.
        """
        if self.labels is None:
            raise RuntimeError("Call fit_final_model first.")

        cols = list(self.data.columns)
        results = []
        for col in cols:
            X_reduced = self.data.drop(columns=[col])
            km = KMeans(
                n_clusters=self.optimal_k,
                init="k-means++",
                n_init=N_INIT,
                random_state=RANDOM_SEED,
            )
            labels_reduced = km.fit_predict(X_reduced)
            ari = adjusted_rand_score(self.labels, labels_reduced)
            results.append({"feature_dropped": col, "ARI_vs_full": ari})

        sens = pd.DataFrame(results).sort_values("ARI_vs_full")
        print("\n  Feature sensitivity (lower ARI = more influential):")
        print(sens.to_string(index=False))
        self.results["feature_sensitivity"] = sens
        return sens

    # ── Cluster profiles ──────────────────────────────────────────────────

    def get_cluster_profiles(self, raw_features):
        """
        Mean raw feature values per cluster.

        Parameters
        ----------
        raw_features : DataFrame
            Un-standardised features, same index as self.data.
        """
        raw_features = raw_features.copy()
        raw_features["cluster"] = self.labels
        profiles = raw_features.groupby("cluster").mean()
        return profiles


if __name__ == "__main__":
    print("Testing clustering module...")
    rng = np.random.default_rng(42)
    fake = pd.DataFrame(rng.random((50, 4)), columns=["a", "b", "c", "d"])
    ca = ClusterAnalyzer(fake, name="test")
    ca.find_optimal_k(range(2, 6))
    ca.fit_final_model(3)
    ca.validate_stability(n_runs=10)
    ca.feature_sensitivity()
    print("\nClustering test passed.")
