# preprocessing.py
"""
Preprocessing pipeline: log transformation, standardization, VIF checks.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from config import *


def log_transform_features(df, features_to_transform):
    """
    Apply log10(x + 1) transformation to specified features.
    Replaces the original columns with their log-transformed versions.
    """
    df_out = df.copy()
    for feat in features_to_transform:
        if feat in df_out.columns:
            df_out[feat] = np.log10(df_out[feat].clip(lower=0) + 1)
    return df_out


def standardize_features(df):
    """
    Z-score standardisation (mean=0, std=1).
    Returns the standardised DataFrame and the fitted scaler.
    """
    scaler = StandardScaler()
    arr = scaler.fit_transform(df)
    return pd.DataFrame(arr, index=df.index, columns=df.columns), scaler


def calculate_vif(df):
    """
    Variance Inflation Factor for each feature.
    VIF > 10 → severe multicollinearity.
    """
    cols = list(df.columns)
    X = df.values.astype(float)
    # Guard against constant columns
    keep = [i for i in range(X.shape[1]) if X[:, i].std() > 0]
    X_clean = X[:, keep]
    cols_clean = [cols[i] for i in keep]

    vif_data = pd.DataFrame({
        "Feature": cols_clean,
        "VIF": [
            variance_inflation_factor(X_clean, i)
            for i in range(len(cols_clean))
        ],
    })
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)


def drop_high_vif(df, threshold=10.0, verbose=True):
    """
    Iteratively drop the feature with the highest VIF until all VIF < threshold.
    """
    df_work = df.copy()
    while True:
        vif = calculate_vif(df_work)
        if vif["VIF"].max() <= threshold:
            break
        worst = vif.iloc[0]
        if verbose:
            print(f"  Dropping '{worst['Feature']}' (VIF={worst['VIF']:.1f})")
        df_work = df_work.drop(columns=[worst["Feature"]])
    return df_work


def preprocess_pipeline(features_df, log_features, vif_threshold=10.0):
    """
    Complete preprocessing pipeline.

    Parameters
    ----------
    features_df : DataFrame
        Raw feature matrix (rows=entities, columns=features).
    log_features : list[str]
        Column names to log-transform.
    vif_threshold : float
        Drop features above this VIF iteratively.

    Returns
    -------
    X_std : DataFrame  – standardised feature matrix ready for clustering
    scaler : StandardScaler
    vif_results : DataFrame
    dropped_features : list[str]
    """
    # 1. Log transformation
    df = log_transform_features(features_df, log_features)

    # 2. Drop zero-variance columns
    df = df.loc[:, df.std() > 0]

    # 3. VIF check (before standardisation, on raw/log scale)
    vif_before = calculate_vif(df)
    print("\nVIF before feature removal:")
    print(vif_before.to_string(index=False))

    original_cols = set(df.columns)
    df = drop_high_vif(df, threshold=vif_threshold)
    dropped = sorted(original_cols - set(df.columns))
    if dropped:
        print(f"\nDropped for high VIF: {dropped}")

    # 4. Final VIF
    vif_after = calculate_vif(df)
    print("\nVIF after cleanup:")
    print(vif_after.to_string(index=False))

    # 5. Standardise
    X_std, scaler = standardize_features(df)

    return X_std, scaler, vif_after, dropped


if __name__ == "__main__":
    print("Testing preprocessing pipeline...")
    # Quick sanity check with random data
    rng = np.random.default_rng(42)
    fake = pd.DataFrame(
        rng.random((20, 5)),
        columns=["a", "b", "c", "d", "e"],
    )
    X, sc, vif, dropped = preprocess_pipeline(fake, log_features=["a", "b"])
    print(f"\nFinal features: {list(X.columns)}")
    print(f"Dropped: {dropped}")
    print("Preprocessing test passed.")
