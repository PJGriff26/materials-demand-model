"""
CV Borrowing for Small Sample Sizes
====================================

Implements the simple CV borrowing approach decided by PI:
- For any n < 5: borrow the median CV from all samples with n > 5
- For n ≥ 5: use the fitted CV as-is

This modifies fitted lognormal distributions to use borrowed CV values
while preserving the median intensity from the original data.

Mathematical Basis:
------------------
For lognormal distribution:
    CV = sqrt(exp(sigma^2) - 1)
    sigma = sqrt(log(1 + CV^2))

Given borrowed CV and original median, we reconstruct:
    sigma_new = sqrt(log(1 + CV_borrowed^2))
    scale_new = median  # Preserve median from data

Author: Materials Demand Research Team
Date: March 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_lognormal_cv(sigma: float) -> float:
    """
    Compute coefficient of variation from lognormal shape parameter.

    Parameters
    ----------
    sigma : float
        Lognormal shape parameter (s parameter in scipy.stats.lognorm)

    Returns
    -------
    float
        Coefficient of variation
    """
    return np.sqrt(np.exp(sigma**2) - 1)


def compute_lognormal_sigma(cv: float) -> float:
    """
    Compute lognormal shape parameter from coefficient of variation.

    Parameters
    ----------
    cv : float
        Coefficient of variation

    Returns
    -------
    float
        Lognormal shape parameter (sigma)
    """
    return np.sqrt(np.log(1 + cv**2))


def apply_cv_borrowing(fitted_distributions: Dict, threshold_n: int = 5,
                      reference_n: int = 5) -> Dict:
    """
    Apply simple CV borrowing for low-sample-size pairs.

    Algorithm:
    1. Compute median CV from all pairs with n > reference_n
    2. For pairs with n < threshold_n:
       - Extract borrowed CV
       - Refit lognormal distribution with borrowed sigma
       - Preserve median intensity from original data
    3. For pairs with n ≥ threshold_n:
       - Keep original fitted distribution unchanged

    Parameters
    ----------
    fitted_distributions : Dict[Tuple[str, str], MaterialIntensityDistribution]
        Dictionary mapping (technology, material) to fitted distributions
    threshold_n : int, default=5
        Sample size threshold. Pairs with n < threshold_n get borrowed CV.
    reference_n : int, default=5
        Reference sample size. Median CV computed from all n > reference_n.

    Returns
    -------
    Dict
        Updated fitted_distributions with borrowed CVs applied

    Notes
    -----
    - Only modifies lognormal distributions
    - Non-lognormal distributions are left unchanged
    - Logs summary statistics of borrowing application
    """
    logger.info("=" * 80)
    logger.info("APPLYING CV BORROWING FOR LOW-SAMPLE-SIZE PAIRS")
    logger.info("=" * 80)
    logger.info(f"Threshold: n < {threshold_n} gets borrowed CV")
    logger.info(f"Reference: median CV from all n > {reference_n}")

    # Step 1: Compute reference CV from well-characterized pairs
    reference_cvs = []

    for (tech, mat), dist_info in fitted_distributions.items():
        if dist_info.n_samples > reference_n:
            if dist_info.best_fit is not None:
                if dist_info.best_fit.distribution_name == 'lognormal':
                    sigma = dist_info.best_fit.parameters.get('s', 0)
                    cv = compute_lognormal_cv(sigma)
                    reference_cvs.append(cv)

    if not reference_cvs:
        logger.warning(f"No lognormal distributions with n > {reference_n} found!")
        logger.warning("Cannot compute reference CV. Returning unchanged distributions.")
        return fitted_distributions

    median_cv = np.median(reference_cvs)
    borrowed_sigma = compute_lognormal_sigma(median_cv)

    logger.info(f"\nReference CV statistics from {len(reference_cvs)} pairs with n > {reference_n}:")
    logger.info(f"  Median CV: {median_cv:.3f}")
    logger.info(f"  Min CV: {np.min(reference_cvs):.3f}")
    logger.info(f"  Max CV: {np.max(reference_cvs):.3f}")
    logger.info(f"  Borrowed sigma (from median CV): {borrowed_sigma:.3f}")

    # Step 2: Apply borrowing to low-n pairs
    n_total = len(fitted_distributions)
    n_low_n = sum(1 for d in fitted_distributions.values() if d.n_samples < threshold_n)
    n_borrowed = 0
    n_skipped_not_lognormal = 0
    n_unchanged = 0

    logger.info(f"\nApplying CV borrowing:")
    logger.info(f"  Total pairs: {n_total}")
    logger.info(f"  Pairs with n < {threshold_n}: {n_low_n}")

    for (tech, mat), dist_info in fitted_distributions.items():
        # Only apply to low-n pairs
        if dist_info.n_samples >= threshold_n:
            n_unchanged += 1
            continue

        # Only apply to lognormal distributions
        if dist_info.best_fit is None or dist_info.best_fit.distribution_name != 'lognormal':
            n_skipped_not_lognormal += 1
            continue

        # Extract original parameters
        original_sigma = dist_info.best_fit.parameters.get('s', 0)
        original_loc = dist_info.best_fit.parameters.get('loc', 0)
        original_scale = dist_info.best_fit.parameters.get('scale', 1)
        original_cv = compute_lognormal_cv(original_sigma)

        # Compute median from original data to preserve central tendency
        data_median = np.median(dist_info.raw_data)

        # Refit with borrowed sigma
        # For lognormal: median = scale * exp(0) = scale (when loc=0)
        # So we set scale = data_median to preserve median
        new_scale = data_median if original_loc == 0 else original_scale

        # Update distribution parameters
        dist_info.best_fit.parameters['s'] = borrowed_sigma
        dist_info.best_fit.parameters['scale'] = new_scale
        dist_info.best_fit.parameters['loc'] = 0  # Force loc=0 for consistency

        # Update recommendation message
        dist_info.recommendation = (
            f"CV borrowed from median of n>{reference_n} pairs "
            f"(n={dist_info.n_samples}, CV: {original_cv:.3f} → {median_cv:.3f}, "
            f"median preserved: {data_median:.2f})"
        )

        n_borrowed += 1

    logger.info(f"\nCV Borrowing Summary:")
    logger.info(f"  Borrowed CV applied: {n_borrowed} pairs")
    logger.info(f"  Unchanged (n ≥ {threshold_n}): {n_unchanged} pairs")
    logger.info(f"  Skipped (not lognormal): {n_skipped_not_lognormal} pairs")
    logger.info(f"  Total: {n_total} pairs")
    logger.info("=" * 80)

    return fitted_distributions


def create_cv_borrowing_report(
    fitted_distributions: Dict,
    output_path: str,
    threshold_n: int = 5,
    reference_n: int = 5
):
    """
    Create a detailed report showing CV borrowing results.

    Parameters
    ----------
    fitted_distributions : Dict
        Fitted distributions with CV borrowing applied
    output_path : str
        Path to output CSV file
    threshold_n : int
        Sample size threshold for borrowing
    reference_n : int
        Reference sample size for median CV
    """
    rows = []

    # Compute reference CV
    reference_cvs = []
    for dist_info in fitted_distributions.values():
        if dist_info.n_samples > reference_n:
            if dist_info.best_fit is not None:
                if dist_info.best_fit.distribution_name == 'lognormal':
                    sigma = dist_info.best_fit.parameters.get('s', 0)
                    cv = compute_lognormal_cv(sigma)
                    reference_cvs.append(cv)

    median_cv = np.median(reference_cvs) if reference_cvs else np.nan

    for (tech, mat), dist_info in fitted_distributions.items():
        if dist_info.best_fit is None:
            continue

        n = dist_info.n_samples
        dist_name = dist_info.best_fit.distribution_name

        # Extract CV if lognormal
        if dist_name == 'lognormal':
            sigma = dist_info.best_fit.parameters.get('s', 0)
            cv = compute_lognormal_cv(sigma)
        else:
            cv = np.nan

        # Determine if borrowed
        was_borrowed = (n < threshold_n and dist_name == 'lognormal')
        cv_source = 'borrowed' if was_borrowed else 'fitted'

        rows.append({
            'technology': tech,
            'material': mat,
            'n': n,
            'distribution': dist_name,
            'cv': cv,
            'cv_source': cv_source,
            'median_intensity': dist_info.median,
            'borrowed_cv': median_cv if was_borrowed else np.nan,
            'recommendation': dist_info.recommendation
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, float_format='%.4f')

    logger.info(f"CV borrowing report saved to: {output_path}")

    # Print summary statistics
    logger.info(f"\n{'='*80}")
    logger.info("CV BORROWING REPORT SUMMARY")
    logger.info(f"{'='*80}")

    borrowed_df = df[df['cv_source'] == 'borrowed']
    fitted_df = df[df['cv_source'] == 'fitted']

    logger.info(f"Total pairs: {len(df)}")
    logger.info(f"  Borrowed CV: {len(borrowed_df)} pairs (n < {threshold_n}, lognormal)")
    logger.info(f"  Fitted CV: {len(fitted_df)} pairs")

    if len(borrowed_df) > 0:
        logger.info(f"\nBorrowed CV statistics:")
        logger.info(f"  Reference median CV: {median_cv:.3f}")
        logger.info(f"  Sample size range: {borrowed_df['n'].min():.0f} to {borrowed_df['n'].max():.0f}")
        logger.info(f"  Number of n=1: {(borrowed_df['n'] == 1).sum()}")
        logger.info(f"  Number of n=2: {(borrowed_df['n'] == 2).sum()}")
        logger.info(f"  Number of n=3: {(borrowed_df['n'] == 3).sum()}")
        logger.info(f"  Number of n=4: {(borrowed_df['n'] == 4).sum()}")

    logger.info(f"{'='*80}")
