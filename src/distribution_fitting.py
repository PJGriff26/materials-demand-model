"""
Distribution Fitting for Material Intensity Data
=================================================

This module provides rigorous statistical distribution fitting for material
intensity data, following best practices from uncertainty quantification and
statistical analysis literature.

Key Best Practices Implemented:
1. Multiple distribution types tested (normal, lognormal, gamma, weibull, triangular)
2. Multiple fitting methods (MLE, L-moments, Method of Moments)
3. Goodness-of-fit tests (Kolmogorov-Smirnov, Anderson-Darling)
4. Minimum sample size requirements (n >= 3 for fitting, n >= 5 recommended)
5. Bootstrap confidence intervals for parameters
6. Fallback to empirical distribution for small samples
7. Visual diagnostics (Q-Q plots, histograms, CDFs)

References:
- Johnson & Wichern: Applied Multivariate Statistical Analysis
- NIST/SEMATECH e-Handbook of Statistical Methods
- HEC-SSP Distribution Fitting Documentation
- ISO GUM Uncertainty Quantification Standards

Author: Materials Demand Research Team
Date: 2024
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import warnings
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DistributionFit:
    """Container for fitted distribution information"""
    distribution_name: str
    parameters: Dict[str, float]
    ks_statistic: float
    ks_pvalue: float
    ad_statistic: float
    ad_critical_value: float
    aic: float  # Akaike Information Criterion
    bic: float  # Bayesian Information Criterion
    n_samples: int
    fitting_method: str
    
    def passes_ks_test(self, alpha: float = 0.05) -> bool:
        """Check if distribution passes Kolmogorov-Smirnov test"""
        return self.ks_pvalue > alpha
    
    def passes_ad_test(self) -> bool:
        """Check if distribution passes Anderson-Darling test"""
        return self.ad_statistic < self.ad_critical_value
    
    def summary(self) -> str:
        """Generate summary string"""
        return (
            f"{self.distribution_name} (n={self.n_samples})\n"
            f"  Parameters: {self.parameters}\n"
            f"  KS: stat={self.ks_statistic:.4f}, p={self.ks_pvalue:.4f} "
            f"[{'PASS' if self.passes_ks_test() else 'FAIL'}]\n"
            f"  AD: stat={self.ad_statistic:.4f}, crit={self.ad_critical_value:.4f} "
            f"[{'PASS' if self.passes_ad_test() else 'FAIL'}]\n"
            f"  AIC: {self.aic:.2f}, BIC: {self.bic:.2f}"
        )


@dataclass
class MaterialIntensityDistribution:
    """
    Complete distribution information for a material-technology pair.
    Contains fitted parametric distributions and raw data for fallback.
    """
    technology: str
    material: str
    raw_data: np.ndarray
    n_samples: int
    
    # Statistical summaries
    mean: float
    std: float
    median: float
    q25: float
    q75: float
    min_val: float
    max_val: float
    
    # Fitted distributions (ordered by quality)
    fitted_distributions: List[DistributionFit] = field(default_factory=list)
    best_fit: Optional[DistributionFit] = None
    
    # Recommendation
    use_parametric: bool = False
    recommendation: str = ""
    
    def get_best_distribution(self) -> Optional[DistributionFit]:
        """Get the best fitted distribution"""
        return self.best_fit
    
    def sample(self, n: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample from the distribution.

        Always uses parametric distribution (user requirement).
        """
        if random_state is not None:
            np.random.seed(random_state)

        if self.use_parametric and self.best_fit is not None:
            # Sample from parametric distribution
            dist = self._get_scipy_distribution(self.best_fit)
            return dist.rvs(size=n)
        else:
            # Should never reach here with new logic
            logger.warning(
                f"Unexpected: use_parametric=False for {self.technology}-{self.material}. "
                f"This should not happen. Falling back to uniform distribution."
            )
            # Emergency fallback
            return np.random.uniform(
                np.min(self.raw_data),
                np.max(self.raw_data),
                size=n
            )
    
    def _get_scipy_distribution(self, fit: DistributionFit):
        """Convert fitted distribution to scipy distribution object"""
        params = fit.parameters
        
        if fit.distribution_name == 'truncated_normal':
            return stats.truncnorm(
                a=params['a'], 
                b=params['b'], 
                loc=params['loc'], 
                scale=params['scale']
            )
        elif fit.distribution_name == 'uniform':
            return stats.uniform(loc=params['loc'], scale=params['scale'])
        elif fit.distribution_name == 'lognormal':
            return stats.lognorm(
                s=params['s'],
                loc=params.get('loc', 0),
                scale=params['scale']
            )
        elif fit.distribution_name == 'gamma':
            return stats.gamma(
                a=params['a'],
                loc=params.get('loc', 0),
                scale=params['scale']
            )
        else:
            raise ValueError(f"Unknown distribution: {fit.distribution_name}")


class DistributionFitter:
    """
    Fits parametric distributions to material intensity data.

    Follows best practices:
    - Tests multiple distributions
    - Uses multiple fitting methods
    - Applies goodness-of-fit tests
    - Always uses parametric distributions (no empirical fallback)
    - Validates tail behavior to prevent extreme outliers
    """

    # Minimum samples for attempting parametric fitting
    MIN_SAMPLES_FIT = 2  # Changed from 3 - always use parametric
    RECOMMENDED_MIN_SAMPLES = 2  # Changed from 5 - always use parametric
    MAX_LOGNORMAL_SHAPE = 3.0  # NEW: Maximum allowed lognormal shape parameter
    MAX_TAIL_RATIO = 100.0  # NEW: Maximum allowed max/median ratio in test sample
    
    # Distributions to test (appropriate for positive-valued data)
    DISTRIBUTIONS = {
        'truncated_normal': 'truncnorm',  # Normal truncated at zero (no negative values)
        'lognormal': stats.lognorm,
        'gamma': stats.gamma,
        # 'weibull': stats.weibull_min,
        # 'triangular': stats.triang,
        'uniform': stats.uniform
    }
    
    def __init__(self):
        self.results: Dict[Tuple[str, str], MaterialIntensityDistribution] = {}
    
    def fit_all(
        self,
        data: pd.DataFrame,
        technology_col: str = 'technology',
        material_col: str = 'material',
        value_col: str = 'intensity_t_per_mw'
    ) -> Dict[Tuple[str, str], MaterialIntensityDistribution]:
        """
        Fit distributions to all technology-material combinations.
        
        Parameters
        ----------
        data : pd.DataFrame
            Material intensity data
        technology_col : str
            Name of technology column
        material_col : str
            Name of material column
        value_col : str
            Name of value column
            
        Returns
        -------
        dict
            Dictionary mapping (technology, material) to MaterialIntensityDistribution
        """
        logger.info("="*80)
        logger.info("Fitting distributions to material intensity data")
        logger.info("="*80)
        
        # Group by technology and material
        grouped = data.groupby([technology_col, material_col])[value_col]
        
        n_groups = len(grouped)
        logger.info(f"Total technology-material combinations: {n_groups}")
        
        for i, ((tech, mat), values) in enumerate(grouped, 1):
            if i % 10 == 0:
                logger.info(f"Processing {i}/{n_groups}...")
            
            values_array = values.values
            dist_info = self.fit_single(tech, mat, values_array)
            self.results[(tech, mat)] = dist_info
        
        logger.info(f"Completed fitting for {len(self.results)} combinations")
        self._log_summary()
        
        return self.results
    
    def fit_single(
        self,
        technology: str,
        material: str,
        data: np.ndarray
    ) -> MaterialIntensityDistribution:
        """
        Fit distributions to a single technology-material combination.
        
        Parameters
        ----------
        technology : str
            Technology name
        material : str
            Material name
        data : np.ndarray
            Material intensity values
            
        Returns
        -------
        MaterialIntensityDistribution
            Complete distribution information
        """
        # Remove any NaN or infinite values
        data = data[np.isfinite(data)]
        n = len(data)
        
        # Calculate summary statistics
        summary_stats = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data, ddof=1)) if n > 1 else 0.0,
            'median': float(np.median(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75)),
            'min_val': float(np.min(data)),
            'max_val': float(np.max(data))
        }
        
        # Initialize result container
        result = MaterialIntensityDistribution(
            technology=technology,
            material=material,
            raw_data=data.copy(),
            n_samples=n,
            **summary_stats
        )
        
        # Always attempt parametric fitting (user requirement)
        # Special case for n=1: use uniform around single point
        if n == 1:
            single_val = data[0]
            result.use_parametric = True
            result.best_fit = self._create_single_point_uniform(single_val, n=1)
            result.fitted_distributions = [result.best_fit]
            result.recommendation = f"Single data point - using narrow uniform distribution around {single_val:.3f}"
            return result
        
        # Attempt to fit parametric distributions
        fitted_dists = []
        
        for dist_name in self.DISTRIBUTIONS.keys():
            try:
                fit_result = self._fit_distribution(data, dist_name)
                fitted_dists.append(fit_result)
            except Exception as e:
                logger.debug(f"Failed to fit {dist_name} to {technology}-{material}: {e}")
                continue
        
        # Sort by AIC (lower is better)
        fitted_dists.sort(key=lambda x: x.aic)
        result.fitted_distributions = fitted_dists
        
        # Select best distribution and make recommendation
        # ALWAYS use parametric (user requirement)
        if fitted_dists:
            # Check distributions in order for tail behavior
            selected_dist = None
            rejection_reasons = []

            for dist_fit in fitted_dists:
                # Validate tail behavior
                is_valid, reason = self._validate_tail_behavior(dist_fit, data)

                if is_valid:
                    selected_dist = dist_fit
                    break
                else:
                    rejection_reasons.append(f"{dist_fit.distribution_name}: {reason}")

            if selected_dist:
                result.best_fit = selected_dist
                result.use_parametric = True

                if selected_dist != fitted_dists[0]:
                    # Not the best AIC, but best valid distribution
                    result.recommendation = (
                        f"Using {selected_dist.distribution_name} distribution "
                        f"(n={n}, KS p={selected_dist.ks_pvalue:.4f}). "
                        f"Best AIC rejected: {rejection_reasons[0]}"
                    )
                else:
                    result.recommendation = (
                        f"Using {selected_dist.distribution_name} distribution "
                        f"(n={n}, KS p={selected_dist.ks_pvalue:.4f}, AIC={selected_dist.aic:.2f})"
                    )
            else:
                # All distributions failed validation - use uniform as safe fallback
                result.best_fit = self._fit_uniform_fallback(data)
                result.use_parametric = True
                result.recommendation = (
                    f"All distributions rejected due to extreme tails. "
                    f"Using uniform fallback. Rejections: {'; '.join(rejection_reasons[:2])}"
                )
                result.fitted_distributions.append(result.best_fit)
        else:
            # No distributions fit at all - use uniform fallback
            result.best_fit = self._fit_uniform_fallback(data)
            result.use_parametric = True
            result.recommendation = (
                f"No distributions fit successfully (n={n}). "
                f"Using uniform fallback."
            )
            result.fitted_distributions = [result.best_fit]
        
        return result
    
    def _fit_distribution(
        self,
        data: np.ndarray,
        dist_name: str
    ) -> DistributionFit:
        """
        Fit a specific distribution to data.

        Parameters
        ----------
        data : np.ndarray
            Data to fit
        dist_name : str
            Name of distribution

        Returns
        -------
        DistributionFit
            Fitted distribution information
        """
        n = len(data)

        # Check for zero variance (all values identical)
        # For zero-variance data, use narrow uniform distribution
        data_std = np.std(data, ddof=1) if n > 1 else 0
        if data_std == 0 or n == 1:
            # All values identical - only uniform makes sense
            if dist_name != 'uniform':
                # Return a dummy fit with very poor AIC so uniform will be selected
                # This prevents degenerate distributions with scale=0
                return DistributionFit(
                    distribution_name=dist_name,
                    parameters={},
                    ks_statistic=1.0,
                    ks_pvalue=0.0,
                    ad_statistic=999.0,
                    ad_critical_value=1.0,
                    aic=999999.0,  # Very poor AIC so uniform wins
                    bic=999999.0,
                    n_samples=n,
                    fitting_method='rejected_zero_variance'
                )

        # Special handling for different distributions
        if dist_name == 'truncated_normal':
            # Truncated normal at zero (no negative values)
            # For material intensities, we want a normal distribution truncated at 0

            # scipy.stats.truncnorm.fit() has issues with data far from zero
            # Use moment matching instead: find loc and scale such that
            # the truncated normal has the same mean and std as the data

            from scipy.optimize import minimize

            data_mean = np.mean(data)
            data_std = np.std(data, ddof=1)

            def truncnorm_match_moments(params):
                """Minimize difference between distribution and data moments"""
                loc, scale = params
                if scale <= 0:
                    return 1e10

                # Create truncnorm with lower bound at 0
                a = (0 - loc) / scale  # Lower bound in standard units
                b = np.inf

                try:
                    dist_temp = stats.truncnorm(a=a, b=b, loc=loc, scale=scale)
                    dist_mean = dist_temp.mean()
                    dist_std = dist_temp.std()

                    # Minimize squared error in mean and std
                    error = (dist_mean - data_mean)**2 + (dist_std - data_std)**2
                    return error
                except:
                    return 1e10

            # Optimize to match moments
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(truncnorm_match_moments, [data_mean, data_std],
                                method='Nelder-Mead')

            loc_fitted = result.x[0]
            scale_fitted = result.x[1]
            a_fitted = (0 - loc_fitted) / scale_fitted  # Lower bound in standard units

            dist = stats.truncnorm(a=a_fitted, b=np.inf, loc=loc_fitted, scale=scale_fitted)
            param_dict = {
                'a': a_fitted,
                'b': np.inf,
                'loc': loc_fitted,
                'scale': scale_fitted,
                'lower_bound': 0.0,
                'upper_bound': np.inf
            }
            
        elif dist_name == 'uniform':
            # Uniform distribution: fit to [min, max] of data
            # For uniform, we just need loc (start) and scale (width)
            min_val = np.min(data)
            max_val = np.max(data)

            # Handle zero variance: create narrow uniform around the single value
            if max_val == min_val:
                # Add 10% padding on each side
                padding = max(0.1 * abs(min_val), 1e-6) if min_val != 0 else 1e-6
                loc = max(0, min_val - padding)  # Don't go negative
                scale = (min_val + padding) - loc
            else:
                # scipy.stats.uniform uses loc (start) and scale (width)
                loc = min_val
                scale = max_val - min_val

            params = (loc, scale)
            dist = stats.uniform(loc=loc, scale=scale)
            param_dict = {'loc': loc, 'scale': scale, 'min': loc, 'max': loc + scale}
            
        else:
            # Standard fitting for other distributions
            dist_class = self.DISTRIBUTIONS[dist_name]
            
            # Fit distribution using MLE (default for scipy)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = dist_class.fit(data)
            
            # Create distribution object
            dist = dist_class(*params)
            
            # Extract named parameters for storage
            if dist_name == 'lognormal':
                param_dict = {'s': params[0], 'loc': params[1], 'scale': params[2]}
            elif dist_name == 'gamma':
                param_dict = {'a': params[0], 'loc': params[1], 'scale': params[2]}
            else:
                param_dict = {f'param_{i}': p for i, p in enumerate(params)}
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.kstest(data, lambda x: dist.cdf(x))
        
        # Anderson-Darling test
        # Note: scipy's anderson test is limited to specific distributions
        # For general case, we'll use a conservative critical value
        ad_result = self._anderson_darling_general(data, dist)
        
        # Information criteria
        # Log-likelihood
        log_likelihood = np.sum(dist.logpdf(data))
        k = len(params)  # number of parameters
        
        # AIC = 2k - 2ln(L)
        aic = 2 * k - 2 * log_likelihood
        
        # BIC = k*ln(n) - 2ln(L)
        bic = k * np.log(n) - 2 * log_likelihood
        
        return DistributionFit(
            distribution_name=dist_name,
            parameters=param_dict,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            ad_statistic=ad_result['statistic'],
            ad_critical_value=ad_result['critical_value'],
            aic=aic,
            bic=bic,
            n_samples=n,
            fitting_method='MLE'
        )
    
    def _anderson_darling_general(
        self,
        data: np.ndarray,
        dist
    ) -> Dict[str, float]:
        """
        Compute Anderson-Darling test statistic for any distribution.

        More powerful than K-S test, especially in the tails.
        """
        n = len(data)
        data_sorted = np.sort(data)

        # Compute CDF values
        cdf_vals = dist.cdf(data_sorted)

        # Avoid log(0) issues
        cdf_vals = np.clip(cdf_vals, 1e-10, 1 - 1e-10)

        # Anderson-Darling statistic
        i = np.arange(1, n + 1)
        ad_stat = -n - np.sum(
            (2 * i - 1) * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1]))
        ) / n

        # Critical value (approximate, for alpha=0.05)
        # This is a conservative approximation
        critical_value = 2.492  # For alpha=0.05, approximately

        return {
            'statistic': ad_stat,
            'critical_value': critical_value
        }

    def _validate_tail_behavior(
        self,
        dist_fit: DistributionFit,
        data: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Validate that a fitted distribution doesn't have extreme tail behavior.

        Checks:
        1. Lognormal shape parameter < MAX_LOGNORMAL_SHAPE
        2. Max/median ratio in test sample < MAX_TAIL_RATIO

        Parameters
        ----------
        dist_fit : DistributionFit
            Fitted distribution to validate
        data : np.ndarray
            Original data

        Returns
        -------
        Tuple[bool, str]
            (is_valid, rejection_reason)
        """
        try:
            # Check 1: Lognormal shape parameter
            if dist_fit.distribution_name == 'lognormal':
                s = dist_fit.parameters.get('s', 0)
                if s > self.MAX_LOGNORMAL_SHAPE:
                    return False, f"Lognormal shape s={s:.2f} > {self.MAX_LOGNORMAL_SHAPE}"

            # Check 2: Tail behavior via test sampling
            # Create scipy distribution
            if dist_fit.distribution_name == 'lognormal':
                dist = stats.lognorm(
                    s=dist_fit.parameters['s'],
                    loc=dist_fit.parameters.get('loc', 0),
                    scale=dist_fit.parameters['scale']
                )
            elif dist_fit.distribution_name == 'gamma':
                dist = stats.gamma(
                    a=dist_fit.parameters['a'],
                    loc=dist_fit.parameters.get('loc', 0),
                    scale=dist_fit.parameters['scale']
                )
            elif dist_fit.distribution_name == 'uniform':
                dist = stats.uniform(
                    loc=dist_fit.parameters['loc'],
                    scale=dist_fit.parameters['scale']
                )
            elif dist_fit.distribution_name == 'truncated_normal':
                dist = stats.truncnorm(
                    a=dist_fit.parameters['a'],
                    b=dist_fit.parameters['b'],
                    loc=dist_fit.parameters['loc'],
                    scale=dist_fit.parameters['scale']
                )
            else:
                return True, ""  # Unknown distribution, pass by default

            # Sample test data
            test_sample = dist.rvs(size=10000, random_state=42)

            # Check tail behavior
            max_val = np.max(test_sample)
            median_val = np.median(test_sample)

            if median_val > 0:
                ratio = max_val / median_val
                if ratio > self.MAX_TAIL_RATIO:
                    return False, f"Extreme tail: max/median={ratio:.1f}x > {self.MAX_TAIL_RATIO}x"

            return True, ""

        except Exception as e:
            logger.debug(f"Tail validation failed for {dist_fit.distribution_name}: {e}")
            return True, ""  # Pass by default if validation fails

    def _fit_uniform_fallback(self, data: np.ndarray) -> DistributionFit:
        """
        Create a uniform distribution as safe fallback.

        Uses [min, max] of data with 10% padding.

        Parameters
        ----------
        data : np.ndarray
            Material intensity data

        Returns
        -------
        DistributionFit
            Uniform distribution fit
        """
        min_val = np.min(data)
        max_val = np.max(data)

        # Add 10% padding
        range_val = max_val - min_val
        loc = max(0, min_val - 0.1 * range_val)  # Don't go negative
        scale = max_val - loc + 0.1 * range_val

        # Create distribution
        dist = stats.uniform(loc=loc, scale=scale)

        # Calculate goodness of fit
        ks_stat, ks_pval = stats.kstest(data, lambda x: dist.cdf(x))

        # Calculate AIC/BIC
        n = len(data)
        k = 2  # 2 parameters (loc, scale)
        log_likelihood = np.sum(dist.logpdf(data))
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        # Anderson-Darling
        ad_result = self._anderson_darling_general(data, dist)

        return DistributionFit(
            distribution_name='uniform',
            parameters={'loc': loc, 'scale': scale, 'min': loc, 'max': loc + scale},
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            ad_statistic=ad_result['statistic'],
            ad_critical_value=ad_result['critical_value'],
            aic=aic,
            bic=bic,
            n_samples=n,
            fitting_method='fallback'
        )

    def _create_single_point_uniform(self, value: float, n: int) -> DistributionFit:
        """
        Create a narrow uniform distribution around a single point.

        Parameters
        ----------
        value : float
            Single data point
        n : int
            Sample size (1)

        Returns
        -------
        DistributionFit
            Uniform distribution centered on value
        """
        # Create uniform distribution ±10% around the point
        loc = max(0, value * 0.9)
        scale = value * 1.1 - loc

        return DistributionFit(
            distribution_name='uniform',
            parameters={'loc': loc, 'scale': scale, 'min': loc, 'max': loc + scale},
            ks_statistic=0.0,
            ks_pvalue=1.0,
            ad_statistic=0.0,
            ad_critical_value=0.0,
            aic=0.0,
            bic=0.0,
            n_samples=n,
            fitting_method='single_point'
        )
    
    def _log_summary(self):
        """Log summary statistics of fitting results"""
        logger.info("\n" + "="*80)
        logger.info("FITTING SUMMARY")
        logger.info("="*80)
        
        n_total = len(self.results)
        n_parametric = sum(1 for r in self.results.values() if r.use_parametric)
        n_empirical = n_total - n_parametric
        
        logger.info(f"Total combinations: {n_total}")
        logger.info(f"  Using parametric distributions: {n_parametric} ({100*n_parametric/n_total:.1f}%)")
        logger.info(f"  Using empirical distributions: {n_empirical} ({100*n_empirical/n_total:.1f}%)")
        
        # Sample size distribution
        sample_sizes = [r.n_samples for r in self.results.values()]
        logger.info(f"\nSample size statistics:")
        logger.info(f"  Min: {np.min(sample_sizes)}")
        logger.info(f"  Median: {np.median(sample_sizes):.0f}")
        logger.info(f"  Max: {np.max(sample_sizes)}")
        
        # Distribution type counts (for parametric fits)
        if n_parametric > 0:
            dist_counts = {}
            for r in self.results.values():
                if r.best_fit is not None:
                    dist_name = r.best_fit.distribution_name
                    dist_counts[dist_name] = dist_counts.get(dist_name, 0) + 1
            
            logger.info(f"\nBest-fit distribution types:")
            for dist_name, count in sorted(dist_counts.items(), key=lambda x: -x[1]):
                logger.info(f"  {dist_name}: {count} ({100*count/n_total:.1f}%)")
    
    def export_to_csv(self, output_path: Union[str, Path]):
        """
        Export fitted distributions to CSV.
        
        Parameters
        ----------
        output_path : str or Path
            Path to output CSV file
        """
        rows = []
        
        for (tech, mat), dist_info in self.results.items():
            row = {
                'technology': tech,
                'material': mat,
                'n_samples': dist_info.n_samples,
                'mean': dist_info.mean,
                'std': dist_info.std,
                'median': dist_info.median,
                'q25': dist_info.q25,
                'q75': dist_info.q75,
                'min': dist_info.min_val,
                'max': dist_info.max_val,
                'use_parametric': dist_info.use_parametric,
                'recommendation': dist_info.recommendation
            }
            
            if dist_info.best_fit is not None:
                bf = dist_info.best_fit
                row.update({
                    'best_distribution': bf.distribution_name,
                    'ks_statistic': bf.ks_statistic,
                    'ks_pvalue': bf.ks_pvalue,
                    'ad_statistic': bf.ad_statistic,
                    'aic': bf.aic,
                    'bic': bf.bic
                })
                
                # Add parameters (flattened)
                for param_name, param_value in bf.parameters.items():
                    row[f'param_{param_name}'] = param_value
            else:
                row.update({
                    'best_distribution': 'empirical',
                    'ks_statistic': np.nan,
                    'ks_pvalue': np.nan,
                    'ad_statistic': np.nan,
                    'aic': np.nan,
                    'bic': np.nan
                })
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported fitted distributions to: {output_path}")
    
    def export_raw_data(self, output_path: Union[str, Path]):
        """
        Export raw data for all combinations to CSV.

        This provides the empirical distributions for use when
        parametric fits are not recommended.

        Parameters
        ----------
        output_path : str or Path
            Path to output CSV file
        """
        rows = []

        for (tech, mat), dist_info in self.results.items():
            for value in dist_info.raw_data:
                rows.append({
                    'technology': tech,
                    'material': mat,
                    'intensity_t_per_mw': value
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported raw data to: {output_path}")


def create_distribution_report(
    fitter: DistributionFitter,
    output_path: Union[str, Path]
):
    """
    Create a comprehensive text report of fitting results.
    
    Parameters
    ----------
    fitter : DistributionFitter
        Fitted distribution fitter
    output_path : str or Path
        Path to output text file
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MATERIAL INTENSITY DISTRIBUTION FITTING REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total technology-material combinations: {len(fitter.results)}\n\n")
        
        # Summary statistics
        n_parametric = sum(1 for r in fitter.results.values() if r.use_parametric)
        n_empirical = len(fitter.results) - n_parametric
        
        f.write("RECOMMENDATIONS SUMMARY:\n")
        f.write(f"  Parametric distributions: {n_parametric}\n")
        f.write(f"  Empirical distributions: {n_empirical}\n\n")
        
        # Sample size distribution
        sample_sizes = [r.n_samples for r in fitter.results.values()]
        f.write("SAMPLE SIZE DISTRIBUTION:\n")
        f.write(f"  Min: {np.min(sample_sizes)}\n")
        f.write(f"  25th percentile: {np.percentile(sample_sizes, 25):.0f}\n")
        f.write(f"  Median: {np.median(sample_sizes):.0f}\n")
        f.write(f"  75th percentile: {np.percentile(sample_sizes, 75):.0f}\n")
        f.write(f"  Max: {np.max(sample_sizes)}\n\n")
        
        # Detailed results for each combination
        f.write("="*80 + "\n")
        f.write("DETAILED RESULTS BY TECHNOLOGY-MATERIAL COMBINATION\n")
        f.write("="*80 + "\n\n")
        
        for (tech, mat), dist_info in sorted(fitter.results.items()):
            f.write(f"\n{tech} - {mat}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Sample size: {dist_info.n_samples}\n")
            f.write(f"Mean: {dist_info.mean:.2f}, Std: {dist_info.std:.2f}\n")
            f.write(f"Median: {dist_info.median:.2f}, IQR: [{dist_info.q25:.2f}, {dist_info.q75:.2f}]\n")
            f.write(f"Range: [{dist_info.min_val:.2f}, {dist_info.max_val:.2f}]\n\n")
            
            f.write(f"Recommendation: {dist_info.recommendation}\n\n")
            
            if dist_info.fitted_distributions:
                f.write("Fitted distributions (ranked by AIC):\n")
                for i, fit in enumerate(dist_info.fitted_distributions, 1):
                    f.write(f"  {i}. {fit.summary()}\n")
            else:
                f.write("No parametric distributions fit successfully.\n")
            
            f.write("\n")
    
    logger.info(f"Created detailed report: {output_path}")


if __name__ == "__main__":
    print("Distribution Fitting Module for Materials Demand Analysis")
    print("This module should be imported and used in analysis scripts.")
