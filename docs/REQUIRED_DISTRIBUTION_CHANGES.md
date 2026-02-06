# Required Distribution Fitting Changes

## User Requirement

**Always fit parametric distributions** - never fall back to empirical/bootstrap sampling.

All materials must use one of:
- Truncated normal
- Lognormal
- Gamma
- Uniform

Even for small sample sizes (n = 2, 3, 4), fit a parametric distribution and use it.

---

## Current Behavior (Incorrect)

The code currently falls back to empirical sampling in three cases:

### 1. Insufficient Data (n < 3)
**Location**: [distribution_fitting.py:278-284](src/distribution_fitting.py#L278-L284)

```python
if n < self.MIN_SAMPLES_FIT:
    result.recommendation = (
        f"Insufficient data (n={n} < {self.MIN_SAMPLES_FIT}). "
        f"Use raw data/empirical distribution."
    )
    result.use_parametric = False
    return result  # ❌ Returns without fitting
```

### 2. Small Sample (n < 5)
**Location**: [distribution_fitting.py:308-314](src/distribution_fitting.py#L308-L314)

```python
if n < self.RECOMMENDED_MIN_SAMPLES:
    result.use_parametric = False  # ❌ Forces empirical
    result.recommendation = (
        f"Small sample (n={n} < {self.RECOMMENDED_MIN_SAMPLES}). "
        f"Best fit: {best.distribution_name} (AIC={best.aic:.2f}), "
        f"but recommend using empirical distribution for robustness."
    )
```

### 3. Failed K-S Test
**Location**: [distribution_fitting.py:315-320](src/distribution_fitting.py#L315-L320)

```python
elif not best.passes_ks_test(alpha=0.05):
    result.use_parametric = False  # ❌ Forces empirical
    result.recommendation = (
        f"Best fit ({best.distribution_name}) fails K-S test "
        f"(p={best.ks_pvalue:.4f}). Recommend empirical distribution."
    )
```

---

## Required Changes

### Change 1: Remove Minimum Sample Check

**Current** (lines 278-284):
```python
# Check if we have enough data for parametric fitting
if n < self.MIN_SAMPLES_FIT:
    result.recommendation = (
        f"Insufficient data (n={n} < {self.MIN_SAMPLES_FIT}). "
        f"Use raw data/empirical distribution."
    )
    result.use_parametric = False
    return result
```

**New**:
```python
# Always attempt parametric fitting (user requirement)
# Minimum n=2 for most distributions
if n < 2:
    # For n=1, use uniform distribution around single point
    single_val = data[0]
    # Create a narrow uniform distribution
    result.use_parametric = True
    result.best_fit = DistributionFit(
        distribution_name='uniform',
        parameters={'loc': single_val * 0.9, 'scale': single_val * 0.2},
        ks_statistic=0.0,
        ks_pvalue=1.0,
        ad_statistic=0.0,
        ad_critical_value=0.0,
        aic=0.0,
        bic=0.0,
        n_samples=1,
        fitting_method='single_point'
    )
    result.recommendation = f"Single data point - using narrow uniform distribution around {single_val:.3f}"
    return result
```

### Change 2: Always Use Parametric

**Current** (lines 308-320):
```python
if n < self.RECOMMENDED_MIN_SAMPLES:
    result.use_parametric = False
    result.recommendation = (...)
elif not best.passes_ks_test(alpha=0.05):
    result.use_parametric = False
    result.recommendation = (...)
else:
    result.use_parametric = True
    result.recommendation = (...)
```

**New**:
```python
# Always use parametric distribution (user requirement)
result.use_parametric = True

# But add validation to prevent extreme tail behavior
if best.distribution_name == 'lognormal':
    s = best.parameters.get('s', 0)
    if s > 3.0:  # Shape parameter too large
        # Try next best distribution
        if len(fitted_dists) > 1:
            result.best_fit = fitted_dists[1]  # Use second-best
            result.recommendation = (
                f"Lognormal rejected (s={s:.2f} > 3.0). "
                f"Using {fitted_dists[1].distribution_name} instead."
            )
        else:
            # Fall back to uniform as safest option
            result.best_fit = self._fit_uniform_fallback(data)
            result.recommendation = (
                f"Lognormal rejected (s={s:.2f} > 3.0). "
                f"Using uniform distribution as fallback."
            )
else:
    result.recommendation = (
        f"Using {best.distribution_name} distribution "
        f"(n={n}, KS p={best.ks_pvalue:.4f}, AIC={best.aic:.2f})"
    )
```

### Change 3: Add Tail Validation

Add a new method to validate tail behavior:

```python
def _validate_tail_behavior(
    self,
    dist_fit: DistributionFit,
    data: np.ndarray,
    max_ratio: float = 100.0
) -> bool:
    """
    Validate that a fitted distribution doesn't have extreme tail behavior.

    Parameters
    ----------
    dist_fit : DistributionFit
        Fitted distribution to validate
    data : np.ndarray
        Original data
    max_ratio : float
        Maximum allowed ratio of max/median in test sample

    Returns
    -------
    bool
        True if distribution passes validation, False if extreme tails detected
    """
    try:
        # Create scipy distribution
        if dist_fit.distribution_name == 'lognormal':
            dist = stats.lognorm(
                s=dist_fit.parameters['s'],
                scale=dist_fit.parameters['scale']
            )
        elif dist_fit.distribution_name == 'gamma':
            dist = stats.gamma(
                a=dist_fit.parameters['a'],
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
            return True  # Unknown distribution, pass by default

        # Sample test data
        test_sample = dist.rvs(size=10000, random_state=42)

        # Check tail behavior
        max_val = np.max(test_sample)
        median_val = np.median(test_sample)

        if median_val > 0:
            ratio = max_val / median_val
            if ratio > max_ratio:
                logger.warning(
                    f"{dist_fit.distribution_name} has extreme tail: "
                    f"max/median = {ratio:.1f}x > {max_ratio}x"
                )
                return False

        return True

    except Exception as e:
        logger.debug(f"Tail validation failed: {e}")
        return True  # Pass by default if validation fails
```

Then use it in `fit_single()`:

```python
# After selecting best fit
result.best_fit = fitted_dists[0]

# Validate tail behavior
if not self._validate_tail_behavior(result.best_fit, data, max_ratio=100.0):
    # Try next best distribution
    for alt_dist in fitted_dists[1:]:
        if self._validate_tail_behavior(alt_dist, data, max_ratio=100.0):
            result.best_fit = alt_dist
            result.recommendation = (
                f"Best fit rejected due to extreme tails. "
                f"Using {alt_dist.distribution_name} instead."
            )
            break
    else:
        # All distributions failed - use uniform as safest fallback
        result.best_fit = self._fit_uniform_fallback(data)
        result.recommendation = (
            f"All distributions show extreme tails. "
            f"Using uniform distribution as safe fallback."
        )
```

### Change 4: Add Uniform Fallback

Add a method to create a safe uniform distribution:

```python
def _fit_uniform_fallback(self, data: np.ndarray) -> DistributionFit:
    """
    Create a uniform distribution as safe fallback.

    Uses [min, max] of data with small padding.
    """
    min_val = np.min(data)
    max_val = np.max(data)

    # Add 10% padding
    range_val = max_val - min_val
    loc = max(0, min_val - 0.1 * range_val)  # Don't go negative
    scale = max_val - loc + 0.1 * range_val

    # Create distribution fit
    dist = stats.uniform(loc=loc, scale=scale)

    # Calculate goodness of fit
    ks_stat, ks_pval = stats.kstest(data, lambda x: dist.cdf(x))

    # Calculate AIC (simple version)
    n = len(data)
    k = 2  # 2 parameters (loc, scale)
    log_likelihood = np.sum(dist.logpdf(data))
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    return DistributionFit(
        distribution_name='uniform',
        parameters={'loc': loc, 'scale': scale, 'min': loc, 'max': loc + scale},
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
        ad_statistic=0.0,
        ad_critical_value=0.0,
        aic=aic,
        bic=bic,
        n_samples=n,
        fitting_method='fallback'
    )
```

---

## Summary of Changes

### File: `src/distribution_fitting.py`

**1. Class constants** (lines ~165-167):
```python
# Remove or set to 2
MIN_SAMPLES_FIT = 2  # Changed from 3
RECOMMENDED_MIN_SAMPLES = 2  # Changed from 5
MAX_LOGNORMAL_SHAPE = 3.0  # NEW: Prevent extreme tails
MAX_TAIL_RATIO = 100.0  # NEW: Max allowed max/median ratio
```

**2. Method `fit_single()`** (lines ~278-334):
- Remove early return for n < 3
- Add special case for n = 1
- Always set `use_parametric = True`
- Remove K-S test rejection
- Add tail validation
- Add fallback to uniform if all distributions fail

**3. New method `_validate_tail_behavior()`**:
- Check max/median ratio on test sample
- Reject distributions with ratio > 100

**4. New method `_fit_uniform_fallback()`**:
- Create safe uniform distribution
- Used when all other distributions fail validation

**5. Method `sample()`** (lines ~113-129):
- Remove the `else` branch that does empirical sampling
- Should never be reached now that `use_parametric` is always True
- But keep as safety fallback with warning

```python
def sample(self, n: int = 1, random_state: Optional[int] = None) -> np.ndarray:
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
```

---

## Expected Impact

### Before Changes

```
Total materials: 266
- Using parametric: 34 (13%)
- Using empirical: 232 (87%)
```

**Example: Gas + Cement** (n=4):
- Current: Empirical (bootstrap from [7.35, 10.08, 11.40, 15.64])
- Samples: Only those 4 discrete values

### After Changes

```
Total materials: 266
- Using parametric: 266 (100%)
- Using empirical: 0 (0%)
```

**Example: Gas + Cement** (n=4):
- New: Lognormal (or gamma/uniform if lognormal rejected)
- Samples: Continuous distribution fitted to [7.35, 10.08, 11.40, 15.64]
- Can sample intermediate values: 8.5, 12.2, 14.7, etc.

### Materials That Will Change

**Small samples (n < 5)**: ~232 materials
- Currently using empirical bootstrap
- Will now use fitted parametric distributions
- Expect smoother, continuous distributions

**Failed K-S tests**: Some materials with n ≥ 5
- Currently using empirical
- Will now use parametric with tail validation

**Extreme lognormals**: ~4 Cement combinations (+ Nickel, Copper)
- Currently using lognormal with s > 10
- Will be rejected by tail validation
- Will fall back to second-best distribution or uniform

---

## Testing After Changes

### 1. Check All Materials Use Parametric

```python
from src.data_ingestion import load_all_data
from src.distribution_fitting import DistributionFitter

data = load_all_data('data/intensity_data.csv', 'data/StdScen24_annual_national.csv')
fitter = DistributionFitter()
fitted_dists = fitter.fit_all(data['intensity'])

n_parametric = sum(1 for d in fitted_dists.values() if d.use_parametric)
n_empirical = sum(1 for d in fitted_dists.values() if not d.use_parametric)

print(f"Parametric: {n_parametric} (should be {len(fitted_dists)})")
print(f"Empirical: {n_empirical} (should be 0)")

assert n_empirical == 0, "Some materials still using empirical!"
```

### 2. Check No Extreme Tails

```python
problematic = []
for (tech, mat), dist_obj in fitted_dists.items():
    sample = dist_obj.sample(100000, random_state=42)
    ratio = np.max(sample) / np.median(sample)
    if ratio > 100:
        problematic.append((tech, mat, ratio))

print(f"Materials with extreme tails: {len(problematic)} (should be 0)")
for tech, mat, ratio in sorted(problematic, key=lambda x: x[2], reverse=True):
    print(f"  {tech} + {mat}: {ratio:.1f}×")
```

### 3. Run Full Simulation

```bash
python examples/run_simulation.py
```

Check outputs:
- All materials should have CV < 200%
- No mean values > 10¹⁵
- Median values should be reasonable

### 4. Visual Diagnostic

```bash
python examples/check_distribution_fitting.py
```

Verify:
- All plots show smooth continuous distributions (not discrete)
- No extreme outliers in Monte Carlo samples

---

## Implementation Priority

**High Priority** (must fix):
1. ✓ Remove empirical fallback (always use parametric)
2. ✓ Add tail validation (prevent extreme lognormals)
3. ✓ Add uniform fallback (when all distributions fail)

**Medium Priority** (nice to have):
4. Add shape parameter bounds for lognormal
5. Improve small-n fitting (n=2, n=3)

**Low Priority** (optional):
6. Add bandwidth control for very small samples
7. Add expert judgment bounds per material

---

**Document Created**: January 27, 2026
**Status**: Implementation Required
**Priority**: High - User requirement to always use parametric distributions
