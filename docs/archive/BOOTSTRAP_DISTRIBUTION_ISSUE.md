# Bootstrap Distribution Issue

## Current Behavior (Incorrect)

When a material has insufficient data for parametric fitting (n < 5), the current code at [distribution_fitting.py:129](src/distribution_fitting.py#L129) does:

```python
def sample(self, n: int = 1, random_state: Optional[int] = None):
    if self.use_parametric and self.best_fit is not None:
        # Sample from parametric distribution
        dist = self._get_scipy_distribution(self.best_fit)
        return dist.rvs(size=n)
    else:
        # ❌ CURRENT: Direct bootstrap from raw data
        return np.random.choice(self.raw_data, size=n, replace=True)
```

### Problem

With this approach, each Monte Carlo sample is **constrained to be exactly one of the original data points**.

**Example: Aluminum (ASIGE)** with raw data `[15.39, 15.39, 31.3]`:
- Monte Carlo draws can ONLY be: 15.39, 15.39, or 31.3
- No interpolation or smoothing occurs
- The distribution is discrete, not continuous

**Result in 10,000 Monte Carlo iterations**:
```
Value     Count    Frequency
15.39     ~6,667   ~67%
31.3      ~3,333   ~33%
```

This is **correct** for representing the empirical distribution, but doesn't match the desired behavior.

---

## Desired Behavior (Correct)

The user wants:

1. **Generate bootstrap samples** from the raw data (e.g., 1,000 resamples with replacement)
2. **Fit a distribution** to the bootstrap samples (e.g., kernel density estimate or parametric fit)
3. **Sample from the fitted distribution** during Monte Carlo

This creates a **smooth continuous distribution** that reflects the uncertainty in the data.

### Approach 1: Bootstrap + Parametric Fit

```python
def sample(self, n: int = 1, random_state: Optional[int] = None):
    if self.use_parametric and self.best_fit is not None:
        # Sample from parametric distribution
        dist = self._get_scipy_distribution(self.best_fit)
        return dist.rvs(size=n)
    else:
        # ✓ NEW: Bootstrap + fit distribution
        # Generate bootstrap samples
        n_bootstrap = 1000
        bootstrap_samples = np.random.choice(
            self.raw_data,
            size=n_bootstrap,
            replace=True
        )

        # Fit a simple distribution (e.g., uniform or KDE)
        # For small n, uniform over [min, max] is reasonable
        lower = np.min(bootstrap_samples)
        upper = np.max(bootstrap_samples)

        # Sample from uniform distribution
        return np.random.uniform(lower, upper, size=n)
```

**Issues**:
- Still limited to range of original data
- Doesn't add much beyond simple uniform[min, max]

### Approach 2: Kernel Density Estimation (Recommended)

```python
from scipy.stats import gaussian_kde

def sample(self, n: int = 1, random_state: Optional[int] = None):
    if self.use_parametric and self.best_fit is not None:
        # Sample from parametric distribution
        dist = self._get_scipy_distribution(self.best_fit)
        return dist.rvs(size=n)
    else:
        # ✓ NEW: Use Kernel Density Estimation
        # KDE creates a smooth continuous distribution from discrete data
        kde = gaussian_kde(self.raw_data)
        samples = kde.resample(size=n, seed=random_state)
        return samples.flatten()
```

**Advantages**:
- ✓ Creates smooth continuous distribution
- ✓ Can interpolate between data points
- ✓ Automatically handles bandwidth selection
- ✓ Well-established statistical method

**Example: Aluminum [15.39, 15.39, 31.3]**:
- KDE creates a bimodal distribution with peaks at 15.39 and 31.3
- Can sample values like 18.5, 22.7, etc. (between the peaks)
- Maintains the overall shape and variance

### Approach 3: Bootstrap + Jittering

```python
def sample(self, n: int = 1, random_state: Optional[int] = None):
    if self.use_parametric and self.best_fit is not None:
        # Sample from parametric distribution
        dist = self._get_scipy_distribution(self.best_fit)
        return dist.rvs(size=n)
    else:
        # ✓ NEW: Bootstrap with jittering
        # Sample from raw data
        base_samples = np.random.choice(self.raw_data, size=n, replace=True)

        # Add small random noise (jittering)
        std_est = np.std(self.raw_data, ddof=1)
        jitter_scale = std_est * 0.1  # 10% of std as jitter
        jitter = np.random.normal(0, jitter_scale, size=n)

        return base_samples + jitter
```

**Advantages**:
- ✓ Simple to implement
- ✓ Creates continuous distribution
- ✓ Can control jitter amount

**Disadvantages**:
- ⚠️  Arbitrary choice of jitter scale
- ⚠️  Can generate negative values (need to clip)

---

## Comparison of Approaches

### Current (Direct Bootstrap)

**Example: 10,000 draws from [15.39, 15.39, 31.3]**

```
Histogram:
        |
 6667   |██████
        |
        |
        |
 3333   |        ███
        |
        +------------
          15.39  31.3
```

**Characteristics**:
- Discrete values only
- Exact frequencies match data proportions
- No interpolation

### KDE (Recommended)

**Example: 10,000 draws from [15.39, 15.39, 31.3]**

```
Histogram:
        |   ▄▄▄
 2000   |  █████▄
        |  ███████▄
        |  █████████▄
 1000   | ████████████▄▄
        | ████████████████▄▄▄
        +------------------------
          12   20   28   36
```

**Characteristics**:
- Smooth continuous distribution
- Bimodal with peaks at 15.39 and 31.3
- Can sample intermediate values
- Bandwidth automatically chosen

### Jittering

**Example: 10,000 draws from [15.39, 15.39, 31.3]**

```
Histogram:
        |
 2000   |  ▒▒▒
        |  ███
        | ████▒
 1000   | ████▒       ▒▒▒
        | ████▒      ████
        +-----------------
          13  17  21  29  33
```

**Characteristics**:
- Continuous but "fuzzy" discrete peaks
- Controllable smoothing
- Simple implementation

---

## Recommendation

**Use Kernel Density Estimation (KDE)** for empirical distributions:

```python
from scipy.stats import gaussian_kde

class MaterialIntensityDistribution:
    # ... existing code ...

    def sample(self, n: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample from the distribution.

        - For parametric fits: sample from fitted distribution
        - For empirical (n < 5): use Kernel Density Estimation
        """
        if random_state is not None:
            np.random.seed(random_state)

        if self.use_parametric and self.best_fit is not None:
            # Sample from parametric distribution
            dist = self._get_scipy_distribution(self.best_fit)
            return dist.rvs(size=n)
        else:
            # Use Kernel Density Estimation for smooth empirical distribution
            kde = gaussian_kde(self.raw_data)
            samples = kde.resample(size=n, seed=random_state)

            # Clip to non-negative (material intensities must be >= 0)
            samples = np.maximum(samples.flatten(), 0)

            return samples
```

### Why KDE?

1. ✓ **Statistically rigorous**: Well-established method for density estimation
2. ✓ **Automatic bandwidth selection**: Uses Scott's or Silverman's rule
3. ✓ **Smooth interpolation**: Creates continuous distribution from discrete data
4. ✓ **Preserves shape**: Maintains multimodality and variance
5. ✓ **No negative values**: Can clip after sampling

### Potential Issue: Bandwidth for Small n

For very small n (e.g., n=2 or n=3), KDE bandwidth might be too large, over-smoothing the distribution.

**Solution**: Add minimum bandwidth constraint

```python
# Calculate bandwidth
kde = gaussian_kde(self.raw_data, bw_method='scott')

# For small n, limit bandwidth to avoid over-smoothing
if len(self.raw_data) < 5:
    data_range = np.max(self.raw_data) - np.min(self.raw_data)
    max_bandwidth = data_range * 0.15  # 15% of range
    if kde.factor * np.std(self.raw_data) > max_bandwidth:
        kde.set_bandwidth(bw_method=max_bandwidth / np.std(self.raw_data))
```

---

## Impact on Current Results

### Aluminum (ASIGE) - n=3

**Current (Direct Bootstrap)**:
- Samples: Only [15.39, 15.39, 31.3]
- Mean: 20.69 t/MW
- Median: 15.39 t/MW

**With KDE**:
- Samples: Continuous distribution, bimodal around 15.39 and 31.3
- Mean: ~20.7 t/MW (similar)
- Median: ~15.4-18 t/MW (slightly higher due to smoothing)
- Can sample values like 17.5, 22.3, 28.1, etc.

**Expected difference**: Minimal (~5% change in mean/median) because the raw data is already well-behaved.

### Gas + Cement - n=4

**Current (Direct Bootstrap)**:
- Samples: Only [7.35, 10.08, 11.40, 15.64]
- Mean: 11.12 t/MW
- Median: 10.08 t/MW

**With KDE**:
- Samples: Continuous distribution between ~7-16 t/MW
- Mean: ~11.1 t/MW (similar)
- Median: ~10.5-11 t/MW
- Can sample values like 8.7, 12.5, 14.2, etc.

**Expected difference**: Minimal for well-behaved materials.

### Impact on Problematic Materials

The KDE change will **NOT fix the extreme outlier problem** for materials using parametric lognormal with s > 10. That requires the shape parameter bounds documented in [DISTRIBUTION_FITTING_ROOT_CAUSE.md](DISTRIBUTION_FITTING_ROOT_CAUSE.md).

However, KDE will:
- ✓ Create more realistic uncertainty representation for small-n materials
- ✓ Allow Monte Carlo to explore intermediate values
- ✓ Better match the intended "smooth distribution" behavior

---

## Implementation Steps

1. **Modify `MaterialIntensityDistribution.sample()` method** in [distribution_fitting.py](src/distribution_fitting.py):
   - Import `from scipy.stats import gaussian_kde`
   - Replace direct bootstrap with KDE sampling
   - Add non-negative clipping

2. **Test with diagnostic script**:
   ```bash
   python examples/check_distribution_fitting.py
   ```
   - Verify that empirical distributions now show smooth continuous histograms
   - Check that means/medians remain similar to before

3. **Re-run full simulation**:
   ```bash
   python examples/run_simulation.py
   ```
   - Compare outputs to previous run
   - Expect minimal changes for most materials
   - Slightly smoother distributions for small-n cases

4. **Verify with hand calculations**:
   ```bash
   python examples/hand_calculation.py
   ```
   - Check that ratios remain reasonable

---

## Code Change Summary

**File**: [src/distribution_fitting.py](src/distribution_fitting.py)

**Change**:
```python
# OLD (line 128-129):
else:
    # Sample from empirical distribution (with replacement)
    return np.random.choice(self.raw_data, size=n, replace=True)

# NEW:
else:
    # Use Kernel Density Estimation for smooth empirical distribution
    kde = gaussian_kde(self.raw_data, bw_method='scott')
    samples = kde.resample(size=n, seed=random_state)
    # Clip to non-negative (material intensities must be >= 0)
    return np.maximum(samples.flatten(), 0)
```

**Lines affected**: ~128-129
**Impact**: Changes sampling behavior for materials with n < 5 (empirical distributions)
**Risk**: Low - KDE is conservative and well-tested

---

## Related Issues

This change addresses the user's request for "bootstrap to form a distribution, then Monte Carlo draws from the fitted distribution."

This is **separate from** the extreme outlier issue documented in:
- [DISTRIBUTION_FITTING_ROOT_CAUSE.md](DISTRIBUTION_FITTING_ROOT_CAUSE.md) - Lognormal with s > 10
- [DISTRIBUTION_FITTING_ISSUE.md](DISTRIBUTION_FITTING_ISSUE.md) - Symptom analysis

**Both issues should be fixed**:
1. ✓ Add KDE for empirical sampling (this document)
2. ✓ Add shape parameter bounds for parametric distributions (root cause document)

---

**Document Created**: January 27, 2026
**Status**: Issue Identified - Implementation Recommended
**Priority**: Medium (affects interpretation of uncertainty, not magnitude of results)
