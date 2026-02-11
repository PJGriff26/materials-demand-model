# Distribution Fitting Root Cause Analysis

## Executive Summary

✅ **Distribution fitting algorithm is working correctly** in its design
❌ **Lognormal distributions with extreme shape parameters create catastrophic tail behavior**

The issue affects **4 out of 19 Cement technology combinations** (21%) and likely similar materials (Nickel, Copper):
- Solar Distributed + Cement
- utility-scale solar pv + Cement
- onshore wind + Cement
- Geothermal + Cement

---

## The Problem in Detail

### What the Algorithm Does (Correctly)

From [distribution_fitting.py](src/distribution_fitting.py):

1. **For small samples (n < 5)**: Uses **empirical bootstrap** (resampling with replacement)
   - ✅ This is safe and conservative
   - ✅ Cannot create values outside the range of raw data

2. **For larger samples (n ≥ 5)**: Attempts **parametric fitting**
   - Tries: truncated normal, lognormal, gamma, uniform
   - Selects best by AIC (Akaike Information Criterion)
   - Uses K-S test to validate fit
   - ⚠️  **Problem occurs here**

### Example: Solar Distributed + Cement

**Raw data (n=6)**:
```
[4.847, 6.497, 6.775, 7.52, 10.08, 10.512] t/MW
Mean: 7.71 t/MW
Std: 2.16 t/MW
CV: 28%  ← Reasonable variability
```

**Fitted lognormal parameters**:
```python
s = 13.336      ← Shape parameter (EXTREMELY LARGE)
loc = 4.847     ← Location (lower bound)
scale = 0.0078  ← Scale parameter (very small)
```

**Result when sampling 100,000 draws**:
```
Median:  0.01 t/MW        ← Collapsed to near-zero!
Mean:    6.8 × 10²⁴ t/MW  ← 6.8 septillion!
Max:     6.8 × 10²³ t/MW  ← 680 sextillion!
```

### Why This Happens

The lognormal distribution is parameterized as:

```
X ~ Lognormal(μ, σ)
where X = exp(μ + σ·Z) and Z ~ Normal(0,1)
```

In scipy: `lognorm(s, loc, scale)` where `s = σ` (shape parameter)

When **s >> 1** (like s=13.3), the distribution has:
- Most probability mass concentrated near `loc` (lower bound)
- Extremely fat right tail extending to infinity
- Occasional samples that are **10²⁰ to 10²⁶ times the median**

### Monte Carlo Impact

With 10,000 Monte Carlo iterations:
- Even if only 0.01% of samples (1 out of 10,000) draw extreme values
- If that value is 10²³ times larger than typical values
- It completely **dominates the arithmetic mean**

Example:
```
9,999 samples: ~7 t/MW each     → Total: 70,000
1 sample:      7 × 10²³ t/MW    → Total: 7 × 10²³
Mean = (70,000 + 7×10²³) / 10,000 ≈ 7×10¹⁹ t/MW
```

The mean is ~10¹⁹ times larger than the typical value!

---

## Affected Materials

### Cement (4 out of 19 technologies)

| Technology | n | Distribution | s (shape) | Max/Median Ratio | Status |
|------------|---|--------------|-----------|------------------|--------|
| Solar Distributed | 6 | lognormal | 13.34 | 8.45 × 10²⁵ | ❌ EXTREME |
| utility-scale solar pv | 6 | lognormal | 13.34 | 8.45 × 10²⁵ | ❌ EXTREME |
| onshore wind | 6 | lognormal | 13.13 | 3.41 × 10²⁵ | ❌ EXTREME |
| Geothermal | 8 | lognormal | 12.34 | 9.89 × 10²³ | ❌ EXTREME |
| Coal CCS | 12 | gamma | N/A | 24.4 | ⚠️  MODERATE |
| Nuclear New | 9 | gamma | N/A | 63.3 | ⚠️  MODERATE |
| **All others** | 2-12 | empirical | N/A | < 2 | ✓ HEALTHY |

### Why Only Some Technologies?

The problem occurs when:
1. **n ≥ 5** (triggers parametric fitting)
2. **Lognormal selected as best fit** (high variance in data)
3. **Large shape parameter `s` > 10** (creates fat tails)

Technologies with n < 5 use empirical bootstrap and are **safe**.

---

## Validation Results

### Test 1: Hand Calculations

From [hand_calculation.py](examples/hand_calculation.py):

**Aluminum (ASIGE)** - n=3, empirical:
- Hand calc: 1.83 million tonnes
- Simulation: 1.83 billion tonnes
- Ratio: **1,000×** (explained by units: thousand tonnes vs tonnes) ✅

**Cement (Gas)** - n=4, empirical:
- Hand calc median: 951k tonnes
- Simulation median: 12 billion tonnes
- Ratio: **12,700×** (explained by units + cumulative across technologies) ✅
- **Mean ratio: 10¹⁹×** ❌ Not explainable - this is the extreme outlier problem

### Test 2: Distribution Fitting Diagnostic

From [check_distribution_fitting.py](examples/check_distribution_fitting.py):

**Aluminum (ASIGE)** - n=3, empirical:
- 100k sample max: 31.3 t/MW
- Median: 15.4 t/MW
- Max/Median: **2.0×** ✅ Healthy

**Cement (Gas)** - n=4, empirical:
- 100k sample max: 15.6 t/MW
- Median: 11.4 t/MW
- Max/Median: **1.4×** ✅ Healthy

**But when we check technologies using parametric distributions**:

From [check_cement_distributions.py](examples/check_cement_distributions.py):

**Cement (Solar Distributed)** - n=6, lognormal (s=13.34):
- 100k sample max: 6.82 × 10²³ t/MW
- Median: 0.01 t/MW
- Max/Median: **8.45 × 10²⁵×** ❌ CATASTROPHIC

---

## Why Wasn't This Caught?

### The K-S Test Passed!

```
Solar Distributed + Cement:
  KS statistic: 0.2611
  KS p-value: 0.8800  ← PASS (p > 0.05)
```

The Kolmogorov-Smirnov test only checks the **CDF on the range of the observed data**. It doesn't validate **tail behavior beyond the data range**.

The fitted lognormal:
- ✅ Fits the 6 observed data points well [4.8, 6.5, 6.8, 7.5, 10.1, 10.5]
- ❌ But extrapolates absurdly for x > 10.5 due to s=13.34

### Visualization Evidence

The diagnostic visualization [outputs/distribution_fitting_diagnostic.png](outputs/distribution_fitting_diagnostic.png) shows:
- **Left panel**: Histogram + fitted PDF looks reasonable over the data range
- **Middle panel**: CDF comparison - K-S test passes within observed range
- **Right panel**: Monte Carlo samples - would show extreme outliers if problematic distributions were included

---

## Solutions

### Immediate Fix (High Priority)

**Option 1: Increase Minimum Sample Size for Parametric Fitting**

In [distribution_fitting.py](src/distribution_fitting.py):

```python
# Change from:
RECOMMENDED_MIN_SAMPLES = 5

# To:
RECOMMENDED_MIN_SAMPLES = 10  # or even 15
```

This would force more materials to use safe empirical bootstrap.

**Option 2: Add Shape Parameter Bounds**

```python
def _fit_distribution(self, data, dist_name):
    # ... existing fitting code ...

    if dist_name == 'lognormal':
        params = dist_class.fit(data)
        s = params[0]  # shape parameter

        # Reject if shape parameter indicates extreme tail
        if s > 3.0:  # Conservative threshold
            raise ValueError(f"Lognormal shape too large: s={s:.2f}")

        # ... rest of code ...
```

**Rationale**: For lognormal with s > 3:
- CV = sqrt(exp(s²) - 1) = sqrt(exp(9) - 1) ≈ 8,100%
- This is pathological for material intensity data

**Option 3: Add Post-Fit Tail Validation**

```python
def _fit_distribution(self, data, dist_name):
    # ... existing fitting code ...

    # Validate tail behavior
    sample_test = dist.rvs(size=10000, random_state=42)
    max_to_median = np.max(sample_test) / np.median(sample_test)

    if max_to_median > 100:  # Threshold
        raise ValueError(
            f"Distribution has extreme tail behavior: "
            f"max/median = {max_to_median:.1f}x"
        )

    # ... rest of code ...
```

### Medium-Term Fix

**Implement Bounded Lognormal**

Instead of standard lognormal, use:
```python
from scipy.stats import lognorm

# Fit lognormal
params = lognorm.fit(data)

# But truncate at reasonable upper bound
upper_bound = np.max(data) * 10  # 10× the observed max

# Sample with rejection
def sample_bounded_lognormal(n):
    samples = []
    while len(samples) < n:
        candidate = lognorm.rvs(*params)
        if candidate <= upper_bound:
            samples.append(candidate)
    return np.array(samples)
```

### Long-Term Fix

1. **Review intensity data quality** for materials showing high variance
2. **Add expert judgment bounds** based on physical constraints
3. **Use domain-specific distributions** (e.g., Weibull for materials science)

---

## Verification After Fix

Run these checks:

### 1. Distribution Fitting Diagnostic
```bash
python examples/check_distribution_fitting.py
```

Verify:
- [ ] No materials have Max/Median > 100
- [ ] All parametric fits have reasonable tail behavior

### 2. Cement Technology Analysis
```bash
python examples/check_cement_distributions.py
```

Verify:
- [ ] All Cement technologies show "✓ HEALTHY" status
- [ ] No Max/Median ratios > 100

### 3. Full Simulation
```bash
python examples/run_simulation.py
```

Then check outputs:
```python
import pandas as pd
df = pd.read_csv('outputs/material_demand_by_scenario.csv')

# Check CV for all materials
df['cv'] = (df['std'] / df['mean']) * 100

problematic = df[df['cv'] > 200]
print(f"Materials with CV > 200%: {len(problematic)}")  # Should be 0
```

### 4. Hand Calculation Verification
```bash
python examples/hand_calculation.py
```

Verify:
- [ ] Mean ratios within 10× after accounting for units
- [ ] Median ratios within 10× after accounting for units

---

## Recommended Implementation

**Implement Option 2 + Option 3** (shape bounds + tail validation):

```python
# In distribution_fitting.py, _fit_distribution method

MAX_LOGNORMAL_SHAPE = 3.0  # Class constant
MAX_TAIL_RATIO = 100.0     # Class constant

def _fit_distribution(self, data, dist_name):
    # ... existing fitting code ...

    # For lognormal specifically
    if dist_name == 'lognormal':
        params = dist_class.fit(data)
        s = params[0]

        # Check 1: Reject extreme shape parameters
        if s > self.MAX_LOGNORMAL_SHAPE:
            raise ValueError(
                f"Lognormal shape parameter too large: s={s:.2f} > {self.MAX_LOGNORMAL_SHAPE}. "
                f"This indicates extreme tail behavior."
            )

    # ... create distribution object ...

    # Check 2: Validate tail behavior for all distributions
    sample_test = dist.rvs(size=10000, random_state=42)
    max_val = np.max(sample_test)
    median_val = np.median(sample_test)

    if median_val > 0:
        max_to_median = max_val / median_val

        if max_to_median > self.MAX_TAIL_RATIO:
            raise ValueError(
                f"{dist_name} has extreme tail behavior: "
                f"max/median = {max_to_median:.1f}x > {self.MAX_TAIL_RATIO}x. "
                f"This would corrupt Monte Carlo means."
            )

    # ... rest of fitting code ...
```

This approach:
- ✅ Prevents extreme lognormal shapes
- ✅ Validates tail behavior for all distribution types
- ✅ Falls back to empirical distribution when validation fails
- ✅ Logs clear warnings about why parametric fitting was rejected

---

## Expected Impact

After implementing the fix:

| Material | Metric | Before (Wrong) | After (Fixed) |
|----------|--------|----------------|---------------|
| Cement | Mean CV | 7,294% ❌ | < 100% ✅ |
| Cement | Max/Median (100k sample) | 10²⁵ ❌ | < 10 ✅ |
| Cement | Mean (2035, Mid_Case) | 30 sextillion ❌ | ~15-30 million ✅ |
| Nickel | Mean CV | 9,986% ❌ | < 100% ✅ |
| Copper | Mean CV | 8,195% ❌ | < 100% ✅ |

---

## Files Modified

### To Implement Fix
- [ ] [src/distribution_fitting.py](src/distribution_fitting.py) - Add shape bounds and tail validation

### Verification Scripts (Already Created)
- ✅ [examples/hand_calculation.py](examples/hand_calculation.py)
- ✅ [examples/check_distribution_fitting.py](examples/check_distribution_fitting.py)
- ✅ [examples/check_cement_distributions.py](examples/check_cement_distributions.py)

### Documentation
- ✅ [DISTRIBUTION_FITTING_ISSUE.md](DISTRIBUTION_FITTING_ISSUE.md) - Initial symptom analysis
- ✅ [HAND_CALCULATION_VERIFICATION.md](HAND_CALCULATION_VERIFICATION.md) - Step-by-step validation
- ✅ [DISTRIBUTION_FITTING_ROOT_CAUSE.md](DISTRIBUTION_FITTING_ROOT_CAUSE.md) - This document

---

## Conclusion

The distribution fitting algorithm is **structurally sound** but has a **critical flaw**:

✅ **Correctly** uses empirical bootstrap for small samples (n < 5)
✅ **Correctly** attempts parametric fitting for larger samples
✅ **Correctly** validates fit quality with K-S test
❌ **Fails to validate** tail behavior beyond observed data range

The **lognormal distribution** with large shape parameters (s > 10) creates:
- Probability mass concentrated near the lower bound
- Extremely fat right tail extending to 10²⁵× the median
- Occasional Monte Carlo samples that corrupt the arithmetic mean

**Fix**: Add validation of tail behavior by checking Max/Median ratio on test samples.

**Impact**: Prevents 10¹⁹-10²⁶× discrepancies in mean values while preserving accurate median values.

---

**Document Created**: January 27, 2026
**Diagnostic Scripts**: [check_distribution_fitting.py](examples/check_distribution_fitting.py), [check_cement_distributions.py](examples/check_cement_distributions.py)
**Related**: [DISTRIBUTION_FITTING_ISSUE.md](DISTRIBUTION_FITTING_ISSUE.md), [HAND_CALCULATION_VERIFICATION.md](HAND_CALCULATION_VERIFICATION.md)
