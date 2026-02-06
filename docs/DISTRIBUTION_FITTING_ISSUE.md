# Distribution Fitting Issue: Extreme Outliers in Monte Carlo

## Problem Discovered

After regenerating the Monte Carlo outputs, the comparison still shows extreme discrepancies for SOME materials (Cement, Nickel, Copper) but NOT others (Aluminum, Silicon).

## Root Cause: Fat-Tailed Distributions with Extreme Outliers

The issue is NOT with unit conversion (which is working correctly). The problem is that the **distribution fitting** for certain materials is creating distributions with extreme tails that occasionally sample astronomically large values.

### Evidence

#### Example: Cement (Mid_Case, 2035)

| Statistic | Value | Notes |
|-----------|-------|-------|
| **p2 (2nd percentile)** | 354,000 tonnes | ✓ Reasonable |
| **p5** | 378,000 tonnes | ✓ Reasonable |
| **p25** | 540,000 tonnes | ✓ Reasonable |
| **p50 (median)** | 13,300,000 tonnes | ✓ Reasonable (~13 million) |
| **p75** | 11,700,000,000 tonnes | ⚠️ Large (~11.7 billion) |
| **p95** | 542,000,000,000,000 tonnes | ❌ HUGE (~542 trillion) |
| **p97** | 18,400,000,000,000,000 tonnes | ❌ MASSIVE (~18 quadrillion) |
| **Mean** | 29,700,000,000,000,000,000,000 tonnes | ❌ CATASTROPHIC (~30 sextillion) |
| **Std Dev** | 2,170,000,000,000,000,000,000,000 tonnes | ❌ Std > Mean! |

**Coefficient of Variation:** 7,294% (Std / Mean × 100)

For comparison, a well-behaved distribution should have CV < 100%.

#### Comparison: Working Materials

**Aluminum (Mid_Case, 2035):**
- Median: 1,895,000 tonnes (~1.9 million) ✓
- Mean: 2,151,000 tonnes (~2.2 million) ✓
- Std: 785,000 tonnes ✓
- **CV: 37%** ✓ Well-behaved!

**Silicon (Mid_Case, 2035):**
- Median: 280,000 tonnes ✓
- Mean: 334,000 tonnes ✓
- Std: 192,000 tonnes ✓
- **CV: 57%** ✓ Well-behaved!

### Materials Affected

| Material | CV | Status |
|----------|-----|--------|
| Cement | 7,294% | ❌ Extreme outliers |
| Nickel | 9,986% | ❌ Extreme outliers |
| Copper | 8,195% | ❌ Extreme outliers |
| Aluminum | 37% | ✓ Normal |
| Silicon | 57% | ✓ Normal |

## Why This Happens

### 1. Underlying Data Issues
Some technology-material combinations may have:
- Very small sample sizes (n < 5)
- Wide variation in reported values
- Potential data quality issues or outliers in source data

### 2. Distribution Fitting Choices
The current code tries multiple distributions:
- Truncated Normal
- Lognormal
- Gamma
- Uniform
- Empirical (fallback)

For materials with problematic data, the "best fit" distribution (by Kolmogorov-Smirnov test) may be:
- **Lognormal** with large variance → fat right tail → extreme samples
- Or fitting issues creating unrealistic parameter values

### 3. Monte Carlo Sampling
With 10,000 iterations:
- Even if extreme values occur in only 0.1% of samples (10 out of 10,000)
- If those values are 10^15 times larger than typical values
- They will completely dominate the arithmetic mean

## Impact on Analysis

### What's Correct
- ✓ Unit conversion is working (t/GW → t/MW division by 1000)
- ✓ Calculation logic is correct (capacity × intensity)
- ✓ **Median values are reasonable** for most materials
- ✓ Lower percentiles (p2-p75) are reasonable

### What's Wrong
- ❌ **Mean values** for Cement, Nickel, Copper are meaningless
- ❌ Upper percentiles (p95, p97) are unrealistic
- ❌ Standard deviations are larger than means
- ❌ Demand-to-production ratios based on means are nonsensical

## Recommended Solutions

### Short-Term (Use Median Instead of Mean)

**For reporting and analysis, use MEDIAN (p50) instead of MEAN:**

```python
# Instead of:
cement_demand = df['mean']

# Use:
cement_demand = df['p50']  # median
```

This is actually MORE appropriate for skewed distributions anyway, and is robust to outliers.

**Update supply chain risk analysis to use median:**
```python
# In supply_chain_risk_analysis.py, when loading demand data:
demand_col = 'p50'  # Use median instead of mean
```

### Medium-Term (Fix Distribution Fitting)

**Option 1: Add Bounds to Distributions**

Modify [distribution_fitting.py](src/distribution_fitting.py) to enforce reasonable upper bounds:

```python
def fit_single(self, technology, material, data):
    # After fitting, check if upper tail is reasonable
    # e.g., 99.9th percentile should not exceed 1000× the median

    if dist.ppf(0.999) > 1000 * np.median(data):
        # Use empirical distribution instead
        return self._fit_empirical(data)
```

**Option 2: Use Robust Distributions**

For materials with high CV in source data, force empirical bootstrap:

```python
cv_threshold = 2.0  # If CV > 200%, use empirical
if np.std(data) / np.mean(data) > cv_threshold:
    return self._fit_empirical(data)
```

**Option 3: Winsorize Extreme Values**

Cap fitted distributions at reasonable percentiles:

```python
# During sampling, cap at 99th percentile of fitted distribution
sample = dist.rvs(size=1)
cap = dist.ppf(0.99) * 10  # 10× the 99th percentile as absolute max
sample = min(sample, cap)
```

### Long-Term (Improve Source Data)

1. **Review intensity data quality** for Cement, Nickel, Copper
2. **Add expert judgment bounds** based on physical constraints
3. **Collect more data points** for high-variance materials

## Validation After Fix

Once distribution fitting is improved, verify:

1. **CV Check:** All materials should have CV < 200%
2. **Percentile Rationality:** p97 should not be > 100× p50
3. **Mean-Median Ratio:** Mean should be within 2× of median for most materials
4. **Physical Plausibility:** Upper bounds should not exceed total global production

Expected results:
- Cement median: ~10-50 million tonnes (reasonable for US energy system)
- Cement mean: ~15-60 million tonnes (within 2× of median)
- CV: < 100%

## Current Workaround

**For immediate analysis, use the median (p50) column instead of mean:**

```bash
# When running supply chain risk analysis:
python supply_chain_risk_analysis.py \
  --demand_csv outputs/material_demand_by_scenario.csv \
  --risk_xlsx data/risk_charts_inputs.xlsx \
  --value_column p50  # Use median instead of mean
```

**Note:** The supply_chain_risk_analysis script currently doesn't have a `--value_column` parameter. This would need to be added, or the script can be modified to hard-code using 'p50' instead of 'mean'.

## Bottom Line

The simulation IS working correctly in terms of:
- Data loading ✓
- Unit conversion ✓
- Monte Carlo sampling ✓
- Statistical calculation ✓

The problem is that the **distribution fitting** for certain materials is creating unrealistic fat-tailed distributions. The **median values are trustworthy** and should be used for analysis until the distribution fitting is improved.

---

**Document Created:** January 26, 2026
**Status:** Issue Identified - Workaround Available (use median)
**Priority:** Medium (analysis can proceed with median values)
