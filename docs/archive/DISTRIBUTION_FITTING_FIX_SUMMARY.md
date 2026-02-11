# Distribution Fitting Fix Summary

**Date**: January 27, 2026
**Issue**: Extreme tail behavior in Monte Carlo simulations (max/median ratios up to 10^7×)
**Root Cause**: Missing `loc` parameter when creating scipy distributions for sampling
**Status**: ✓ FIXED

---

## The Problem

When running the materials demand simulation, some materials showed catastrophically large values:

**Example: CdTe + Indium**
- Raw data: [0.0155, 0.0155, 0.0159] t/MW
- Expected max/median: ~1.02× (nearly identical values)
- **ACTUAL max/median: 2.78×10^7×** (27 million times!)

This occurred for 29 out of 169 material-technology combinations.

---

## Root Cause Analysis

### The Bug

In [distribution_fitting.py:152-157](src/distribution_fitting.py#L152-L157), the `_get_scipy_distribution()` method was creating distributions for sampling:

```python
# BEFORE (INCORRECT):
elif fit.distribution_name == 'lognormal':
    return stats.lognorm(s=params['s'], scale=params['scale'])
elif fit.distribution_name == 'gamma':
    return stats.gamma(a=params['a'], scale=params['scale'])
```

**Problem**: The `loc` parameter was NOT passed, so it defaulted to `loc=0`.

### Why This Caused Extreme Tails

For the CdTe + Indium example:
- **Fitted parameters**: a=0.047, loc=0.0155, scale=0.000084
- **What should happen**: Distribution centered around 0.0155 with small variance
- **What actually happened**: Distribution centered around 0 (loc defaulted to 0)

When a gamma distribution with very small shape parameter (a<<1) is centered at 0:
- Most samples are near 0 (median ~ 10^-11)
- But occasional samples reach the "natural" scale (max ~ 0.0006)
- Result: max/median = 2.78×10^7

### Why Only Some Materials Were Affected

The issue affected:
- **Gamma distributions** with small shape parameters (a < 1)
- **Lognormal distributions** (also uses loc parameter)
- **Uniform and truncated_normal** were not affected (already included loc)

---

## The Fix

### Code Change

Updated [distribution_fitting.py:152-162](src/distribution_fitting.py#L152-L162):

```python
# AFTER (CORRECT):
elif fit.distribution_name == 'lognormal':
    return stats.lognorm(
        s=params['s'],
        loc=params.get('loc', 0),  # ← NOW INCLUDED
        scale=params['scale']
    )
elif fit.distribution_name == 'gamma':
    return stats.gamma(
        a=params['a'],
        loc=params.get('loc', 0),  # ← NOW INCLUDED
        scale=params['scale']
    )
```

### Verification

**Before fix:**
```
Total materials with max/median > 100: 29
Worst case (CdTe + Indium): 2.78×10^7×
```

**After fix:**
```
Total materials with max/median > 100: 0 ✓
Worst case: 86.17× (healthy!)
Max/median distribution:
  - Median: 1.58×
  - 95th percentile: 29.25×
  - 99th percentile: 77.57×
  - Max: 86.17×
```

**All distributions now pass tail validation.**

---

## Impact on Simulation Results

### Expected Changes

**Before fix** (with extreme tails):
- Some materials had means > 10^15 tonnes
- Coefficient of variation > 10,000%
- Means completely dominated by rare extreme outliers
- **Medians were still reasonable** (not affected by outliers)

**After fix**:
- All CV values should be < 200%
- No means > 10^12 tonnes
- Means and medians should agree within 2-5×
- Results should match order of magnitude with old non-Monte Carlo study

### Testing

To verify the fix worked:

```bash
cd examples

# 1. Check all distributions are parametric
python -c "
import sys; sys.path.insert(0, '../src')
from data_ingestion import load_all_data
from distribution_fitting import DistributionFitter
data = load_all_data('../data/intensity_data.csv', '../data/StdScen24_annual_national.csv')
fitter = DistributionFitter()
fitted_dists = fitter.fit_all(data['intensity'])
n_para = sum(1 for d in fitted_dists.values() if d.use_parametric)
print(f'Parametric: {n_para}/{len(fitted_dists)} (expect 100%)')
"

# 2. Check tail behavior
python -c "
import sys, numpy as np
sys.path.insert(0, '../src')
from data_ingestion import load_all_data
from distribution_fitting import DistributionFitter
data = load_all_data('../data/intensity_data.csv', '../data/StdScen24_annual_national.csv')
fitter = DistributionFitter()
fitted_dists = fitter.fit_all(data['intensity'])
ratios = []
for dist_obj in fitted_dists.values():
    sample = dist_obj.sample(100000, random_state=42)
    ratios.append(np.max(sample) / np.median(sample))
print(f'Max ratio: {max(ratios):.2f}× (should be < 100)')
print(f'Materials > 100: {sum(1 for r in ratios if r > 100)} (should be 0)')
"

# 3. Run full simulation
python run_simulation.py

# 4. Check output quality
python -c "
import pandas as pd
df = pd.read_csv('../outputs/material_demand_by_scenario.csv')
df['cv'] = (df['std'] / df['mean']) * 100
print(f'Materials with CV > 200%: {(df.cv > 200).sum()} (should be 0)')
print(f'Max CV: {df.cv.max():.1f}% (should be < 200%)')
print(f'Max mean value: {df[\"mean\"].max():.2e} (should be < 1e12)')
"

# 5. Visualize specific distributions
python inspect_distributions.py --material Indium --technology CdTe --outdir ../outputs/distribution_inspect
python inspect_distributions.py --material Cement --technology Gas --outdir ../outputs/distribution_inspect
```

---

## Related Issues Fixed

### 1. Always Use Parametric Distributions ✓

**User requirement**: "I want all of the fits to be truncated normal, lognormal, gamma, or uniform."

**Implementation**: Modified [distribution_fitting.py](src/distribution_fitting.py) to:
- Set `MIN_SAMPLES_FIT = 2` (was 3)
- Always set `use_parametric = True`
- For n=1: use narrow uniform around single point
- For n≥2: fit parametric distribution
- Validate tail behavior before accepting fit
- Fall back to uniform if all distributions fail validation

**Result**: 100% of materials now use parametric distributions (was 13% before).

### 2. Tail Validation ✓

Added `_validate_tail_behavior()` method that:
1. Checks lognormal shape parameter s < 3.0
2. Samples 10,000 draws from fitted distribution
3. Checks max/median ratio < 100
4. Rejects distributions with extreme tails

**Note**: The validation was already implemented, but the sampling bug meant it was checking the wrong distribution (with loc=0 instead of correct loc). After fixing the loc parameter bug, the validation now works correctly.

### 3. Uniform Fallback ✓

Added `_fit_uniform_fallback()` method that creates safe uniform distribution over [min, max] with 10% padding when all parametric fits fail validation.

---

## Files Modified

### Source Code
- **[src/distribution_fitting.py](src/distribution_fitting.py)** (lines 152-162)
  - Added `loc` parameter to lognormal and gamma distribution creation

### Documentation Created
- ✓ [REQUIRED_DISTRIBUTION_CHANGES.md](REQUIRED_DISTRIBUTION_CHANGES.md) - Specification for always using parametric
- ✓ [BOOTSTRAP_DISTRIBUTION_ISSUE.md](BOOTSTRAP_DISTRIBUTION_ISSUE.md) - Analysis of bootstrap vs parametric approach
- ✓ [DISTRIBUTION_FITTING_ROOT_CAUSE.md](DISTRIBUTION_FITTING_ROOT_CAUSE.md) - Original root cause analysis (focused on lognormal shape parameters)
- ✓ [DISTRIBUTION_FITTING_FIX_SUMMARY.md](DISTRIBUTION_FITTING_FIX_SUMMARY.md) - This document

### Diagnostic Tools Created
- ✓ [examples/check_distribution_fitting.py](examples/check_distribution_fitting.py)
- ✓ [examples/check_cement_distributions.py](examples/check_cement_distributions.py)
- ✓ [examples/hand_calculation.py](examples/hand_calculation.py)
- ✓ [examples/inspect_distributions.py](examples/inspect_distributions.py) - **NEW: Comprehensive visualization tool**

---

## Visualization Tool

Created [inspect_distributions.py](examples/inspect_distributions.py) for visual inspection of fitted distributions:

**Features:**
- 5-panel visualization:
  1. Histogram + fitted PDF
  2. CDF comparison (goodness-of-fit)
  3. Q-Q plot
  4. Monte Carlo samples analysis (10k draws)
  5. Tail behavior with percentile breakdown
- Color-coded status indicators (green/orange/red)
- Command-line interface:
  - `--material <name> --technology <name>` for specific inspection
  - `--show_all` to find and plot all problematic distributions
- Saves high-resolution PNGs

**Usage:**
```bash
# Inspect specific combination
python inspect_distributions.py --material Cement --technology Gas

# Find all problematic distributions (CV > 50% or max/median > 10)
python inspect_distributions.py --show_all
```

---

## Success Criteria

- [x] All 169 materials use parametric distributions (100%)
- [x] No materials have max/median ratio > 100
- [x] 99th percentile of max/median ratios < 100
- [x] No distributions with extreme lognormal shape (s > 3)
- [x] Visualization tool works correctly
- [ ] Full simulation runs without extreme outliers (pending completion)
- [ ] Output CV values all < 200% (pending verification)
- [ ] Comparison with old outputs shows order-of-magnitude agreement (pending)

---

## Next Steps

1. **Wait for simulation to complete** - Currently running with 10,000 Monte Carlo iterations
2. **Verify output quality** - Check that all materials have reasonable CV and mean values
3. **Compare with old outputs** - Run [compare_outputs.py](examples/compare_outputs.py) to verify order-of-magnitude agreement
4. **Document final results** - Update [DISTRIBUTION_FITTING_ISSUE.md](DISTRIBUTION_FITTING_ISSUE.md) with final verification

---

**Key Takeaway**: A single missing parameter (`loc`) in distribution sampling caused extreme tail behavior for 17% of materials. The fix was simple (3 lines of code) but critical. All distributions now pass validation and the simulation should produce reasonable results.
