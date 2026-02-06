# Materials Demand Pipeline Changes — Claude Code Sessions

## Overview
This document summarizes modifications made to the materials demand analysis pipeline during Claude Code sessions. Changes span clustering analysis, feature engineering, and supply chain risk visualization.

---

## 1. Scenario Clustering with Interpretive Labels
**File:** `clustering/main_analysis.py`

Added two scenario clustering configurations with meaningful centroid labels:

| Config | VIF Threshold | Clusters | Interpretation |
|--------|---------------|----------|----------------|
| A | 50 | k=4 (enforced) | High-stress front-loaded, Baseline steady-state, Deep decarbonization, Sustained growth |
| B | 10 | silhouette-optimal | Rapid decarbonization, Constrained transition, Baseline, Sustained H2 growth, High-stress rapid decline, Low-stress balanced |

PCA biplots now display interpretive cluster names at centroids rather than numeric labels.

---

## 2. Net Import Reliance (NIR) Calculation Fix
**File:** `clustering/feature_engineering.py` (lines 139-202)

### Problem
Net exporter materials (Boron, Molybdenum) were incorrectly assigned `import_dependency = 1.0` because:
- "E" designation in USGS data was skipped → NaN → filled with 1.0

### Solution
Implemented three-tier priority system for NIR calculation:
1. **Primary:** Calculate from aggregate trade data: `NIR = (imports - exports) / consumption`
2. **Secondary:** Use import_dependency sheet; convert "E" → 0.0
3. **Tertiary:** USGS thin-film CSVs with improved text parsing

### Thin-Film NIR Text Parsing
Fixed interpretation of USGS range notation:
- `"<75"` → 37.5% (midpoint of 0-75)
- `">50"` → 75% (midpoint of 50-100)
- `"100"` → 100%

---

## 3. Supply Chain Risk Scatterplot Methodology
**File:** `examples/generate_eda_figures.py`

### Capacity Stress Denominator
| Before | After |
|--------|-------|
| `max(production, imports)` | `consumption` |

Using consumption as the denominator provides a more defensible baseline—it represents demonstrated market capacity rather than an arbitrary max of two supply channels.

### NIR Corrections
| Material | Before | After | Reason |
|----------|--------|-------|--------|
| Aluminum | 0.91 | 0.45 | USGS MCS reports ~45%; raw trade data inflated by variable definition issues |
| Boron | 0.10 | 0.0 | Net exporter |
| Molybdenum | 0.10 | 0.0 | Net exporter (NIR ≈ -200% from trade data) |
| Cadmium | 0.90 | 0.30 | Updated from thin-film USGS data |
| Tellurium | 0.90 | 0.95 | Updated from thin-film USGS data |

### Visualization Updates
- **Removed** vertical line at x=0 (no longer relevant—all displayed materials are net importers or clamped to 0)
- **Updated** y-axis label to clarify 3-year normalization: "3-yr peak demand / 3-yr consumption"
- **Reorganized** label offsets for optimal spacing across new NIR positions
- **Noted** "Gadium" typo in source data (should be Gadolinium)

### Lookup Logic Change
Qualitative overrides now take precedence over USGS-calculated values to correct known data quality issues:
```python
if m in qualitative_import_reliance:
    nir = qualitative_import_reliance[m]  # Override first
elif m in net_import_reliance.index:
    nir = net_import_reliance[m]  # USGS fallback
```

---

## Impact Summary

| Component | Key Improvement |
|-----------|-----------------|
| Material clustering | Molybdenum now clusters with other domestically-produced metals instead of high-risk imports |
| Scenario clustering | Interpretive labels enable qualitative narrative around cluster characteristics |
| Risk scatterplot | Aluminum correctly positioned at moderate import reliance; net exporters at x=0 |

---

## 4. Distribution Fitting Bug Fixes (Critical)
**File:** `src/distribution_fitting.py`

### Problem: Extreme Monte Carlo Outliers
Monte Carlo simulations produced catastrophically large values (10^19 to 10^26 tonnes) for some materials, while others were reasonable. Investigation revealed multiple distribution fitting bugs.

### Bug 1: Missing `loc` Parameter in Sampling
**Lines 152-162**

| Before | After |
|--------|-------|
| `stats.gamma(a=params['a'], scale=params['scale'])` | `stats.gamma(a=params['a'], loc=params.get('loc', 0), scale=params['scale'])` |
| `stats.lognorm(s=params['s'], scale=params['scale'])` | `stats.lognorm(s=params['s'], loc=params.get('loc', 0), scale=params['scale'])` |

**Impact:** For CdTe + Indium (data: [0.0155, 0.0155, 0.0159]):
- Before: max/median ratio = 2.78×10^7 (catastrophic)
- After: max/median ratio = 1.04 (healthy)

### Bug 2: Zero-Variance Data Handling
**Lines 396-415**

Added detection for datasets where all values are identical (std=0). These now return a dummy fit with poor AIC, forcing uniform distribution selection instead of degenerate gamma/lognormal with scale=0.

### Bug 3: Uniform Distribution with Zero Range
**Lines 454-470**

When min=max, uniform now creates a narrow distribution with 10% padding around the single value instead of a degenerate scale=0 distribution.

### Bug 4: Truncated Normal Fitting
**Lines 417-460**

| Before | After |
|--------|-------|
| Used `floc=0` with `truncnorm.fit()` | Method of Moments optimization |
| Produced mean=1.0 for data with mean=54 | Correctly matches data mean and std |

The old approach fixed `loc=0`, but `loc` in scipy's truncnorm is the mean of the underlying normal, not the truncation point.

### Validation Checks Added
- `MAX_LOGNORMAL_SHAPE = 3.0` — Rejects lognormal with extreme shape parameter
- `MAX_TAIL_RATIO = 100.0` — Rejects distributions where max/median > 100 in test sample
- Fallback to uniform if all distributions fail validation

### Results
| Metric | Before | After |
|--------|--------|-------|
| Materials with max/median > 100 | 29 | 0 |
| Worst max/median ratio | 2.78×10^7 | 86.2 |
| 99th percentile ratio | N/A | 77.6 |

---

## 5. Distribution Visualization Tools
**Files:** `examples/inspect_distributions.py`, `examples/inspect_distributions_all_candidates.py`

### Standard View (`inspect_distributions.py`)
5-panel visualization for selected distribution:
1. Histogram + Fitted PDF
2. CDF Comparison with K-S test
3. Q-Q Plot
4. Monte Carlo samples (10k draws)
5. Tail behavior analysis

### All Candidates View (`inspect_distributions_all_candidates.py`)
Shows ALL candidate distributions that were fitted:
- Top row: Individual PDFs for each candidate (truncated_normal, lognormal, gamma, uniform)
- Middle: All CDFs overlaid + Q-Q plot for selected
- Bottom: Comparison table (AIC, BIC, K-S stats) + Monte Carlo samples

**Key features:**
- Selected distribution highlighted in green with ✓ marker
- Color-coded tail warnings (green/orange/red based on max/median ratio)
- Comparison table sorted by AIC with lowest values highlighted

### Usage
```bash
# Inspect specific material-technology combination
python inspect_distributions_all_candidates.py --material Cement --technology Gas

# Standard 5-panel view
python inspect_distributions.py --material Aluminum --technology ASIGE
```

**Documentation:** `examples/README_VISUALIZATION_TOOLS.md`

---

## 6. Distribution Selection Changes

### Before Fixes
| Distribution | Count | Percentage |
|--------------|-------|------------|
| uniform | 94 | 55.6% |
| gamma | 65 | 38.5% |
| lognormal | 9 | 5.3% |
| truncated_normal | 1 | 0.6% |

### After Fixes
| Distribution | Count | Percentage |
|--------------|-------|------------|
| uniform | 99 | 58.6% |
| gamma | 65 | 38.5% |
| lognormal | 5 | 3.0% |
| truncated_normal | 0 | 0.0% |

**Key changes:**
- 4 fewer lognormal (zero-variance cases now use uniform)
- 1 fewer truncated_normal (CIGS + Indium now uses uniform — truncnorm was previously selected with wrong parameters)
- Uniform increased by 5 (absorbing the problematic cases)

---

## 7. Lognormal Fit Analysis
**Issue:** Lognormal sometimes produces "spiky" fits that look wrong but have best AIC.

**Example:** onshore wind + Cement (n=6, data: [38.7, 41.4, 47.1, 60.1, 64.2, 73.1])
- Lognormal MLE: s=13.13, scale=0.038, loc=38.7
- AIC=14.66 (best!) but mean=10^36, max/median=10^22

**Why this happens:**
- MLE puts a high PDF spike at the minimum data value (loc=38.7)
- Fat tail "covers" other points with low probability
- Mathematically maximizes likelihood, but physically nonsensical

**Solution:** Validation checks reject lognormal with s>3.0 or extreme tail ratios. Gamma (AIC=52.8) is selected instead — it has mean=53.5, std=13.3 matching the data.

---

## Files Modified

| File | Changes |
|------|---------|
| `src/distribution_fitting.py` | Fixed loc parameter, zero-variance handling, truncnorm fitting, added validation |
| `examples/inspect_distributions.py` | Created — standard 5-panel visualization |
| `examples/inspect_distributions_all_candidates.py` | Created — all candidates comparison view |
| `examples/README_VISUALIZATION_TOOLS.md` | Created — documentation for visualization tools |
| `DISTRIBUTION_FITTING_FIX_SUMMARY.md` | Created — detailed fix summary |

---

*Last updated: January 27, 2026*
