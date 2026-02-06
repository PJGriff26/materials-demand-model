# Distribution Visualization Tools

Two comprehensive visualization tools for inspecting fitted distributions:

## 1. inspect_distributions.py - Standard 5-Panel View

Shows comprehensive analysis of the SELECTED distribution.

**Usage:**
```bash
python inspect_distributions.py --material Cement --technology Gas
python inspect_distributions.py --material Aluminum --technology ASIGE
python inspect_distributions.py --show_all  # Find all problematic distributions
```

**Panels:**
1. **Histogram + Fitted PDF** - Shows raw data histogram with fitted distribution overlay
2. **CDF Comparison** - Empirical vs fitted CDF with K-S test results
3. **Q-Q Plot** - Quantile-quantile plot to assess goodness of fit
4. **Monte Carlo Samples** - Histogram of 10k samples from fitted distribution
5. **Tail Behavior Analysis** - Percentile breakdown and tail statistics

**Output:** `{technology}_{material}_inspection.png`

---

## 2. inspect_distributions_all_candidates.py - All Candidates View

**NEW!** Shows ALL candidate distributions that were fitted, highlighting which was selected.

**Usage:**
```bash
python inspect_distributions_all_candidates.py --material Cement --technology Gas
python inspect_distributions_all_candidates.py --material Indium --technology CdTe
```

**Layout:**

### Top Row (up to 4 panels):
- **Individual PDFs** - One subplot per candidate distribution
- Shows histogram + fitted PDF for each candidate
- **Selected distribution has:**
  - ✓ Checkmark in title
  - Green color
  - Green border around panel
  - Bold solid line

### Middle Row:
- **Left (2 columns): All CDFs on Same Plot**
  - Black empirical CDF (dashed-dot)
  - All candidate CDFs overlaid
  - Selected distribution: green, thick, solid line
  - Others: colored, thinner, dashed lines

- **Right (2 columns): Q-Q Plot**
  - For the SELECTED distribution only
  - Shows how well theoretical quantiles match empirical

### Bottom Row:
- **Left (2 columns): Comparison Table**
  - Shows AIC, BIC, K-S statistic, K-S p-value for all candidates
  - Sorted by AIC (best first)
  - Selected row highlighted in green
  - Lowest AIC/BIC cells highlighted in light green

- **Right (2 columns): Monte Carlo Samples**
  - Histogram of 10,000 samples from SELECTED distribution
  - Shows mean, median, std
  - Overlays raw data points for comparison
  - **Color-coded tail behavior warning:**
    - ✓ Green: Max/Median < 10× (healthy)
    - ℹ️  Orange: Max/Median 10-50× (moderate tail)
    - ⚠️  Red: Max/Median > 50× (large tail - warning!)

**Output:** `{technology}_{material}_all_candidates.png`

---

## Key Features

### Visual Indicators

**Selected Distribution:**
- ✓ Checkmark in panel title
- Green color throughout
- Bold/thick lines
- Green border around panel
- "SELECTED" label in legends

**Distribution Quality:**
- **AIC/BIC**: Lower is better (information criteria)
- **K-S statistic**: Lower is better (distance between CDFs)
- **K-S p-value**: Higher is better (p > 0.05 = pass)

### Tail Behavior Assessment

Both tools show tail behavior, but the all-candidates view has explicit warnings:

```
✓ Max/Median = 1.04× (healthy)      # No fat tail
ℹ️  Max/Median = 15.2× (moderate)   # Some tail, acceptable
⚠️  Max/Median = 125× (large tail)  # Problem! Should not occur with fixed code
```

---

## Examples

### Example 1: Gas + Cement (n=4)
```bash
python inspect_distributions_all_candidates.py --material Cement --technology Gas
```

**Expected to show:**
- Small sample (n=4)
- Uniform distribution likely selected (AIC-based)
- All candidates: truncated_normal, lognormal, gamma, uniform
- Max/Median ratio: ~1.4× (healthy)

### Example 2: CdTe + Indium (n=3)
```bash
python inspect_distributions_all_candidates.py --material Indium --technology CdTe
```

**Previously problematic (before fix):**
- Gamma with a=0.047, missing loc parameter
- Max/Median: 2.78×10^7× (catastrophic!)

**After fix:**
- Gamma with a=0.047, loc=0.0155 (correct)
- Max/Median: ~1.04× (healthy!)

### Example 3: Onshore Wind + Cement (n=6)
```bash
python inspect_distributions_all_candidates.py --material Cement --technology "onshore wind"
```

**Expected to show:**
- Larger sample (n=6)
- Multiple candidates with close AIC values
- Likely gamma or lognormal selected
- Can compare how different distributions fit the same data

---

## Interpreting Results

### Good Fit Indicators:
- ✓ K-S p-value > 0.05
- ✓ Q-Q plot points close to diagonal line
- ✓ Max/Median ratio < 10×
- ✓ Monte Carlo samples cover similar range as raw data
- ✓ AIC close to other candidates (no single clear winner often means uniform is safest)

### Poor Fit Indicators:
- ✗ K-S p-value < 0.05 (distribution rejected by test)
- ✗ Q-Q plot points far from diagonal
- ✗ Max/Median ratio > 50×
- ✗ Monte Carlo samples much wider than raw data range
- ✗ Extreme outliers in MC samples

### Distribution Selection Logic:
1. All candidates fitted (truncated_normal, lognormal, gamma, uniform)
2. Sorted by AIC (lower is better)
3. Check best fit for tail validation:
   - Lognormal: reject if shape s > 3.0
   - All: sample 10k, reject if max/median > 100×
4. If best passes validation: SELECT IT ✓
5. If best fails: try next-best candidate
6. If all fail: fall back to uniform (always safe)

---

## Comparison of Tools

| Feature | inspect_distributions.py | inspect_distributions_all_candidates.py |
|---------|-------------------------|----------------------------------------|
| **Focus** | In-depth analysis of selected | Compare all candidates |
| **Panels** | 5 comprehensive panels | 6 sections with comparison table |
| **Best For** | Verifying final distribution | Understanding selection process |
| **Shows** | Selected distribution only | All fitted distributions |
| **Table** | No comparison table | Yes - AIC/BIC/KS for all |
| **Use When** | "Is the fitted distribution good?" | "Why was this distribution chosen?" |

---

## Output Files

All visualizations saved to: `../outputs/distribution_inspect/`

**File naming:**
- Standard view: `{technology}_{material}_inspection.png`
- All candidates: `{technology}_{material}_all_candidates.png`

**File size:** ~700-1000 KB (high resolution, dpi=300)

---

## Troubleshooting

### "Combination not found"
Make sure the technology and material names match exactly (case-sensitive):
```bash
# WRONG:
python inspect_distributions_all_candidates.py --material cement --technology gas

# CORRECT:
python inspect_distributions_all_candidates.py --material Cement --technology Gas
```

### "No fitted distributions found"
This means `fitted_distributions` attribute is empty. Possible causes:
- Material-technology combination has no data
- Distribution fitting failed for all candidates
- Check that data files are loaded correctly

### Plots look crowded
This is expected when n is small and data points are clustered. The visualization shows:
- Raw data as red diamonds at y=0
- Red vertical lines (transparent) to show data locations
- This makes it easy to see where actual data points are vs fitted distribution

---

## Future Enhancements

Possible additions:
- [ ] Add `--compare` flag to show multiple material-technology combinations side-by-side
- [ ] Add `--distribution` filter to only show specific distribution types
- [ ] Export comparison table as CSV
- [ ] Add Anderson-Darling test results to table
- [ ] Show bootstrap confidence intervals for small n

---

**Created**: January 27, 2026
**Tools**: inspect_distributions.py, inspect_distributions_all_candidates.py
**Purpose**: Visual quality assurance for distribution fitting
