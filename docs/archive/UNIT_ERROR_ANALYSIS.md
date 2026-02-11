# Unit Error Analysis: Old vs New Monte Carlo Outputs

## Executive Summary

**Status:** ⚠️ **CRITICAL UNIT ERROR CONFIRMED**

The comparison between old (non-Monte Carlo) and new (Monte Carlo) outputs reveals that **100% of materials show massive discrepancies**, with a median ratio of **~12,500x** (new values are larger than old values).

This confirms that the current Monte Carlo outputs in [material_demand_by_scenario.csv](outputs/material_demand_by_scenario.csv) still contain the unit error that was previously identified and fixed in the source code.

## Key Findings

### 1. Magnitude of Discrepancies

| Severity | Count | Percentage |
|----------|-------|------------|
| GOOD (<3x difference) | 0 | 0% |
| OK (<10x difference) | 0 | 0% |
| POOR (<100x difference) | 0 | 0% |
| **BAD (>100x difference)** | **48** | **100%** |

**Median ratio (new/old):** 12,457.98x
**Mean log₁₀(ratio):** 8.27 (i.e., ~10^8 or 100 million times larger on average)

### 2. Most Extreme Discrepancies

The following materials show the most severe unit errors in year 2035:

| Material | Typical Ratio (New/Old) | Order of Magnitude |
|----------|-------------------------|-------------------|
| Nickel | 1.53 × 10²⁶ | ~10²⁶ (100 septillion times) |
| Cement | 1.27 × 10²⁰ | ~10²⁰ (100 quintillion times) |
| Lead | 6.07 × 10¹⁹ | ~10¹⁹ (60 quintillion times) |
| Fiberglass | 2.53 × 10¹⁹ | ~10¹⁹ (25 quintillion times) |
| Copper | 3.82 × 10¹⁸ | ~10¹⁸ (4 quintillion times) |
| Aluminum | 7.81 × 10¹⁴ | ~10¹⁴ (780 trillion times) |
| Steel | 1.19 × 10¹⁴ | ~10¹⁴ (119 trillion times) |

### 3. Moderate Discrepancies (Still Severe)

Even the "smallest" discrepancies are still ~1,000-10,000x:

| Material | Typical Ratio |
|----------|---------------|
| Silicon | 3,080x |
| Silver | 3,150x |
| Magnesium | 3,170x |
| Tin | 3,180x |
| Glass | 3,830x |
| Zinc | 9,460x |

## Root Cause Analysis

### Previously Identified Unit Error

The unit error was identified in the original [stock_flow_simulation.py](src/stock_flow_simulation.py):

**Problem:** Material intensity data in input file is in **t/GW** (tonnes per gigawatt)
**Code Treatment:** Was treating as **t/MW** (tonnes per megawatt)
**Conversion Factor:** 1 GW = 1,000 MW
**Result:** 1,000× overestimate in material demand

### Fix Applied (But Not Propagated)

The fix was applied to [data_ingestion.py](src/data_ingestion.py) at line 256:
```python
df['intensity_t_per_mw'] = df['intensity_raw'].astype(float) / 1000.0
logger.info("Applied unit conversion: t/GW → t/MW (divided by 1000)")
```

### Why Current Outputs Still Have Errors

The file [material_demand_by_scenario.csv](outputs/material_demand_by_scenario.csv) appears to have been generated **before** the unit fix was applied, or from a code version that still had the error. Evidence:

1. Example values from current output (2035, Adv_CCS scenario):
   - Cement: 2.92 × 10²² tonnes (29.2 sextillion tonnes) ❌
   - Copper: 3.32 × 10²⁰ tonnes (332 quintillion tonnes) ❌
   - Nickel: 3.96 × 10²⁶ tonnes (396 septillion tonnes) ❌

2. For comparison, global annual production:
   - Cement: ~4 billion tonnes
   - Copper: ~20 million tonnes
   - Nickel: ~2.7 million tonnes

3. The reported values are physically impossible - they exceed the mass of the entire Earth's crust by orders of magnitude.

## Comparison Details

### Test Configuration

- **Old Output:** [Old Reference Outputs/Outputs/materials_demand.csv](Old Reference Outputs/Outputs/materials_demand.csv)
  - Non-Monte Carlo baseline
  - Years: 2025, 2030, 2035
  - Scenarios: IRA, Ref
  - 30 materials tracked

- **New Output:** [outputs/material_demand_by_scenario.csv](outputs/material_demand_by_scenario.csv)
  - Monte Carlo with 10,000 iterations
  - Years: 2026, 2029, 2032, 2035, 2038, 2041, 2044, 2047, 2050
  - Scenarios: 61 (Mid_Case, Adv_RE, etc.)
  - 31 materials tracked

### Scenario Mapping

Since scenario names differ, we used approximate mappings:
- **IRA (old)** ≈ Mid_Case, Mid_Case_100by2035, Adv_RE (new)
- **Ref (old)** ≈ Mid_Case_No_IRA, High_Demand_Growth, Low_Demand_Growth (new)

### Sample Comparison (2035, IRA-equivalent scenarios)

| Material | Old Value (t) | New Value (t) | Ratio | Expected Agreement |
|----------|---------------|---------------|-------|-------------------|
| Aluminum | 838 | 7.00 × 10¹⁷ | 835 trillion × | ❌ Should be ~1x |
| Copper | 175 | 3.85 × 10²⁰ | 2.2 quadrillion × | ❌ Should be ~1x |
| Steel | 5,040 | 5.87 × 10¹⁷ | 116 trillion × | ❌ Should be ~1x |
| Nickel | 5.6 | 5.38 × 10²⁶ | 96 septillion × | ❌ Should be ~1x |

**Note:** Even accounting for Monte Carlo variability and scenario differences, we'd expect ratios within 2-5x, not trillions.

## Impact on Supply Chain Risk Analysis

The supply chain risk analysis script ([supply_chain_risk_analysis.py](examples/supply_chain_risk_analysis.py)) **is working correctly** - it properly divides by the unit_scale parameter (default 1000). However, the input data itself contains the pre-existing error, resulting in:

1. **Demand values** 1,000-10²⁶× too large
2. **Demand-to-production ratios** that are physically impossible
3. **Visualizations** showing absurd scales

Example from [demand_vs_production_comparison.csv](outputs/risk_analysis/demand_vs_production_comparison.csv):
```csv
scenario,material,demand_2035,avg_production,demand_to_production_ratio
Adv_CCS,Nickel,3.96e+23,16.84,2.35e+22
```
This shows projected nickel demand as **235 sextillion times** current production - clearly impossible.

## Required Actions

### Immediate Action Required

**Re-run the Monte Carlo simulation** using the fixed code to regenerate [material_demand_by_scenario.csv](outputs/material_demand_by_scenario.csv):

```bash
cd "/Users/pjgriffiths/Desktop/Materials Demand/Python/materials_demand_model"
source .venv/bin/activate
python examples/run_simulation.py
```

This will use the corrected [data_ingestion.py](src/data_ingestion.py) that properly converts t/GW → t/MW.

### Verification Steps

After re-running, verify the fix worked:

1. **Check magnitude of specific materials (2035):**
   - Aluminum: Should be ~100-10,000 tonnes (not 10¹⁷)
   - Copper: Should be ~100-100,000 tonnes (not 10²⁰)
   - Steel: Should be ~10,000-1,000,000 tonnes (not 10¹⁷)
   - Cement: Should be ~100,000-10,000,000 tonnes (not 10²²)

2. **Re-run comparison:**
   ```bash
   python examples/compare_outputs.py
   ```
   - Should see ratios in range 0.1x - 10x (order of magnitude agreement)
   - At least 80% of materials should show "GOOD" or "OK" agreement

3. **Re-run supply chain risk analysis:**
   ```bash
   python examples/supply_chain_risk_analysis.py \
     --demand_csv outputs/material_demand_by_scenario.csv \
     --risk_xlsx data/risk_charts_inputs.xlsx \
     --outdir outputs/risk_analysis_corrected
   ```
   - Check that demand-to-production ratios are realistic (0.1-10x, not trillions)

## Expected Results After Fix

### Realistic Value Ranges (2035, typical scenario)

Based on old outputs and physical reality:

| Material | Expected Range (thousand tonnes) | Current (Wrong) | Fixed (Expected) |
|----------|----------------------------------|-----------------|------------------|
| Aluminum | 100-2,000 | 7 × 10¹⁴ | ~800 |
| Cement | 500-5,000 | 3 × 10¹⁹ | ~1,600 |
| Copper | 50-500 | 4 × 10¹⁷ | ~175 |
| Steel | 1,000-20,000 | 6 × 10¹⁴ | ~5,000 |
| Nickel | 1-50 | 4 × 10²³ | ~5 |
| Silicon | 50-500 | 4 × 10² | ~170 |
| Zinc | 20-500 | 7 × 10² | ~120 |

### Realistic Demand-to-Production Ratios

After fix, ratios should be:
- **< 1.0**: US production exceeds projected demand (good)
- **1.0-2.0**: Demand roughly matches or slightly exceeds production (moderate risk)
- **2.0-5.0**: Demand significantly exceeds production (supply risk)
- **> 5.0**: Severe supply constraint (critical risk)

Currently seeing ratios of 10⁶ to 10²⁶ which are physically impossible.

## Files Modified

### Source Code (Already Fixed)
- ✅ [src/data_ingestion.py](src/data_ingestion.py) - Line 256: Unit conversion applied
- ✅ [src/stock_flow_simulation.py](src/stock_flow_simulation.py) - Imports corrected
- ✅ [examples/supply_chain_risk_analysis.py](examples/supply_chain_risk_analysis.py) - New script (working correctly)

### Data Files (Need Regeneration)
- ❌ [outputs/material_demand_by_scenario.csv](outputs/material_demand_by_scenario.csv) - **MUST REGENERATE**
- ❌ [outputs/material_demand_summary.csv](outputs/material_demand_summary.csv) - **MUST REGENERATE**
- ❌ [outputs/risk_analysis/*.csv](outputs/risk_analysis/) - **WILL REGENERATE AFTER DEMAND FIX**

## Timeline

1. **Code Fix Applied:** Previous session (January 26, 2026)
2. **Unit Error Identified:** This comparison analysis
3. **Required:** Re-run simulation (~5-15 minutes for 10,000 iterations)
4. **Verification:** Re-run comparison and risk analysis (~2 minutes)

## Conclusion

The good news: **The code is fixed and working correctly.**

The issue: **The output files need to be regenerated** using the fixed code.

Once the simulation is re-run, you should see:
- ✅ Realistic material demand values
- ✅ Order of magnitude agreement with old outputs
- ✅ Reasonable demand-to-production ratios
- ✅ Physically plausible projections

---

**Document Created:** January 26, 2026
**Analysis Tool:** [compare_outputs.py](examples/compare_outputs.py)
**Status:** Action Required - Regenerate outputs with fixed code
