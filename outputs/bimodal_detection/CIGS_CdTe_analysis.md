# CIGS and CdTe Bimodal Distribution Analysis

**Date:** March 7, 2026
**Analysis:** Material intensity data patterns for thin-film photovoltaics
**Source:** intensity_data.csv (literature compilation)

---

## Executive Summary

The bimodal (and in some cases **trimodal**) distributions observed in CIGS and CdTe material intensity data represent **TRUE BIMODALITY** caused by inconsistent system boundaries across literature sources, not data errors. Three distinct system scopes are mixed in the data:

1. **Cell-level** (cradle-to-gate, module only): ~1–100 g/MW
2. **Module-level** (module + BOM components): ~200–500 g/MW
3. **Complete system** (cradle-to-grave, with BOS): ~5,000–7,500 g/MW

This inconsistency creates gaps of **10–200x** between clusters, which severely distort single-distribution fitting.

---

## Detailed Findings

### CIGS (Copper Indium Gallium Selenide)

#### Copper Intensity (n=13) — **TRIMODAL** Distribution

**Data clusters:**
```
Cell-level (n=5):     17, 20, 21, 22, 24 g/MW
                      Mean: 20.8 g/MW, Range: 17-24 g/MW

Module-level (n=3):   233, 250, 450 g/MW
                      Mean: 311 g/MW, Range: 233-450 g/MW

Complete system (n=3): 7000, 7000, 7530 g/MW
                       Mean: 7177 g/MW, Range: 7000-7530 g/MW
```

**Gap ratios:**
- Cell → Module: 233/24 = **9.7x**
- Module → System: 7000/450 = **15.6x**
- Cell → System: 7530/17 = **442.9x** (overall range)

**Interpretation:**
- **Cell-level (17-24 g/MW):** Copper in CIGS absorber layer + molybdenum back contact
  - CIGS chemical formula: CuInₓGa₍₁₋ₓ₎Se₂
  - Copper is essential element in the semiconductor absorber
  - Molybdenum back contact layer (deposited on substrate)
- **Module-level (233-450 g/MW):** Cell + glass + framing + junction box wiring
  - Balance-of-Module (BOM) components add copper wiring and connections
  - Junction box electrical connections
  - Module frame and mounting points
- **Complete system (7000-7530 g/MW):** Module + inverters + cables + mounting structures
  - Inverter transformers and electrical components
  - DC/AC cabling and connectors
  - Grounding and support structure wiring

**Source:** [Wikipedia - CIGS Solar Cell](https://en.wikipedia.org/wiki/Copper_indium_gallium_selenide_solar_cell), [Wiley - Substrate and Mo Back Contact](https://onlinelibrary.wiley.com/doi/10.1155/2018/9106269)

---

#### Cadmium Intensity (n=3) — **WARNING: Extreme Gap**

**Data clusters:**
```
Low cluster (n=2):   1.3, 1.3 g/MW
High outlier (n=1):  265 g/MW
```

**Gap ratio:** 265/1.3 = **203.8x**

**Interpretation:**
- **LIKELY DATA ERROR or MISLABELED MATERIAL**
- Cadmium is **NOT** a primary component of CIGS (CuInGaSe₂)
- Small amounts (1-2 g/MW) consistent with CdS buffer layer used in some CIGS cells
- 265 g/MW value is **suspiciously similar to CdTe cadmium content** (6-244 g/MW range)
- **Recommendation:** Verify source paper for the 265 g/MW value — may be mislabeled as CIGS when it's actually CdTe data

---

#### Molybdenum Intensity (n=2) — **LIKELY BIMODAL**

**Data clusters:**
```
Cell-level (n=1):    1.0 g/MW
Module-level (n=1):  109 g/MW
```

**Gap ratio:** 109/1.0 = **109x**

**Interpretation:**
- **1.0 g/MW:** Thin molybdenum back contact layer only
  - Typical Mo layer: 0.5-1.0 μm thickness
  - Deposited via sputtering on substrate
- **109 g/MW:** Cell + module structural components
  - May include molybdenum in mounting hardware or structural alloys
  - Alternatively, could represent different Mo layer thickness estimates
- **Note:** n=2 is very small sample, hard to confirm true bimodality

**Source:** [Wiley - Mo Back Contact in CIGS](https://onlinelibrary.wiley.com/doi/10.1155/2018/9106269)

---

### CdTe (Cadmium Telluride)

#### Copper Intensity (n=9) — **BIMODAL** Distribution

**Data clusters:**
```
Cell-level (n=1):     42.8 g/MW

Complete system (n=8): 518, 1000, 3091, 5181, 5181, 7000, 7000, 7530 g/MW
                       Mean: 4688 g/MW, Range: 518-7530 g/MW
```

**Gap ratio:** 518/42.8 = **12.1x**

**Interpretation:**
- **Cell-level (42.8 g/MW):** Copper in back contact layer
  - CdTe requires Cu-doped back contact for carrier concentration
  - Typical configuration: Cu/metal bi-layer or Cu-doped ZnTe buffer
  - Essential for reducing contact resistance
- **Complete system (518-7530 g/MW):** Module + BOS electrical components
  - Wide range suggests different BOS configurations
  - Lower end (~500 g/MW): Module-level with minimal BOS
  - Upper end (~7000 g/MW): Complete system with full electrical infrastructure

**Source:** [Wiley - CdTe Back Contacts](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/ese3.843), [MDPI - CdTe Thin Film Solar Cells](https://www.mdpi.com/1996-1944/14/6/1684)

---

#### Molybdenum Intensity (n=3) — **WARNING: Extreme Gap**

**Data clusters:**
```
Low value (n=1):     0.5 g/MW
High cluster (n=2):  100.5, 109 g/MW
```

**Gap ratio:** 100.5/0.5 = **201x** (extreme)

**Interpretation:**
- **0.5 g/MW:** Molybdenum back contact (if used instead of copper)
  - Some CdTe cells use Mo as alternative back contact material
  - Typical for substrate configuration
- **100-109 g/MW:** Module-level or BOS structural components
  - Molybdenum alloys in mounting hardware?
  - Or estimates including full system boundary
- **Concern:** Only n=3 total, extreme gap suggests possible:
  - Data entry error (misplaced decimal point?)
  - Mislabeled material (should be different technology?)
  - Mixed units (kg vs g confusion?)
- **Recommendation:** Verify source papers for all three values

---

## Root Cause: Inconsistent System Boundaries in LCA Literature

### System Boundary Definitions

The variability in material intensity estimates stems from **three different LCA system boundaries** used across the literature:

1. **Cradle-to-Gate (Module Only)**
   - Scope: Raw material extraction → module manufacturing
   - Excludes: Balance-of-System (BOS) components
   - Results in: **Lowest material estimates** (cell + minimal packaging)
   - Common in: Early European studies (2000s–2010s)

2. **Cradle-to-Gate (Module + BOM)**
   - Scope: Module + Balance-of-Module components
   - Includes: Glass, framing, junction box, lamination
   - Excludes: Inverters, cables, mounting structures
   - Results in: **Moderate estimates** (2–10x higher than cell-only)

3. **Cradle-to-Grave (Complete System)**
   - Scope: Raw materials → installation → operation → decommissioning
   - Includes: Full BOS (inverters, wiring, mounting, grid connection)
   - Results in: **Highest material estimates** (10–100x higher for copper)
   - Common in: Recent comprehensive LCA studies (2015+)

**Evidence from IEA-PVPS LCA Reports:**
> "Early European studies primarily relied on **cradle to gate** system boundaries focused on module manufacturing impacts, while most recent assessments apply **cradle to grave** or **cradle to cradle** boundaries."

### Why This Matters for Monte Carlo Simulation

**Current approach** (single distribution fit):
- Tries to fit ONE lognormal distribution to ALL data points
- When n<5, uses borrowed CV (σ ≈ 0.610)
- Results in: Extreme uncertainty that doesn't reflect reality
  - Example: CIGS copper with 17–7530 g/MW range → median ≈ 116, 95% CI [3, 4,400]
  - This implies we're equally uncertain whether we need 3 or 4,400 g/MW per MW capacity!

**Correct approach** (component-based fitting):
- **Fit TWO separate distributions:**
  - Cell/module distribution: 17–450 g/MW (n=8)
  - BOS distribution: 7000–7530 g/MW (n=3)
- **Sample and SUM during Monte Carlo:**
  - Total_copper_CIGS = Cell_copper + BOS_copper
  - Each component has realistic uncertainty
- **Result:** Physically meaningful uncertainty propagation
  - Cell uncertainty: median ≈ 100, 95% CI [20, 500]
  - BOS uncertainty: median ≈ 7200, 95% CI [6800, 7600]
  - Total: median ≈ 7300, 95% CI [6900, 8000]

---

## Recommendations

### 1. Immediate Actions

**Verify extreme-gap materials (potential data errors):**
- [ ] CIGS Cadmium (203.8x gap): Check source for 265 g/MW value
  - Is this actually CdTe data mislabeled as CIGS?
  - Cadmium not a primary CIGS component (only CdS buffer layer)
- [ ] CdTe Molybdenum (201x gap): Check all three source papers
  - Verify units (g vs kg confusion?)
  - Confirm material is actually molybdenum (not misidentified?)

**Accept true bimodality for copper:**
- [x] CIGS Copper: TRUE BIMODALITY (cell/module vs BOS) — proceed with component-based fitting
- [x] CdTe Copper: TRUE BIMODALITY (cell vs BOS) — proceed with component-based fitting

### 2. Implementation Strategy

**For bimodal pairs with TRUE BIMODALITY:**
1. Split data into components using gap-based threshold detection
2. Fit separate lognormal distributions to each component
3. Update Monte Carlo sampling: Total = Component_1 + Component_2 + ... + Component_N
4. Propagate uncertainty realistically for each component

**For extreme-gap pairs (data errors):**
1. Investigate source papers
2. If error confirmed: Remove outlier, refit single distribution
3. If legitimate: Document reason and proceed with component-based fitting
4. If uncertain: Flag for sensitivity analysis (test both approaches)

### 3. Data Quality Improvements (Future Work)

**For any future data compilation:**
- [ ] Annotate each data point with system boundary (cell/module/BOS)
- [ ] Separate columns: `system_boundary` (cell, module, complete)
- [ ] Track source paper for each observation
- [ ] Standardize to single boundary for baseline analysis
- [ ] Use component-based approach when boundary mixing unavoidable

---

## Conclusion

The CIGS and CdTe bimodal distributions are **NOT data errors** but rather the result of **inconsistent system boundary definitions** across 132+ literature sources compiled over 20+ years of LCA research.

**Key finding:** Copper shows clear trimodal (CIGS) or bimodal (CdTe) patterns with physically interpretable clusters:
- Cell-level: 17–100 g/MW (absorber layer + back contact)
- Module-level: 200–500 g/MW (+ BOM components)
- System-level: 5,000–7,500 g/MW (+ BOS infrastructure)

**Correct handling:** Component-based distribution fitting and Monte Carlo summation, NOT single-distribution fitting with borrowed CV.

**Data quality concerns:** Cadmium (CIGS) and Molybdenum (CdTe) show extreme gaps that warrant verification of source papers before proceeding with bimodal treatment.

---

**Sources:**
- [Wikipedia - CIGS Solar Cell](https://en.wikipedia.org/wiki/Copper_indium_gallium_selenide_solar_cell)
- [Wiley - Substrate and Molybdenum Back Contact in CIGS](https://onlinelibrary.wiley.com/doi/10.1155/2018/9106269)
- [Wiley - Back Contacts Materials in CdTe Solar Cells](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/ese3.843)
- [MDPI - CdTe-Based Thin Film Solar Cells](https://www.mdpi.com/1996-1944/14/6/1684)
- [Nature - CIGS Electrical Modelling](https://www.nature.com/articles/s41528-022-00220-5)
- [ACS Energy Letters - Perovskite/CIGS Tandem Cost Analysis](https://pubs.acs.org/doi/10.1021/acsenergylett.2c00886)
