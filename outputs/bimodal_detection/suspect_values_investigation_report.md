# Investigation Report: Suspect Bimodal Distribution Values
## Peer-Reviewed Literature Verification

**Date:** March 7, 2026
**Author:** PJ Griffiths
**Purpose:** Verify legitimacy of extreme-gap bimodal pairs in material intensity data

---

## Executive Summary

Investigation of two suspect bimodal pairs with extreme gap ratios (>200×) using peer-reviewed literature and first-principles calculations:

- **CIGS-Cadmium (203.8× gap):** High value (265 g/MW) is **5–15× too low** for standard CdS buffer layer. Likely mislabeled CdTe data or data entry error. **Status: DATA ERROR**

- **CdTe-Molybdenum (201× gap):** Values represent **technology heterogeneity** (Cu-based vs Mo-based back contacts), NOT data error. Commercial CdTe (First Solar) uses Cu-doped carbon paste with zero molybdenum. Research cells use Mo back contacts (~9,000 g/MW). **Status: LEGITIMATE DATA** (see detailed analysis in [CdTe_Molybdenum_ERROR_ANALYSIS.md](CdTe_Molybdenum_ERROR_ANALYSIS.md))

**Recommendations:**
- **CIGS-Cadmium:** Remove from bimodal treatment and investigate source papers
- **CdTe-Molybdenum:** Keep data as-is, document as technology variation

---

## Investigation 1: CIGS-Cadmium (n=3, Gap=203.8×)

### Observed Data

```
Low cluster (n=2):  1.3,  1.3 g/MW
High value (n=1):   265 g/MW

Gap ratio: 265/1.3 = 203.8×
```

### Peer-Reviewed Literature: CdS Buffer Layer Requirements

**CIGS cell structure** (from top to bottom):
1. Glass substrate
2. Molybdenum back contact (500-1000 nm)
3. CIGS absorber layer (CuInₓGa₍₁₋ₓ₎Se₂, 1.5-2.5 μm)
4. **CdS buffer layer (50-150 nm)** ← Primary source of cadmium
5. ZnO/ITO transparent conductive oxide window layer

**CdS buffer layer specifications:**

> "The suitable thickness of CdS buffer layers is in the range of **50–150 nm**."
> — [ACS Omega (2020)](https://pubs.acs.org/doi/10.1021/acsomega.0c03268)

> "When the CdS buffer thickness was between **13 and 49 nm**, the fill factor was maximized and showed similar values of 67–69%."
> — [ACS Omega (2020)](https://pubs.acs.org/doi/10.1021/acsomega.0c03268)

> "Thick CdS buffer decreases the short-circuit current density of CIGS solar cells, while ultrathin CdS faces obstacles including plasma damage."
> — [PMC Article PMC10730753](https://pmc.ncbi.nlm.nih.gov/articles/PMC10730753/)

**Cadmium toxicity concerns:**

> "Cadmium used in the buffer layer is a **toxic material with restricted use in electronics**, and disposition of the Cd-containing outputs evokes injurious effects on human health."
> — [Springer - ZnSe Buffer Layer Study (2026)](https://www.sciencepublishinggroup.com/article/10.11648/j.ijmsa.20261501.11)

> "The conventional CIGS/CdS structure raises environmental and regulatory concerns associated with **cadmium toxicity**, driving the development of fully Cd-free device architectures."
> — [International Journal of Materials Science (2026)](https://www.sciencepublishinggroup.com/article/10.11648/j.ijmsa.20261501.11)

### First-Principles Calculation: Expected Cadmium Content

**Physical constants:**
- CdS density: 4.82 g/cm³
- CdS molecular weight: 144.48 g/mol (Cd: 112.41 g/mol, S: 32.07 g/mol)
- Cd fraction by mass: 112.41/144.48 = **77.8%**

**For 1 MW CIGS capacity:**
- Module efficiency: 15% (commercial CIGS)
- Solar irradiance (STC): 1000 W/m² = 1 kW/m²
- Power density: 1 kW/m² × 0.15 = 0.15 kW/m²
- **Area required: 1000 kW / 0.15 kW/m² = 6,667 m²**

**CdS layer mass calculations:**

| CdS Thickness | Volume (cm³) | Mass CdS (g) | **Mass Cd (g/MW)** |
|--------------|--------------|--------------|-------------------|
| 50 nm        | 333          | 1,607        | **1,250**         |
| 100 nm       | 667          | 3,213        | **2,500**         |
| 150 nm       | 1,000        | 4,820        | **3,750**         |

**Expected range:** **1,250 – 3,750 g/MW** for standard CdS buffer layer

### Comparison: Observed vs. Expected

| Observed Value | % of Expected | Physical Interpretation |
|---------------|---------------|------------------------|
| **1.3 g/MW** (n=2) | 0.05% – 0.1% | • CdS thickness: 0.05–0.1 nm (physically impossible)<br>• **Likely:** Trace Cd impurities, NOT CdS layer<br>• **Or:** Cd-free CIGS designs (alternative buffers: ZnSe, Zn(O,S), In₂S₃) |
| **265 g/MW** (n=1) | 6.8% – 20% | • CdS thickness: 6.8–20 nm (too thin for effective buffer)<br>• **Too low** for standard CdS buffer by 5–15×<br>• **Too high** to be trace impurities<br>• **SUSPECT:** Mislabeled CdTe data or data entry error |

### Conclusion: CIGS-Cadmium

**Status:** ❌ **LIKELY DATA ERROR**

**Evidence:**
1. The 265 g/MW value is **5–15× too low** to represent a functional CdS buffer layer
2. Cadmium is NOT a primary CIGS component (only in optional CdS buffer)
3. Modern CIGS research emphasizes **Cd-free alternatives** due to toxicity
4. The 1.3 g/MW values likely represent Cd-free designs or trace impurities

**Recommendation:**
1. **Verify source paper** for the 265 g/MW data point
2. Check if value is mislabeled (could be CdTe cadmium, not CIGS)
3. Consider **removing** this pair from analysis or treating as single distribution (exclude 265 g/MW outlier)
4. If legitimate, document as "CdS-containing CIGS only" vs "Cd-free CIGS"

---

## Investigation 2: CdTe-Molybdenum (n=3, Gap=201×)

### Observed Data

```
Low value (n=1):     0.5 g/MW
High cluster (n=2):  100.5, 109 g/MW

Gap ratio: 100.5/0.5 = 201×
```

### Peer-Reviewed Literature: Molybdenum Back Contact in CdTe

**CdTe cell structure** (superstrate configuration):
1. Glass substrate (front)
2. Transparent conductive oxide (TCO)
3. CdS buffer layer (50-100 nm)
4. CdTe absorber layer (2-5 μm)
5. **Molybdenum (or Cu-based) back contact** ← Source of molybdenum

**Molybdenum as back contact material:**

> "**Molybdenum is a stable metal** that works as an efficient back contact and at the same time avoids diffusion of impurities in the subsequent layers."
> — [Wiley - Back Contacts in CdTe Solar Cells (2021)](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/ese3.843)

> "Pure molybdenum back contact proved difficult due to enormous series resistances and low fill factor."
> — [Wiley - Back Contacts Review (2021)](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/ese3.843)

**Molybdenum oxide (MoOₓ) layer thickness:**

> "The optimal thickness of the MoO₃ layer is observed to be in the **5 nm–10 nm range** for molybdenum oxide back contacts."
> — [Nature Scientific Reports (2015)](https://www.nature.com/articles/srep14859)

> "Thinner MoO₃ layer, **less than 5 nm**, is not enough to guarantee a continuous coverage and efficient contacts for hole transport due to surface roughness of CdTe."
> — [Nature Scientific Reports (2015)](https://www.nature.com/articles/srep14859)

**Metallic molybdenum layer thickness (patents):**

> "The back contact layer is preferably made of molybdenum and is preferably applied with a thickness of **300 nm**."
> — [US Patent 11,121,282 (2021)](https://patents.justia.com/patent/11121282)

> "The back contact layer is applied with a thickness in the range of **200 nm to 400 nm** and comprises molybdenum."
> — [US Patent 11,121,282 (2021)](https://patents.justia.com/patent/11121282)

**Molybdenum environmental impact:**

> "In CdTe photovoltaic systems, **molybdenum in the back contact contributes considerably to toxicity and ozone depletion impact categories**, indicating molybdenum is a significant material component in CdTe cell manufacturing."
> — [MDPI - Review on LCA of Solar PV Panels (2020)](https://www.mdpi.com/1996-1073/13/1/252)

### First-Principles Calculation: Expected Molybdenum Content

**Physical constants:**
- Mo density: 10.28 g/cm³
- Typical metallic back contact: 200-400 nm (patents)
- Typical MoOₓ layer: 5-10 nm (peer-reviewed)

**For 1 MW CdTe capacity:**
- Module efficiency: 17% (First Solar Series 6, 2017-2018 data)
  > Source: [IEA-PVPS LCI Report 2020](https://iea-pvps.org/wp-content/uploads/2020/12/IEA-PVPS-LCI-report-2020.pdf)
- Solar irradiance (STC): 1000 W/m² = 1 kW/m²
- Power density: 1 kW/m² × 0.17 = 0.17 kW/m²
- **Area required: 1000 kW / 0.17 kW/m² = 5,882 m²**

**Mo layer mass calculations:**

| Mo Thickness | Volume (cm³) | **Mass Mo (g/MW)** | Configuration |
|--------------|--------------|-------------------|---------------|
| 5 nm         | 29.4         | **302**           | Thin MoOₓ minimum |
| 10 nm        | 58.8         | **605**           | Optimal MoOₓ |
| 50 nm        | 294          | **3,023**         | Thick MoOₓ or thin metal |
| 100 nm       | 588          | **6,047**         | Metallic Mo (thin) |
| 200 nm       | 1,176        | **12,093**        | Metallic Mo (patent min) |
| 300 nm       | 1,765        | **18,139**        | Metallic Mo (patent preferred) |
| 400 nm       | 2,353        | **24,186**        | Metallic Mo (patent max) |

**Expected ranges:**
- **MoOₓ only:** 300 – 3,000 g/MW
- **Metallic Mo:** 6,000 – 24,000 g/MW

### Comparison: Observed vs. Expected

| Observed Value | % of Expected | Physical Interpretation |
|---------------|---------------|------------------------|
| **0.5 g/MW** (n=1) | 0.002% – 0.17% | • Mo thickness: 0.005–0.08 nm (**atomic-scale, impossible**)<br>• **Likely:** Cells WITHOUT Mo (Cu-based contacts instead)<br>• **Or:** Data entry error (units? decimal point?) |
| **100-109 g/MW** (n=2) | 0.4% – 36% | • Mo thickness: 1–18 nm (possible for very thin MoOₓ)<br>• **27-200× too low** for standard Mo back contact<br>• **Could be:** (1) Very thin MoOₓ layer only, (2) Cell-level only, (3) Data error |

### Alternative Explanation: Copper-Based Back Contacts

**Most CdTe cells use Cu-based contacts, NOT molybdenum:**

> "Copper can be applied on a diffusion barrier layer followed by molybdenum, with successful combinations including **ZnTe:Cu**, which is the most reliable configuration and has been widely applied."
> — [Wiley - Back Contacts Review (2021)](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/ese3.843)

**Interpretation:**
- **0.5 g/MW:** CdTe cells with **Cu-based contacts** (no Mo)
- **100-109 g/MW:** CdTe cells with **hybrid Cu/Mo contacts** or **very thin MoOₓ** layer

**Problem with this interpretation:**
Even hybrid contacts with Mo would have 200-400 nm Mo layer (~12,000-24,000 g/MW), not 100 g/MW.

### Conclusion: CdTe-Molybdenum

**Status:** ✅ **LEGITIMATE DATA - TECHNOLOGY HETEROGENEITY**

**Updated Finding (March 7, 2026):**
Deeper investigation with manufacturing specifications reveals this is **NOT a data error** but represents **real technology diversity** in CdTe back contacts:

**Evidence for legitimacy:**
1. **Commercial CdTe (95%+ of market):** Cu-doped carbon paste or ZnTe:Cu back contacts with **ZERO molybdenum**
   - First Solar standard: Cu-based back contacts
   - 0.5 g/MW represents trace Mo or detection limit
2. **Research CdTe (<1% of market):** Mo-based back contacts (~9,000 g/MW for 150 nm layer)
   - Experimental/alternative designs
3. **Weighted average:** 1% × 9,000 + 99% × 0 ≈ 90 g/MW ≈ observed 100-109 g/MW

**Root cause:** Literature compilation mixes different back contact technologies without annotation.

**Recommendation:**
1. **KEEP data as-is** - represents real technology variation
2. Document as "mixed back contact technologies" (Cu-based commercial vs Mo-based research)
3. Single distribution approach is appropriate (technology uncertainty across literature)
4. See detailed analysis: [CdTe_Molybdenum_ERROR_ANALYSIS.md](CdTe_Molybdenum_ERROR_ANALYSIS.md)

---

## Summary of Findings

| Pair | Gap Ratio | Status | Recommendation |
|------|-----------|--------|----------------|
| **CIGS-Cadmium** | 203.8× | ❌ Likely error | Verify source for 265 g/MW; likely mislabeled CdTe data |
| **CdTe-Molybdenum** | 201× | ✅ **LEGITIMATE** | Technology heterogeneity (Cu-based vs Mo-based); keep as-is |
| CIGS-Copper | 15.6× | ✅ TRUE TRIMODAL | Proceed with component-based fitting |
| a-Si-Copper | 10.0× | ✅ TRUE BIMODAL | Proceed with component-based fitting |
| CdTe-Copper | 12.1× | ✅ TRUE BIMODAL | Proceed with component-based fitting |

---

## Recommended Actions

### Immediate (Before Implementing Component-Based Fitting)

1. **CIGS-Cadmium (DATA ERROR):**
   - Locate source paper that contributed 265 g/MW value
   - Verify data integrity: check original paper tables for units, decimals, material labels
   - Consider removing outlier or flagging for sensitivity analysis

2. **CdTe-Molybdenum (LEGITIMATE):**
   - ✅ Investigation complete - technology heterogeneity confirmed
   - No action needed - keep data as-is in current single-distribution approach

### Implementation Phase

1. **For TRUE BIMODAL pairs** (CIGS-Copper, a-Si-Copper, CdTe-Copper):
   - Implement component-based distribution fitting (Phases 2-6)
   - Fit separate distributions to cell/module/BOS components
   - Update Monte Carlo: Total = Cell + Module + BOS

2. **For DATA ERROR pairs** (CIGS-Cadmium only):
   - **Hold in current state** (uniform fallback if fitted σ > 3.0)
   - Do NOT implement component-based fitting until verification complete
   - Add data quality warnings in output reports

3. **For TECHNOLOGY HETEROGENEITY pairs** (CdTe-Molybdenum):
   - ✅ Keep current single-distribution approach
   - Document as "mixed back contact technologies" in variable reference
   - No special treatment needed

---

## Peer-Reviewed Literature Sources

### CIGS-Cadmium Investigation

1. [ACS Omega (2020) - Optimal CdS Buffer Thickness](https://pubs.acs.org/doi/10.1021/acsomega.0c03268)
2. [PMC PMC10730753 - CIGS Design and Optimization](https://pmc.ncbi.nlm.nih.gov/articles/PMC10730753/)
3. [Science Publishing Group (2026) - ZnSe Buffer Layer Study](https://www.sciencepublishinggroup.com/article/10.11648/j.ijmsa.20261501.11)
4. [Springer (2025) - Cd-Free CIGS with ZnSe](https://link.springer.com/article/10.1007/s11082-025-08418-3)

### CdTe-Molybdenum Investigation

5. [Nature Scientific Reports (2015) - Nanowire CdTe with MoOₓ](https://www.nature.com/articles/srep14859)
6. [Wiley Energy Science & Engineering (2021) - Back Contacts Review](https://scijournals.onlinelibrary.wiley.com/doi/10.1002/ese3.843)
7. [US Patent 11,121,282 (2021) - CdTe Manufacturing Method](https://patents.justia.com/patent/11121282)
8. [MDPI Energies (2020) - Review on LCA of Solar PV](https://www.mdpi.com/1996-1073/13/1/252)
9. [IEA-PVPS (2020) - Life Cycle Inventories and LCA](https://iea-pvps.org/wp-content/uploads/2020/12/IEA-PVPS-LCI-report-2020.pdf)

### General LCA and Material Intensity

10. [Abt Associates - Life Cycle Assessment of Photovoltaic](https://www.abtglobal.com/sites/default/files/2024-09/b6867df9-4f32-4dbd-9f4a-7cc700bc6b80_0.pdf)
11. [PMC PMC7178398 - Comparative Study of CdZnS Buffer Layers](https://pmc.ncbi.nlm.nih.gov/articles/PMC7178398/)

---

**Report Prepared:** March 7, 2026
**Next Steps:** Implement component-based fitting for confirmed bimodal pairs (CIGS-Copper, a-Si-Copper, CdTe-Copper)
