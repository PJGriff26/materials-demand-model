# AI Disclosure

This document describes the use of AI tools in the development of this research software, in accordance with responsible AI use and academic transparency standards.

---

## Model Used

**Claude** (Anthropic) — accessed through **Claude Code**, Anthropic's CLI-based agentic coding tool. Sessions spanned multiple Claude model generations (Claude 3.5 Sonnet through Claude Opus 4) between approximately October 2025 and February 2026.

---

## How AI Was Used

### 1. Code Development and Debugging

Claude Code served as an interactive programming assistant throughout the development of the Monte Carlo simulation pipeline. Specific contributions include:

- **Distribution fitting fixes** — Identified and corrected four bugs in `src/distribution_fitting.py` that caused catastrophic outliers (10^19–10^26 tonnes) in Monte Carlo output. Fixes included restoring a missing `loc` parameter in scipy distribution sampling, handling zero-variance data, correcting truncated normal fitting via method of moments, and adding validation guards (shape parameter caps, tail ratio limits).

- **Net Import Reliance (NIR) calculation** — Implemented a three-tier priority system for NIR sourcing in `clustering/feature_engineering.py`, correcting misclassification of net-exporter materials (Boron, Molybdenum) that were erroneously assigned 100% import dependency.

- **Stock-flow simulation engine** — Assisted in developing and refining `src/stock_flow_simulation.py`, which tracks capacity additions and retirements using technology-specific lifetimes.

- **Clustering and dimensionality reduction** — Helped implement Sparse PCA, NMF, and Factor Analysis pipelines in `clustering/`, including preprocessing (log transform, VIF filtering, StandardScaler) and interpretive labeling of cluster centroids.

- **Supply chain risk analysis** — Developed risk ranking visualization (`visualizations/risk_ranking_chart.py`) and supply chain analysis modules with CRC-weighted sourcing concentration metrics.

- **Visualization toolkit** — Created publication-quality figure generation scripts (`visualizations/manuscript_fig1.py`, `manuscript_figures.py`, `feature_scatterplots.py`) and diagnostic distribution inspection tools (`examples/inspect_distributions*.py`).

### 2. Documentation

Claude Code assisted in writing and maintaining:

- `PIPELINE_DOCUMENTATION.md` — Technical reference for the 10-step pipeline
- `docs/variable_reference.csv` — Master variable inventory (100+ entries)
- `docs/visualization_inventory.csv` — Visualization catalog (50+ figures)
- `CHANGELOG_CLAUDE_CODE.md` — Detailed record of AI-assisted changes
- Inline code comments and docstrings

### 3. Diagnostic Analysis

Claude Code was used to trace execution paths during debugging sessions, including:

- Walking through distribution fitting logic to isolate the missing `loc` parameter bug
- Comparing before/after Monte Carlo output statistics to validate fixes
- Analyzing USGS source data to resolve NIR discrepancies (e.g., Aluminum: 91% raw trade data vs. 45% USGS Mineral Commodity Summaries)

---

## How AI Contributions Were Verified

All AI-generated code and analysis were subject to the following verification procedures:

### Automated Testing
- **Regression test suite** (`tests/test_pipeline.py`) — Tests data loading, technology mapping, distribution fitting, and simulation output against expected values.
- **Unit validation** (`tests/validate_units.py`) — Confirms correct t/GW to t/MW conversion (division by 1000).

### Hand Calculations
- Material demand values were independently verified by manual calculation: tracing raw intensity data through unit conversion, distribution fitting, Monte Carlo sampling, and stock-flow aggregation for selected material-technology-scenario combinations.
- Before/after comparison tables were produced for every bug fix (documented in `CHANGELOG_CLAUDE_CODE.md`).

### Distribution Diagnostics
- 5-panel diagnostic visualizations (histogram + PDF, CDF with K-S test, Q-Q plot, Monte Carlo samples, tail behavior) were generated for all 169 material-technology combinations to verify distribution fit quality.
- All-candidates comparison views show every fitted distribution (truncated normal, lognormal, gamma, uniform) with AIC/BIC rankings.

### Statistical Validation
- Monte Carlo output checked against known physical constraints (non-negative demand, reasonable magnitude ranges).
- Max/median ratio monitoring: reduced from 2.78 x 10^7 (pre-fix worst case) to < 100 for all material-technology pairs.
- 95% confidence intervals validated against published literature estimates where available.

### Code Review
- All AI-suggested changes were reviewed by the lead researcher before integration.
- Changes were applied incrementally with before/after validation at each step.
- A detailed changelog (`CHANGELOG_CLAUDE_CODE.md`) documents every modification, its rationale, and its measured impact.

---

## Scope of AI Involvement

| Component | AI Role | Human Role |
|-----------|---------|------------|
| Research design and methodology | None | Full |
| Data collection and sourcing | None | Full |
| Algorithm design (stock-flow model, MC framework) | Advisory | Full |
| Code implementation | Collaborative | Supervisory + verification |
| Bug identification and fixing | Collaborative | Verification via hand calculations |
| Visualization design | Collaborative | Specification + quality review |
| Documentation | Collaborative | Review and editing |
| Statistical interpretation | Advisory | Full |
| Manuscript writing | None | Full |

---

## Reproducibility

The complete pipeline is reproducible via:

```bash
python run_pipeline.py
```

All input data, source code, and configuration are included in the repository. No AI tools are required to run the pipeline — all AI-assisted development produced standard Python code with no runtime AI dependencies.

---

## Contact

For questions about AI usage in this project, contact the lead researcher.
