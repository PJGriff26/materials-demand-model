# Changelog

All notable changes to the Materials Demand Model will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-26

### Added
- Initial release of research-grade Monte Carlo simulation framework
- Complete stock-flow accounting model for material demand
- Robust distribution fitting with multiple parametric and empirical options
- Technology mapping system for capacity-to-intensity relationships
- Publication-quality visualization suite
- Comprehensive documentation and examples
- Unit validation testing framework

### Fixed
- **CRITICAL**: Unit conversion from t/GW to t/MW (previous versions overestimated by 1000×)
  - See `docs/UNIT_FIX_SUMMARY.md` for details
  - All results from versions prior to 1.0.0 are invalid

### Technical Details
- Monte Carlo implementation follows ISO/JCGM 101:2008 standards
- N=10,000 iterations with full percentile reporting (2.5-97.5)
- Support for 21 technologies and 31 materials
- 61 NREL Standard Scenarios included

### Known Limitations
- Material intensities assumed independent (no correlation structure)
- Static intensities over time (no learning curves)
- No material recycling/recovery modeled
- Simplified retirement model (baseline stock assumed new)

## [Unreleased]

### Planned Enhancements
- Convergence diagnostics and visualization
- Sobol sensitivity analysis
- Latin Hypercube Sampling option
- Correlation structure for related materials
- Time-varying material intensities
- Material recycling pathways
- Age-distributed baseline stock

---

**Note**: This is Version 1.0.0 - the first research-grade release. Previous developmental versions (11.30.25 and earlier) contained a critical unit error and should not be used.
