# Contributing to Materials Demand Model

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/materials_demand_model.git
   cd materials_demand_model
   ```
3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/sensitivity-analysis` - new features
- `bugfix/unit-conversion` - bug fixes
- `docs/update-readme` - documentation
- `test/convergence-check` - tests

### 2. Make Changes

- Write clean, documented code
- Follow existing code style (PEP 8)
- Add docstrings to all functions/classes
- Include type hints where appropriate

### 3. Test Your Changes

```bash
# Run validation tests
cd tests
python validate_units.py

# Run example to ensure it works
cd ../examples
python run_simulation.py
```

### 4. Document Your Changes

- Update relevant docstrings
- Add examples if adding new features
- Update README.md if changing user-facing behavior
- Add entry to CHANGELOG.md

### 5. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: add Sobol sensitivity analysis

- Implement variance-based sensitivity using SALib
- Add plotting function for tornado diagrams
- Include example in documentation
- Addresses issue #123"
```

**Commit message format**:
- `feat:` - new feature
- `fix:` - bug fix
- `docs:` - documentation changes
- `test:` - adding tests
- `refactor:` - code refactoring
- `style:` - formatting, no code change
- `perf:` - performance improvements

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Reference to related issues
- Screenshots/plots if applicable
- Confirmation that tests pass

## Code Style Guidelines

### Python Style

Follow [PEP 8](https://pep8.org/) with these specifics:

- **Line length**: 100 characters max (except long strings/URLs)
- **Imports**: Grouped (stdlib, third-party, local) and alphabetized
- **Docstrings**: NumPy style (see examples in existing code)
- **Type hints**: Use for function signatures
- **Naming**:
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants

### Example Function

```python
def calculate_material_demand(
    capacity_mw: float,
    intensity_t_per_mw: float,
    weight: float = 1.0
) -> float:
    """
    Calculate material demand from capacity and intensity.

    Parameters
    ----------
    capacity_mw : float
        Capacity in megawatts (MW)
    intensity_t_per_mw : float
        Material intensity in tonnes per megawatt (t/MW)
    weight : float, default=1.0
        Weighting factor (0-1)

    Returns
    -------
    float
        Material demand in tonnes (t)

    Examples
    --------
    >>> calculate_material_demand(1000, 5.0)
    5000.0
    """
    return capacity_mw * intensity_t_per_mw * weight
```

## Testing Guidelines

### Unit Tests

Add tests for new functions:

```python
def test_material_demand_calculation():
    """Test that material demand is calculated correctly."""
    demand = calculate_material_demand(
        capacity_mw=1000,
        intensity_t_per_mw=5.0,
        weight=1.0
    )
    assert demand == 5000.0

    # Test with weighting
    demand_weighted = calculate_material_demand(
        capacity_mw=1000,
        intensity_t_per_mw=5.0,
        weight=0.7
    )
    assert demand_weighted == 3500.0
```

### Validation Tests

Add validation for new features:

```python
# tests/validate_new_feature.py
def validate_sensitivity_analysis():
    """Validate that sensitivity analysis produces reasonable results."""
    # Run sensitivity analysis
    indices = run_sobol_analysis()

    # Check that indices sum to approximately 1
    total = sum(indices['S1'])
    assert 0.9 <= total <= 1.1, f"First-order indices sum to {total}, expected ~1.0"

    print("✓ Sensitivity analysis validation passed")
```

## Documentation Guidelines

### Docstrings

Every module, class, and function should have a docstring:

```python
"""
Module: sensitivity_analysis.py

Implements variance-based global sensitivity analysis using Sobol indices.

This module provides tools for identifying which input uncertainties
contribute most to output variance, following the methodology of
Saltelli et al. (2008).

Examples
--------
>>> from src.sensitivity_analysis import sobol_analysis
>>> indices = sobol_analysis(model, param_ranges)
>>> print(indices['S1'])  # First-order effects
"""
```

### README Updates

When adding features, update README.md:

1. Add to feature list if major new capability
2. Update "Basic Python Usage" section with examples
3. Add to "Customization" section if user-configurable
4. Update "Known Limitations" if relevant

### Technical Documentation

For complex additions, create detailed documentation in `docs/`:

```
docs/
├── MONTE_CARLO_ASSESSMENT.md  (existing)
├── UNIT_FIX_SUMMARY.md         (existing)
└── SENSITIVITY_ANALYSIS.md     (your addition)
```

## Areas Needing Contributions

### High Priority

1. **Convergence Diagnostics** ([Issue #1](link))
   - Implement convergence plots
   - Add Monte Carlo uncertainty quantification
   - Calculate effective sample size

2. **Sensitivity Analysis** ([Issue #2](link))
   - Integrate SALib for Sobol indices
   - Create visualization functions
   - Add example workflow

3. **Validation Tests** ([Issue #3](link))
   - Comparison to literature benchmarks
   - Analytical test cases
   - Reproducibility checks

### Medium Priority

4. **Latin Hypercube Sampling** ([Issue #4](link))
   - Implement LHS option
   - Compare efficiency to SRS
   - Handle correlation structure

5. **Correlation Modeling** ([Issue #5](link))
   - Copula-based correlation
   - Empirical correlation estimation
   - Sensitivity to correlation assumptions

6. **Documentation Examples** ([Issue #6](link))
   - Jupyter notebooks
   - Case studies
   - Tutorial series

### Long-term Enhancements

7. Time-varying material intensities
8. Material recycling pathways
9. Spatially-resolved analysis (state-level)
10. Integration with supply chain models

## Research Standards

### Publications Using This Model

If you use this model in research that leads to publication:

1. **Cite the model**: Use citation format in README.md
2. **Share modifications**: Consider contributing back improvements
3. **Report issues**: Let us know if you find bugs or limitations
4. **Validate results**: Compare to benchmarks and document methodology

### Maintaining Research Quality

Contributions should maintain research-grade standards:

- ✓ Follow ISO/JCGM and NIST guidelines where applicable
- ✓ Document assumptions explicitly
- ✓ Include uncertainty quantification
- ✓ Provide reproducible examples
- ✓ Validate against published benchmarks

## Questions?

- **General questions**: Open a [GitHub Discussion](link)
- **Bug reports**: Open a [GitHub Issue](link)
- **Feature requests**: Open a [GitHub Issue](link) with "enhancement" label
- **Email**: [your-email@institution.edu]

## Code of Conduct

This project follows a Code of Conduct to ensure a welcoming environment:

- Be respectful and inclusive
- Focus on constructive feedback
- Assume good intentions
- Prioritize scientific rigor and accuracy

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping improve the Materials Demand Model! 🎉
