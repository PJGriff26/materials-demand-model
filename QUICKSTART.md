# Quick Start Guide

Get up and running with the Materials Demand Model in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Terminal/command line access

## Installation (2 minutes)

### Step 1: Navigate to Repository

```bash
cd /path/to/materials_demand_model
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected output**:
```
Successfully installed numpy-1.24.x pandas-2.0.x scipy-1.11.x matplotlib-3.7.x ...
```

### Step 3: Verify Installation

```bash
python -c "from src import run_full_simulation; print('✓ Installation successful!')"
```

## Run Your First Simulation (3 minutes)

### Option 1: Using the Example Script (Recommended)

```bash
cd examples
python run_simulation.py
```

**What happens**:
1. Loads material intensity and capacity data
2. Fits probability distributions
3. Runs 10,000 Monte Carlo iterations
4. Generates results and report

**Expected output**:
```
================================================================================
MATERIALS DEMAND STOCK-FLOW MONTE CARLO SIMULATION
================================================================================

Configuration:
  Iterations: 10,000
  Random seed: 42
  ...

Step 1: Loading data...
Step 2: Fitting distributions...
Step 3: Validating technology mapping...
Step 4: Initializing simulation...
Step 5: Running Monte Carlo simulation...
  Iteration 1,000/10,000 (10%)
  Iteration 2,000/10,000 (20%)
  ...

✓ Simulation complete!

Results saved to:
  - outputs/material_demand_by_scenario.csv
  - outputs/material_demand_summary.csv
  - outputs/simulation_report.txt
```

### Option 2: Python Script

Create a file `my_simulation.py`:

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.stock_flow_simulation import run_full_simulation

# Run simulation
simulation, results = run_full_simulation(
    intensity_path='data/intensity_data.csv',
    capacity_path='data/StdScen24_annual_national.csv',
    n_iterations=10000,
    random_state=42
)

# Get statistics
stats = results.get_statistics()

# Print top materials for 2035
copper_2035 = stats[(stats['material'] == 'Copper') & (stats['year'] == 2035)]
print(f"Copper demand (2035): {copper_2035['p50'].values[0]/1e6:.2f} million tonnes")
print(f"95% CI: [{copper_2035['p2'].values[0]/1e6:.2f}, {copper_2035['p97'].values[0]/1e6:.2f}]")

# Save results
stats.to_csv('outputs/my_results.csv', index=False)
print("✓ Results saved to outputs/my_results.csv")
```

Run it:
```bash
python my_simulation.py
```

## Explore Results (5 minutes)

### View Summary Report

```bash
cat outputs/simulation_report.txt
```

**Key sections**:
- Simulation parameters
- Model description
- Top 10 materials by demand
- Temporal trends
- Scenario comparisons

### Load Results in Python

```python
import pandas as pd

# Load detailed results
results = pd.read_csv('outputs/material_demand_by_scenario.csv')

# View structure
print(results.head())
print(f"Shape: {results.shape}")  # ~47,000 rows

# Filter to specific scenario and year
mid_case_2035 = results[
    (results['scenario'] == 'Mid_Case') &
    (results['year'] == 2035)
]

# Top 10 materials
top_10 = mid_case_2035.nlargest(10, 'p50')[['material', 'p50', 'p2', 'p97']]
print(top_10)
```

### Create Visualizations

```python
from src.materials_visualizations import MaterialsDemandVisualizer

# Create visualizer
viz = MaterialsDemandVisualizer(
    results_detailed=pd.read_csv('outputs/material_demand_by_scenario.csv'),
    output_dir='outputs/figures'
)

# Generate figures for key materials
viz.generate_figure_suite(
    key_materials=['Copper', 'Aluminum', 'Steel', 'Lithium'],
    key_year=2035
)

print("✓ Figures saved to outputs/figures/")
```

## Common Tasks

### Run Specific Scenarios Only

```python
scenarios = ['Mid_Case', 'Mid_Case_No_IRA', 'Mid_Case_100by2035']

simulation, results = run_full_simulation(
    intensity_path='data/intensity_data.csv',
    capacity_path='data/StdScen24_annual_national.csv',
    n_iterations=10000,
    scenarios_to_run=scenarios,  # Only these scenarios
    random_state=42
)
```

### Adjust Number of Iterations

```python
# Faster run (fewer iterations)
simulation, results = run_full_simulation(
    intensity_path='data/intensity_data.csv',
    capacity_path='data/StdScen24_annual_national.csv',
    n_iterations=1000,  # 10x faster, less precision
    random_state=42
)

# More precise (more iterations)
simulation, results = run_full_simulation(
    intensity_path='data/intensity_data.csv',
    capacity_path='data/StdScen24_annual_national.csv',
    n_iterations=50000,  # 5x slower, more precision
    random_state=42
)
```

### Modify Technology Mapping

Edit `src/technology_mapping.py`:

```python
TECHNOLOGY_MAPPING = {
    'upv': {
        'utility-scale solar pv': 0.80,  # Changed from 0.70
        'CIGS': 0.10,                     # Changed from 0.15
        'CdTe': 0.10                      # Changed from 0.15
    },
    # ... your modifications
}
```

Then re-run simulation to see impact.

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

**Solution**: Make sure you're running from the repository root or have the correct path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### "FileNotFoundError: intensity_data.csv not found"

**Solution**: Check your paths are correct:
```python
from pathlib import Path

# Use absolute paths
repo_root = Path(__file__).parent.parent
intensity_path = repo_root / 'data' / 'intensity_data.csv'

# Or run from correct directory
# cd /path/to/materials_demand_model
```

### "Output magnitudes are huge (trillions)"

**Solution**: You may be using old data. Ensure you're using the fixed version:
```python
# Check that unit conversion is applied
from src.data_ingestion import MaterialIntensityLoader
loader = MaterialIntensityLoader()
data = loader.load('data/intensity_data.csv')

# Should see this log message:
# "Applied unit conversion: t/GW → t/MW (divided by 1000)"

# Check a value
print(data[data['material'] == 'Copper']['intensity_t_per_mw'].head())
# Should be in range 0.02 - 22 t/MW, NOT 20 - 22,000
```

## Next Steps

1. **Read the full README**: `README.md` has comprehensive documentation
2. **Review methodology**: See `docs/MONTE_CARLO_ASSESSMENT.md` for technical details
3. **Understand the fix**: Read `docs/UNIT_FIX_SUMMARY.md` about the unit conversion
4. **Customize**: Edit `src/technology_mapping.py` for your scenarios
5. **Contribute**: See `CONTRIBUTING.md` to help improve the model

## Getting Help

- **Documentation**: Start with `README.md`
- **Technical details**: See `docs/` directory
- **Issues**: Open a GitHub issue
- **Questions**: Email [your-email@institution.edu]

---

**Time to first results**: ~5 minutes
**Full simulation runtime**: ~5-15 minutes (depending on hardware)
**Ready for research**: Yes, after validation! ✓
