"""
Generate 4 EDA figures for thesis proposal.
Uses the materials_demand_model pipeline output (stock-flow on StdScen24 national data).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DEMAND_FILE = BASE / "outputs" / "material_demand_by_scenario.csv"
USGS_FILE = BASE / "data" / "input_usgs.csv"
OUT_DIR = BASE / "proposal_figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Style defaults ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DEMAND_FILE)
usgs = pd.read_csv(USGS_FILE)

# Clean USGS commodity names to match demand materials
usgs_name_map = {'Cu': 'Copper', 'Mn': 'Manganese', 'Ni': 'Nickel', 'Ag': 'Silver'}
usgs['Commodity'] = usgs['Commodity'].replace(usgs_name_map)

# Material categories for coloring
CATEGORY = {
    'Steel': 'Bulk', 'Aluminum': 'Bulk', 'Cement': 'Bulk', 'Copper': 'Bulk',
    'Glass': 'Bulk', 'Fiberglass': 'Bulk',
    'Chromium': 'Specialty', 'Manganese': 'Specialty', 'Nickel': 'Specialty',
    'Silicon': 'Specialty', 'Molybdenum': 'Specialty', 'Vanadium': 'Specialty',
    'Zinc': 'Specialty', 'Tin': 'Specialty', 'Lead': 'Specialty',
    'Magnesium': 'Specialty', 'Silver': 'Specialty',
    'Neodymium': 'Rare Earth', 'Praseodymium': 'Rare Earth',
    'Dysprosium': 'Rare Earth', 'Terbium': 'Rare Earth', 'Yttrium': 'Rare Earth',
    'Boron': 'Other Critical', 'Niobium': 'Other Critical',
    'Cadmium': 'Other Critical', 'Gallium': 'Other Critical',
    'Gadium': 'Other Critical', 'Germanium': 'Other Critical',  # "Gadium" is typo for Gadolinium
    'Indium': 'Other Critical', 'Selenium': 'Other Critical',
    'Tellurium': 'Other Critical',
}
CAT_COLORS = {'Bulk': '#1f77b4', 'Specialty': '#ff7f0e',
              'Rare Earth': '#d62728', 'Other Critical': '#2ca02c'}

# Exclude baseline year (2026 has zero demand)
df = df[df['year'] > 2026].copy()

scenarios = sorted(df['scenario'].unique())
materials = sorted(df['material'].unique())
years = sorted(df['year'].unique())

print(f"Data: {len(scenarios)} scenarios, {len(materials)} materials, years {years}")

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Scenario Spaghetti Plot
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Figure 1: Scenario Spaghetti ──")

# Total demand per scenario per year, convert to million tonnes
total = df.groupby(['scenario', 'year'])['mean'].sum().reset_index()
total['mt'] = total['mean'] / 1e6

highlights = {
    'Mid_Case': ('Mid Case', '#1f77b4', 2.5),
    'Mid_Case_CO2e_100by2035': ('100% Decarb by 2035', '#d62728', 2.5),
    'Mid_Case_No_IRA': ('No IRA', '#2ca02c', 2.5),
    'High_Demand_Growth': ('High Demand Growth', '#ff7f0e', 2.5),
}

fig, ax = plt.subplots(figsize=(8, 6))
for scen in scenarios:
    if scen in highlights:
        continue
    sd = total[total['scenario'] == scen].sort_values('year')
    ax.plot(sd['year'], sd['mt'], color='gray', alpha=0.25, lw=0.7)

for scen, (label, color, lw) in highlights.items():
    sd = total[total['scenario'] == scen].sort_values('year')
    if len(sd) > 0:
        ax.plot(sd['year'], sd['mt'], color=color, lw=lw, label=label, zorder=5)

ax.set_xlabel('Year')
ax.set_ylabel('Total Material Demand (million tonnes per 3-yr interval)')
ax.set_title(f'Material Demand Trajectories Across {len(scenarios)} NREL Scenarios')
ax.legend(frameon=False, fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for fmt in ['png', 'pdf']:
    fig.savefig(OUT_DIR / f'figure1_scenario_spaghetti.{fmt}')
plt.close(fig)

rng = total.groupby('year')['mt'].agg(['min', 'max'])
print(f"  Demand range: {rng['min'].min():.1f} – {rng['max'].max():.1f} Mt")
print(f"  Saved figure1_scenario_spaghetti.png/.pdf")

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Material Correlation Heatmap
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Figure 2: Correlation Heatmap ──")

# Build (scenario×year) × material matrix — each row is a scenario-year observation
wide = df.pivot_table(index=['scenario', 'year'], columns='material',
                      values='mean', fill_value=0)
# Drop zero-variance materials (can't compute correlation)
wide = wide.loc[:, wide.std() > 0]
corr = wide.corr()

# Hierarchical clustering for ordering
dist = 1 - corr.fillna(0).values
np.fill_diagonal(dist, 0)
dist = (dist + dist.T) / 2
dist = np.clip(dist, 0, 2)
condensed = squareform(dist, checks=False)
link = linkage(condensed, method='average')
order = leaves_list(link)
ordered_mats = [corr.columns[i] for i in order]
corr_ordered = corr.loc[ordered_mats, ordered_mats]

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_ordered.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

n = len(ordered_mats)
ax.set_xticks(range(n))
ax.set_xticklabels(ordered_mats, rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(n))
ax.set_yticklabels(ordered_mats, fontsize=7)
ax.set_title('Cross-Material Demand Correlations')
fig.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')

for fmt in ['png', 'pdf']:
    fig.savefig(OUT_DIR / f'figure2_material_correlations.{fmt}')
plt.close(fig)

high_corr = (corr.values[np.triu_indices_from(corr.values, k=1)] > 0.8).sum()
total_pairs = len(corr) * (len(corr) - 1) // 2
print(f"  {high_corr}/{total_pairs} material pairs with r > 0.8")
print(f"  Saved figure2_material_correlations.png/.pdf")

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Box Plots by Material (year 2035)
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Figure 3: Material Box Plots (2035) ──")

yr = 2035
df_yr = df[df['year'] == yr].copy()

# Only include materials with positive median demand (skip trace materials with ~0)
med_demand = df_yr.groupby('material')['mean'].median()
active_mats = med_demand[med_demand > 0].sort_values(ascending=False)
mat_order = active_mats.index.tolist()

fig, ax = plt.subplots(figsize=(12, 6))
positions = range(len(mat_order))
bp_data = [df_yr[df_yr['material'] == m]['mean'].values for m in mat_order]

face_colors = [CAT_COLORS.get(CATEGORY.get(m, 'Other Critical'), '#999999') for m in mat_order]

bp = ax.boxplot(bp_data, positions=positions, widths=0.6, patch_artist=True,
                whis=(5, 95), showfliers=False,
                medianprops=dict(color='black', lw=1.5))
for patch, fc in zip(bp['boxes'], face_colors):
    patch.set_facecolor(fc)
    patch.set_alpha(0.8)

ax.set_yscale('log')
ax.set_xticks(positions)
ax.set_xticklabels(mat_order, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Demand per 3-yr Interval (tonnes, log scale)')
ax.set_title(f'Material Demand Variability Across Scenarios ({yr})')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_elements = [Patch(facecolor=c, label=cat) for cat, c in CAT_COLORS.items()]
ax.legend(handles=legend_elements, frameon=False, fontsize=9, loc='upper right')

for fmt in ['png', 'pdf']:
    fig.savefig(OUT_DIR / f'figure3_material_boxplots.{fmt}')
plt.close(fig)

print(f"  {len(mat_order)} materials with non-zero demand in {yr}")
print(f"  Median demand range: {active_mats.min():.0f} – {active_mats.max():.0f} tonnes")
print(f"  Saved figure3_material_boxplots.png/.pdf")

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Capacity Stress vs Net Import Reliance Scatter
# ════════════════════════════════════════════════════════════════════════════════
print("\n── Figure 4: Risk Scatter ──")

# Peak demand across all scenarios and years (tonnes per 3-yr interval)
peak_demand = df.groupby('material')['mean'].max()

# USGS: average consumption, production, imports, exports (2018-2022), convert to tonnes
consumption = usgs[usgs['Variable'] == 'Consumption - US'].groupby('Commodity')['Value'].mean() * 1e6
production = usgs[usgs['Variable'] == 'Production - US'].groupby('Commodity')['Value'].mean() * 1e6
imports = usgs[usgs['Variable'] == 'Import - US'].groupby('Commodity')['Value'].mean() * 1e6
exports = usgs[usgs['Variable'] == 'Export - US'].groupby('Commodity')['Value'].mean() * 1e6

# Net Import Reliance = (Imports - Exports) / Consumption
# This is the USGS standard metric used in the manuscript/GitHub
# Negative values indicate net exporter status
net_import_reliance = (imports - exports) / consumption

# Accessible supply = consumption (apparent consumption from USGS)
# This represents demonstrated market capacity - what the economy actually absorbs.
# More defensible than max(production, imports) which underestimates for materials
# with both significant domestic production AND imports.
accessible_supply = consumption

# Qualitative estimates of net import reliance for materials without USGS data
# Values represent (I-E)/C as decimal (0.95 = 95% import reliant)
# Net exporters (NIR ≤ 0) are set to 0.0
# Note: Some materials in input_usgs.csv have variable definition issues that inflate NIR
qualitative_import_reliance = {
    'Aluminum': 0.45,  # Override: USGS MCS reports ~45%, not 91% from raw trade data
    'Neodymium': 0.95, 'Praseodymium': 0.95, 'Dysprosium': 0.95,
    'Terbium': 0.95, 'Yttrium': 0.90,
    'Niobium': 0.85, 'Boron': 0.0, 'Silicon': 0.40,  # Boron: net exporter
    'Chromium': 0.70, 'Vanadium': 0.80, 'Molybdenum': 0.0,  # Molybdenum: net exporter
    'Manganese': 1.00,  # Zero US production per USGS
    'Tin': 0.80, 'Lead': 0.10, 'Zinc': 0.30,
    'Magnesium': 0.50, 'Glass': 0.05, 'Fiberglass': 0.10,
    'Cadmium': 0.30, 'Gallium': 1.00, 'Gadium': 0.90,  # Note: "Gadium" is typo for Gadolinium in source data
    'Germanium': 0.95, 'Indium': 1.00, 'Selenium': 0.50,
    'Tellurium': 0.95,  # Updated from thin-film USGS data
}

# Qualitative consumption (tonnes/year) for non-USGS materials
# Represents apparent US consumption - consistent with accessible_supply = consumption
qualitative_consumption = {
    # Rare earths: US consumption estimates from USGS Mineral Commodity Summaries
    'Neodymium': 15_000, 'Praseodymium': 5_000, 'Dysprosium': 1_500,
    'Terbium': 500, 'Yttrium': 8_000,
    # Other critical minerals
    'Manganese': 890_000,  # USGS: ~890k tonnes apparent consumption
    'Niobium': 10_000, 'Boron': 500_000, 'Silicon': 500_000,
    'Chromium': 500_000, 'Vanadium': 10_000, 'Molybdenum': 50_000,
    'Tin': 40_000, 'Lead': 1_500_000, 'Zinc': 1_000_000,
    'Magnesium': 150_000, 'Glass': 20_000_000, 'Fiberglass': 5_000_000,
    # Thin-film materials: consumption estimates
    'Cadmium': 1_000, 'Gallium': 300, 'Gadium': 100,  # "Gadium" is typo for Gadolinium
    'Germanium': 50, 'Indium': 300, 'Selenium': 1_000,
    'Tellurium': 200,
}

rows = []
for m in materials:
    pk = peak_demand.get(m, np.nan)
    if pd.isna(pk) or pk <= 0:
        continue
    # Net Import Reliance (I-E)/C: USGS standard metric
    # Check qualitative overrides FIRST (some USGS-derived values have variable definition issues)
    if m in qualitative_import_reliance:
        nir = qualitative_import_reliance[m]
    elif m in net_import_reliance.index and not pd.isna(net_import_reliance[m]):
        nir = net_import_reliance[m]
    else:
        nir = 0.5
    # Capacity stress = peak demand / (consumption × 3)
    # Peak demand is per 3-year interval; consumption is annual, so multiply by 3
    if m in accessible_supply.index and accessible_supply[m] > 0:
        cr = pk / (accessible_supply[m] * 3)
    elif m in qualitative_consumption and qualitative_consumption[m] > 0:
        cr = pk / (qualitative_consumption[m] * 3)
    else:
        cr = 10000
    rows.append({'material': m, 'net_import_reliance': nir, 'cap_ratio': cr,
                 'peak_demand': pk, 'category': CATEGORY.get(m, 'Other Critical')})

risk = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(8, 6))
for cat, color in CAT_COLORS.items():
    sub = risk[risk['category'] == cat]
    if len(sub) > 0:
        ax.scatter(sub['net_import_reliance'], sub['cap_ratio'], c=color, s=60, label=cat,
                   edgecolors='black', linewidths=0.5, zorder=3)

ax.set_yscale('log')
ax.set_xlabel('Net Import Reliance (imports − exports) / consumption')
ax.set_ylabel('Capacity Stress Ratio (3-yr peak demand / 3-yr consumption)')
ax.set_title('Supply Chain Risk Profile by Material')

# Horizontal line at capacity stress = 1 (projected demand equals current supply)
ax.axhline(1, ls='--', color='gray', alpha=0.5, lw=1)

# Shortened material names for labeling
SHORT_NAMES = {
    'Aluminum': 'Al', 'Boron': 'B', 'Cadmium': 'Cd', 'Cement': 'Cite',
    'Chromium': 'Cr', 'Copper': 'Cu', 'Dysprosium': 'Dy', 'Fiberglass': 'Fgls',
    'Gadium': 'Gd', 'Gallium': 'Ga', 'Germanium': 'Ge', 'Glass': 'Gls',  # Gadium displays as Gd (Gadolinium)
    'Indium': 'In', 'Lead': 'Pb', 'Magnesium': 'Mg', 'Manganese': 'Mn',
    'Molybdenum': 'Mo', 'Neodymium': 'Nd', 'Nickel': 'Ni', 'Niobium': 'Nb',
    'Praseodymium': 'Pr', 'Selenium': 'Se', 'Silicon': 'Si', 'Silver': 'Ag',
    'Steel': 'Fe', 'Tellurium': 'Te', 'Terbium': 'Tb', 'Tin': 'Sn',
    'Vanadium': 'V', 'Yttrium': 'Y', 'Zinc': 'Zn',
}

# Manual label offsets to avoid overlap with dots
# Coordinates: x=NIR (0-1), y=CapStress (log scale)
# Offsets: positive x = right, negative x = left; positive y = up, negative y = down
LABEL_OFFSETS = {
    # Far left cluster (x≈0.0-0.15): net exporters and low import reliance
    'Boron': (6, -8),           # (0.0, low y)
    'Molybdenum': (6, 4),       # (0.0, low y)
    'Glass': (6, -6),           # (0.05, ~0.3)
    'Fiberglass': (-20, 4),     # (0.10, ~0.2)
    'Lead': (6, 4),             # (0.10, ~0.6)
    'Steel': (-16, -6),         # (~0.13, ~0.3)
    'Cement': (6, 4),           # (~0.16, ~0.1)
    # Middle-left (x≈0.30-0.50): moderate import reliance
    'Cadmium': (6, 4),          # (0.30, high y ~3-4)
    'Zinc': (-16, 4),           # (0.30, ~0.7)
    'Silicon': (6, -6),         # (0.40, ~0.7)
    'Aluminum': (6, 4),         # (0.45, ~0.4)
    'Magnesium': (-18, -6),     # (0.50, low y)
    'Copper': (6, -6),          # (~0.50, ~0.3)
    'Selenium': (-18, 4),       # (0.50, ~1.0)
    'Silver': (6, 4),           # (~0.50, ~0.9)
    'Nickel': (-18, 4),         # (~0.50, ~2.0)
    # Right-middle (x≈0.70-0.85): elevated import reliance
    'Chromium': (6, -6),        # (0.70, ~0.15)
    'Tin': (-18, 4),            # (0.80, ~1.2)
    'Vanadium': (6, 4),         # (0.80, high y ~6)
    'Niobium': (6, -6),         # (0.85, ~0.5)
    # Far right cluster (x≈0.90-1.00): high import reliance
    'Yttrium': (-18, -6),       # (0.90, low y)
    'Gadium': (6, 4),           # (0.90, ~1.1)
    'Praseodymium': (-20, -6),  # (0.95, ~0.3)
    'Dysprosium': (6, -6),      # (0.95, ~0.8)
    'Neodymium': (-18, 4),      # (0.95, ~1.4)
    'Terbium': (6, 4),          # (0.95, ~1.6)
    'Germanium': (-18, -6),     # (0.95, varies)
    'Tellurium': (-18, 4),      # (0.95, high y ~8)
    'Gallium': (6, -6),         # (1.00, ~0.6)
    'Manganese': (-20, 4),      # (1.00, ~0.1)
    'Indium': (-18, 0),         # (1.00, extreme y ~1200)
}

for _, r in risk.iterrows():
    m = r['material']
    label = SHORT_NAMES.get(m, m[:3])
    offset = LABEL_OFFSETS.get(m, (8, 5))
    ax.annotate(label, (r['net_import_reliance'], r['cap_ratio']),
                textcoords='offset points', xytext=offset, fontsize=7,
                ha='left' if offset[0] > 0 else 'right')

ax.legend(frameon=False, fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for fmt in ['png', 'pdf']:
    fig.savefig(OUT_DIR / f'figure4_risk_scatter.{fmt}')
plt.close(fig)

top_right = risk[(risk['net_import_reliance'] > 0) & (risk['cap_ratio'] > 1)]
print(f"  High-risk quadrant: {', '.join(sorted(top_right['material'].tolist()))}")
print(f"  Saved figure4_risk_scatter.png/.pdf")

print(f"\n=== All 4 figures saved to {OUT_DIR}/ ===")
