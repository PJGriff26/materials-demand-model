"""
Visualize raw capacity stock and implied additions across NREL scenarios
to explain the demand convergence around 2035-2038.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "proposal_figures"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

# ── Load capacity data ─────────────────────────────────────────────────────────
df = pd.read_csv(BASE / 'data/StdScen24_annual_national.csv',
                 skiprows=3, low_memory=False)

cap_cols = [c for c in df.columns if c.endswith('_MW')]

# Major technology groups for the stacked view
tech_groups = {
    'Solar (utility + dist.)': ['upv_MW', 'distpv_MW', 'csp_MW'],
    'Wind (on + offshore)':    ['wind_onshore_MW', 'wind_offshore_MW'],
    'Battery storage':         ['battery_4_MW', 'battery_8_MW'],
    'Nuclear':                 ['nuclear_MW', 'nuclear_smr_MW'],
    'Gas':                     ['gas_cc_MW', 'gas_ct_MW', 'gas_cc_ccs_MW', 'h2-ct_MW'],
    'Other':                   ['coal_MW', 'coal_ccs_MW', 'bio_MW', 'bio-ccs_MW',
                                'geo_MW', 'hydro_MW', 'pumped-hydro_MW', 'o-g-s_MW',
                                'dac_MW', 'electrolyzer_MW'],
}
group_colors = {
    'Solar (utility + dist.)': '#f0b429',
    'Wind (on + offshore)':    '#2c7bb6',
    'Battery storage':         '#7b2d8e',
    'Nuclear':                 '#d7191c',
    'Gas':                     '#999999',
    'Other':                   '#cccccc',
}

# Compute total capacity per group per scenario-year
for grp, cols in tech_groups.items():
    present = [c for c in cols if c in df.columns]
    df[grp] = df[present].sum(axis=1)

years = sorted(df['t'].unique())
scenarios = sorted(df['scenario'].unique())

# ── Compute net additions (3-year intervals) ──────────────────────────────────
# For each scenario, additions = capacity(t) - capacity(t-1)
records = []
for scen in scenarios:
    sd = df[df['scenario'] == scen].sort_values('t')
    for i in range(1, len(sd)):
        yr = sd.iloc[i]['t']
        prev_yr = sd.iloc[i-1]['t']
        interval = yr - prev_yr
        row = {'scenario': scen, 'year': yr, 'interval': interval}
        for grp in tech_groups:
            row[grp] = (sd.iloc[i][grp] - sd.iloc[i-1][grp]) / 1000  # GW
        row['Total'] = sum(row[grp] for grp in tech_groups)
        records.append(row)

adds = pd.DataFrame(records)

# Highlights
highlights = {
    'Mid_Case': ('Mid Case', '#1f77b4'),
    'Mid_Case_CO2e_100by2035': ('100% Decarb by 2035', '#d62728'),
    'Mid_Case_No_IRA': ('No IRA', '#2ca02c'),
    'High_Demand_Growth': ('High Demand Growth', '#ff7f0e'),
}

# ════════════════════════════════════════════════════════════════════════════════
# Two-panel figure
# Top: Total net capacity additions spaghetti (mirrors demand spaghetti)
# Bottom: Stacked bar of median additions by technology group
# ════════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), height_ratios=[1, 1],
                                gridspec_kw={'hspace': 0.3})

# ── Panel A: Spaghetti of total net additions ──────────────────────────────────
add_years = sorted(adds['year'].unique())
for scen in scenarios:
    if scen in highlights:
        continue
    sd = adds[adds['scenario'] == scen].sort_values('year')
    ax1.plot(sd['year'], sd['Total'], color='gray', alpha=0.2, lw=0.7)

for scen, (label, color) in highlights.items():
    sd = adds[adds['scenario'] == scen].sort_values('year')
    if len(sd) > 0:
        ax1.plot(sd['year'], sd['Total'], color=color, lw=2.5, label=label, zorder=5)

ax1.axhline(0, color='black', lw=0.5)
ax1.set_xlabel('Year')
ax1.set_ylabel('Net Capacity Additions (GW per 3-yr interval)')
ax1.set_title('A.  Net Capacity Additions Across 61 Scenarios')
ax1.legend(frameon=False, fontsize=9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Annotate the convergence zone
ax1.axvspan(2034.5, 2038.5, alpha=0.08, color='red', zorder=0)
ax1.annotate('Buildout lull', xy=(2036.5, ax1.get_ylim()[1]*0.15),
             ha='center', fontsize=10, color='#aa0000', style='italic')

# ── Panel B: Stacked bar of median additions by tech group ─────────────────────
# Compute median across scenarios for each tech group
bar_width = 2.2
groups_to_plot = ['Solar (utility + dist.)', 'Wind (on + offshore)',
                  'Battery storage', 'Nuclear', 'Gas', 'Other']

medians = {}
for grp in groups_to_plot:
    medians[grp] = adds.groupby('year')[grp].median().values

bottom_pos = np.zeros(len(add_years))
bottom_neg = np.zeros(len(add_years))

for grp in groups_to_plot:
    vals = np.array(medians[grp])
    pos = np.where(vals > 0, vals, 0)
    neg = np.where(vals < 0, vals, 0)
    ax2.bar(add_years, pos, bottom=bottom_pos, width=bar_width,
            color=group_colors[grp], label=grp, edgecolor='white', lw=0.3)
    ax2.bar(add_years, neg, bottom=bottom_neg, width=bar_width,
            color=group_colors[grp], edgecolor='white', lw=0.3)
    bottom_pos += pos
    bottom_neg += neg

ax2.axhline(0, color='black', lw=0.5)
ax2.set_xlabel('Year')
ax2.set_ylabel('Median Net Additions (GW per 3-yr interval)')
ax2.set_title('B.  Median Capacity Additions by Technology')
ax2.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.axvspan(2034.5, 2038.5, alpha=0.08, color='red', zorder=0)

for fmt in ['png', 'pdf']:
    fig.savefig(OUT_DIR / f'figure_capacity_additions.{fmt}')
plt.close(fig)

# Print summary stats
print("Net additions summary (GW, median across scenarios):")
for yr in add_years:
    vals = adds[adds['year'] == yr]['Total']
    print(f"  {yr}: median={vals.median():.0f} GW, range=[{vals.min():.0f}, {vals.max():.0f}]")
