"""
Risk Ranking Chart - Supply Chain Risk Visualization

Creates a ranked horizontal bar chart showing materials ordered by import
dependency, with stacked components showing the breakdown of risk factors.

Risk Components:
1. Import Dependency: Net import reliance (0-100%)
2. Source Concentration: HHI of import sources (0-1, higher = more concentrated)
3. Geopolitical Risk: CRC-weighted average of import sources (0-8 scale, normalized)
4. China Exposure: Share of imports from China (0-100%)

Author: Generated for Materials Demand Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def load_risk_data(xlsx_path):
    """Load all sheets from risk_charts_inputs.xlsx."""
    sheets = {}
    for name in ["aggregate", "import_dependency", "import_shares", "reserves", "crc"]:
        sheets[name] = pd.read_excel(xlsx_path, sheet_name=name)
    return sheets


def calculate_import_dependency(risk_data):
    """Calculate average import dependency (0-1 scale) per material."""
    imp_df = risk_data["import_dependency"].copy()
    year_cols = [c for c in imp_df.columns if c != "material"]

    results = {}
    for _, row in imp_df.iterrows():
        material = row["material"]
        vals = []
        for yc in year_cols:
            val = row[yc]
            if str(val).strip().upper() == "E":
                vals.append(0)  # Net exporter = 0 dependency
            else:
                try:
                    vals.append(float(val))
                except (ValueError, TypeError):
                    pass
        if vals:
            results[material] = np.mean(vals) / 100.0  # Convert to 0-1

    return pd.Series(results, name="import_dependency")


def calculate_crc_weighted_risk(risk_data):
    """
    Calculate CRC-weighted import risk per material (0-1 scale).
    Higher = riskier supply chain.
    """
    # CRC weights: OECD=0, 1-7 scaled, China=7, Undefined=4
    crc_weights = {
        "OECD": 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
        "China": 7, "Undefined": 4,
    }

    imp_shares = risk_data["import_shares"].copy()
    crc_map = risk_data["crc"].iloc[:, :2].copy()
    crc_map.columns = ["country", "crc"]

    # Merge CRC ratings with import shares
    merged = imp_shares.merge(crc_map, on="country", how="left")
    merged.loc[merged["country"] == "China", "crc"] = "China"
    merged["crc"] = merged["crc"].fillna("Undefined")

    results = {}
    for mat, grp in merged.groupby("material"):
        weighted = 0.0
        total_share = 0.0
        for _, row in grp.iterrows():
            w = crc_weights.get(row["crc"], 4)
            s = row["share"] if pd.notna(row["share"]) else 0
            weighted += w * s
            total_share += s
        if total_share > 0:
            # Normalize to 0-1 (max CRC weight is 7)
            results[mat] = (weighted / total_share) / 7.0

    return pd.Series(results, name="crc_risk")


def calculate_china_exposure(risk_data):
    """Calculate China's share of imports (0-1 scale) per material."""
    imp_shares = risk_data["import_shares"].copy()

    results = {}
    for mat, grp in imp_shares.groupby("material"):
        total = grp["share"].sum()
        china = grp.loc[grp["country"] == "China", "share"].sum()
        if total > 0:
            results[mat] = china / total
        else:
            results[mat] = 0.0

    return pd.Series(results, name="china_exposure")


def calculate_source_concentration(risk_data):
    """
    Calculate Herfindahl-Hirschman Index (HHI) of import sources (0-1 scale).
    Higher = more concentrated (riskier).
    """
    imp_shares = risk_data["import_shares"].copy()

    results = {}
    for mat, grp in imp_shares.groupby("material"):
        total = grp["share"].sum()
        if total > 0:
            shares = grp["share"] / total
            hhi = (shares ** 2).sum()
            results[mat] = hhi
        else:
            results[mat] = 0.0

    return pd.Series(results, name="source_concentration")


def build_risk_dataframe(risk_data):
    """Build a DataFrame with all risk components."""
    import_dep = calculate_import_dependency(risk_data)
    crc_risk = calculate_crc_weighted_risk(risk_data)
    china_exp = calculate_china_exposure(risk_data)
    concentration = calculate_source_concentration(risk_data)

    # Combine into DataFrame
    df = pd.DataFrame({
        "Import Dependency": import_dep,
        "Geopolitical Risk": crc_risk,
        "China Exposure": china_exp,
        "Source Concentration": concentration,
    })

    # Fill missing values with median
    df = df.fillna(df.median())

    # Sort by import dependency (primary risk indicator)
    df = df.sort_values("Import Dependency", ascending=True)

    return df


def create_risk_ranking_chart(risk_data, output_path=None, figsize=(12, 10)):
    """
    Create a horizontal grouped bar chart showing materials ranked by
    import dependency with all risk components displayed.
    """
    df = build_risk_dataframe(risk_data)

    components = ["Import Dependency", "Geopolitical Risk", "China Exposure", "Source Concentration"]

    # Color palette
    colors = {
        "Import Dependency": "#e74c3c",      # Red
        "Geopolitical Risk": "#f39c12",      # Orange
        "China Exposure": "#9b59b6",         # Purple
        "Source Concentration": "#3498db",   # Blue
    }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot grouped bars (side by side for each material)
    n_components = len(components)
    bar_height = 0.8 / n_components
    y_positions = np.arange(len(df))

    for i, component in enumerate(components):
        values = df[component].values
        offset = (i - n_components / 2 + 0.5) * bar_height
        bars = ax.barh(
            y_positions + offset,
            values,
            height=bar_height,
            color=colors[component],
            label=component,
            edgecolor="white",
            linewidth=0.5,
        )

    # Customize axes
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df.index, fontsize=11)
    ax.set_xlabel("Risk Score (0-1)", fontsize=12)
    ax.set_title("Supply Chain Risk Components by Material\n(Higher = Greater Risk)",
                 fontsize=14, fontweight="bold", pad=20)

    # Set x-axis limits
    ax.set_xlim(0, 1.1)

    # Add vertical lines for risk thresholds
    ax.axvline(x=0.25, color="green", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=0.50, color="orange", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=0.75, color="red", linestyle="--", alpha=0.5, linewidth=1)

    # Add threshold labels
    ax.text(0.25, len(df) + 0.3, "Low", ha="center", fontsize=9, color="green")
    ax.text(0.50, len(df) + 0.3, "Medium", ha="center", fontsize=9, color="orange")
    ax.text(0.75, len(df) + 0.3, "High", ha="center", fontsize=9, color="red")

    # Legend
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=10)

    # Grid
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Spine styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved to: {output_path}")

    return fig, ax, df


def create_risk_component_heatmap(risk_data, output_path=None, figsize=(10, 12)):
    """
    Create a heatmap showing individual risk component scores per material.
    Complementary view to the bar chart.
    """
    df = build_risk_dataframe(risk_data)

    components = ["Import Dependency", "Geopolitical Risk", "China Exposure", "Source Concentration"]
    heatmap_data = df[components].copy()

    # Sort by import dependency (descending)
    heatmap_data = heatmap_data.loc[df.sort_values("Import Dependency", ascending=False).index]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(heatmap_data.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    # Axes
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels([c.replace(" ", "\n") for c in components], fontsize=10)
    ax.set_yticks(range(len(heatmap_data)))
    ax.set_yticklabels(heatmap_data.index, fontsize=10)

    # Add value annotations
    for i in range(len(heatmap_data)):
        for j in range(len(components)):
            val = heatmap_data.iloc[i, j]
            text_color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                   fontsize=9, color=text_color)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Risk Score (0-1)", fontsize=11)

    ax.set_title("Supply Chain Risk Components by Material\n(0 = Low Risk, 1 = High Risk)",
                fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Heatmap saved to: {output_path}")

    return fig, ax


def main():
    """Generate risk visualization charts."""
    # Paths
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "supply_chain" / "risk_charts_inputs.xlsx"
    output_dir = base_dir / "outputs" / "figures" / "supply_chain"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {data_path}")
    risk_data = load_risk_data(data_path)

    # Generate ranked bar chart
    print("\nGenerating risk ranking chart...")
    fig1, ax1, risk_df = create_risk_ranking_chart(
        risk_data,
        output_path=output_dir / "risk_ranking_chart.png"
    )

    # Generate heatmap
    print("\nGenerating risk component heatmap...")
    fig2, ax2 = create_risk_component_heatmap(
        risk_data,
        output_path=output_dir / "risk_component_heatmap.png"
    )

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUPPLY CHAIN RISK SUMMARY")
    print("=" * 60)
    print(f"\nHighest Import Dependency:")
    top5 = risk_df.nlargest(5, "Import Dependency")[["Import Dependency"]]
    for mat, row in top5.iterrows():
        print(f"  {mat}: {row['Import Dependency']:.3f}")

    print(f"\nLowest Import Dependency:")
    bottom5 = risk_df.nsmallest(5, "Import Dependency")[["Import Dependency"]]
    for mat, row in bottom5.iterrows():
        print(f"  {mat}: {row['Import Dependency']:.3f}")

    # Save summary CSV
    risk_df.to_csv(output_dir / "risk_scores_by_material.csv")
    print(f"\nRisk scores saved to: {output_dir / 'risk_scores_by_material.csv'}")

    plt.show()

    return risk_df


if __name__ == "__main__":
    main()
