"""
Compare old manuscript figures with new pipeline outputs.
Creates side-by-side comparison images.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import os

# Paths
OLD_FIGS = Path("/Users/pjgriffiths/Desktop/Materials Demand/Github/MaterialDemand-main/Figures")
NEW_FIGS = Path("/Users/pjgriffiths/Desktop/Materials Demand/Python/materials_demand_model/outputs/figures")
OUTPUT_DIR = Path("/Users/pjgriffiths/Desktop/Materials Demand/Python/materials_demand_model/outputs/figures/comparison")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mapping: (old_filename, new_path, description)
COMPARISONS = [
    # Demand/supply chain figures
    ("fig3_repeat_linear.png", "supply_chain/risk_analysis_stacked_bars_linear.png",
     "Demand by CRC Sourcing"),
    ("fig3_repeat_log.png", "supply_chain/risk_analysis_stacked_bars_log.png",
     "Demand by CRC Sourcing (Log Scale)"),

    # Reserve adequacy
    ("fig4_repeat_linear.png", "clustering/fig4_reserve_adequacy.png",
     "Reserve Adequacy by CRC"),
    ("figSI_reserves_shares_crc.png", "clustering/fig4_reserve_adequacy.png",
     "Reserve Shares by CRC"),

    # Import/risk analysis
    ("figSI_import_shares_crc.png", "supply_chain/risk_ranking_chart.png",
     "Import Risk Analysis"),
    ("figSI_net_import_shares.png", "supply_chain/risk_component_heatmap.png",
     "Import Dependency"),

    # Material intensity
    ("material_intensity_plot.png", "manuscript/figs3_intensity_distributions.png",
     "Material Intensity Distributions"),

    # Capacity projections
    ("multi_model_capacity_plot.jpg", "manuscript/figs1_capacity_mid_case.png",
     "Capacity Projections"),
    ("repeat_capacity_plot.jpg", "manuscript/figs1_additions_mid_case.png",
     "Capacity Additions"),

    # Results overview
    ("multi_model_results_plot_REF.jpg", "demand/monte_carlo_visualization.png",
     "Monte Carlo Results"),
    ("multi_model_complete_boxplot.jpg", "manuscript/fig2_technology_breakdown.png",
     "Technology Breakdown"),
]


def create_comparison(old_file, new_file, title, output_name):
    """Create a side-by-side comparison figure."""
    old_path = OLD_FIGS / old_file
    new_path = NEW_FIGS / new_file

    # Check if files exist
    old_exists = old_path.exists()
    new_exists = new_path.exists()

    if not old_exists and not new_exists:
        print(f"  SKIP: Neither file exists for {title}")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Old figure
    if old_exists:
        try:
            old_img = mpimg.imread(str(old_path))
            axes[0].imshow(old_img)
            axes[0].set_title(f"OLD: {old_file}", fontsize=10, fontweight='bold')
        except Exception as e:
            axes[0].text(0.5, 0.5, f"Error loading:\n{old_file}\n{e}",
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title(f"OLD: {old_file} (ERROR)", fontsize=10)
    else:
        axes[0].text(0.5, 0.5, f"File not found:\n{old_file}",
                    ha='center', va='center', transform=axes[0].transAxes, fontsize=12)
        axes[0].set_title(f"OLD: {old_file} (MISSING)", fontsize=10)
    axes[0].axis('off')

    # New figure
    if new_exists:
        try:
            new_img = mpimg.imread(str(new_path))
            axes[1].imshow(new_img)
            axes[1].set_title(f"NEW: {new_file}", fontsize=10, fontweight='bold')
        except Exception as e:
            axes[1].text(0.5, 0.5, f"Error loading:\n{new_file}\n{e}",
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title(f"NEW: {new_file} (ERROR)", fontsize=10)
    else:
        axes[1].text(0.5, 0.5, f"File not found:\n{new_file}",
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title(f"NEW: {new_file} (MISSING)", fontsize=10)
    axes[1].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"compare_{output_name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  Created: {output_path.name}")
    return True


def main():
    print("=" * 60)
    print("FIGURE COMPARISON: Old Manuscript vs New Pipeline")
    print("=" * 60)
    print(f"\nOld figures: {OLD_FIGS}")
    print(f"New figures: {NEW_FIGS}")
    print(f"Output: {OUTPUT_DIR}\n")

    created = 0
    for i, (old, new, desc) in enumerate(COMPARISONS, 1):
        output_name = f"{i:02d}_{desc.lower().replace(' ', '_')}"
        print(f"\n{i}. {desc}")
        if create_comparison(old, new, desc, output_name):
            created += 1

    print(f"\n{'=' * 60}")
    print(f"Created {created} comparison figures in {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
