#!/usr/bin/env python3
"""
Generate sample size histogram tables for technologies and materials.

Shows the distribution of sample sizes as compact histogram-style tables.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Set up paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.technology_mapping import TECHNOLOGY_CONSOLIDATION


def load_intensity_data():
    """Load and preprocess intensity data with technology consolidation."""
    data_path = ROOT / 'data' / 'intensity_data.csv'
    df = pd.read_csv(data_path)

    # Standardize column names
    df.columns = ['Technology', 'Material', 'g_per_MW']

    # Apply technology consolidation (CDTE→CdTe, ASIGE→a-Si)
    df['Technology'] = df['Technology'].replace(TECHNOLOGY_CONSOLIDATION)

    return df


def create_histogram_bar(count, max_count, bar_width=40):
    """Create a text-based histogram bar."""
    if max_count == 0:
        return ""

    filled = int((count / max_count) * bar_width)
    bar = "█" * filled
    return bar


def bin_sample_size(n):
    """Assign sample size to a bin."""
    if n == 1:
        return 'n=1'
    elif n == 2:
        return 'n=2'
    elif n == 3:
        return 'n=3'
    elif 4 <= n <= 5:
        return '4≤n≤5'
    elif 6 <= n <= 10:
        return '6≤n≤10'
    elif 11 <= n <= 20:
        return '11≤n≤20'
    else:
        return 'n>20'


def generate_technology_histogram():
    """Generate histogram table by technology."""
    print("Generating technology histogram...")

    df = load_intensity_data()

    # Count sample sizes per pair
    pair_counts = df.groupby(['Technology', 'Material']).size().reset_index(name='n')

    # Assign bins
    pair_counts['bin'] = pair_counts['n'].apply(bin_sample_size)

    # Create histogram data
    bin_order = ['n=1', 'n=2', 'n=3', '4≤n≤5', '6≤n≤10', '11≤n≤20', 'n>20']

    tech_histograms = []

    for tech in sorted(pair_counts['Technology'].unique()):
        tech_data = pair_counts[pair_counts['Technology'] == tech]

        # Count materials in each bin
        bin_counts = tech_data['bin'].value_counts()

        tech_hist = {
            'Technology': tech,
            'Total': len(tech_data)
        }

        for bin_name in bin_order:
            tech_hist[bin_name] = bin_counts.get(bin_name, 0)

        tech_histograms.append(tech_hist)

    # Create DataFrame
    hist_df = pd.DataFrame(tech_histograms)

    # Add histogram bars for each bin
    for bin_name in bin_order:
        max_count = hist_df[bin_name].max()
        hist_df[f'{bin_name}_bar'] = hist_df[bin_name].apply(
            lambda x: create_histogram_bar(x, max_count, bar_width=20)
        )

    return hist_df, bin_order


def generate_material_histogram():
    """Generate histogram table by material."""
    print("Generating material histogram...")

    df = load_intensity_data()

    # Count sample sizes per pair
    pair_counts = df.groupby(['Technology', 'Material']).size().reset_index(name='n')

    # Assign bins
    pair_counts['bin'] = pair_counts['n'].apply(bin_sample_size)

    # Create histogram data
    bin_order = ['n=1', 'n=2', 'n=3', '4≤n≤5', '6≤n≤10', '11≤n≤20', 'n>20']

    mat_histograms = []

    for material in sorted(pair_counts['Material'].unique()):
        mat_data = pair_counts[pair_counts['Material'] == material]

        # Count technologies in each bin
        bin_counts = mat_data['bin'].value_counts()

        mat_hist = {
            'Material': material,
            'Total': len(mat_data)
        }

        for bin_name in bin_order:
            mat_hist[bin_name] = bin_counts.get(bin_name, 0)

        mat_histograms.append(mat_hist)

    # Create DataFrame
    hist_df = pd.DataFrame(mat_histograms)

    # Sort by total observations descending
    hist_df = hist_df.sort_values('Total', ascending=False)

    # Add histogram bars for each bin
    for bin_name in bin_order:
        max_count = hist_df[bin_name].max()
        hist_df[f'{bin_name}_bar'] = hist_df[bin_name].apply(
            lambda x: create_histogram_bar(x, max_count, bar_width=20)
        )

    return hist_df, bin_order


def save_technology_histogram(hist_df, bin_order):
    """Save technology histogram as markdown table."""
    outdir = ROOT / 'outputs' / 'uncertainty_decision_support'
    outfile = outdir / 'table9_sample_size_histogram_by_technology.md'

    with open(outfile, 'w') as f:
        f.write("# Table 9: Sample Size Distribution by Technology\n\n")
        f.write("Distribution of material-technology pairs across sample size bins.\n\n")

        # Create display table with bars
        display_rows = []

        for _, row in hist_df.iterrows():
            tech = row['Technology']
            total = row['Total']

            # Format row with histogram bars
            row_data = {'Technology': tech, 'Total': total}

            for bin_name in bin_order:
                count = row[bin_name]
                bar = row[f'{bin_name}_bar']
                row_data[bin_name] = f"{count} {bar}" if count > 0 else "—"

            display_rows.append(row_data)

        display_df = pd.DataFrame(display_rows)

        # Write to markdown
        f.write(display_df.to_markdown(index=False))
        f.write("\n")

    print(f"✓ Saved: {outfile}")
    return outfile


def save_material_histogram(hist_df, bin_order):
    """Save material histogram as markdown table."""
    outdir = ROOT / 'outputs' / 'uncertainty_decision_support'
    outfile = outdir / 'table10_sample_size_histogram_by_material.md'

    with open(outfile, 'w') as f:
        f.write("# Table 10: Sample Size Distribution by Material\n\n")
        f.write("Distribution of material-technology pairs across sample size bins, sorted by total observations.\n\n")

        # Create display table with bars
        display_rows = []

        for _, row in hist_df.iterrows():
            material = row['Material']
            total = row['Total']

            # Format row with histogram bars
            row_data = {'Material': material, 'Total': total}

            for bin_name in bin_order:
                count = row[bin_name]
                bar = row[f'{bin_name}_bar']
                row_data[bin_name] = f"{count} {bar}" if count > 0 else "—"

            display_rows.append(row_data)

        display_df = pd.DataFrame(display_rows)

        # Write to markdown
        f.write(display_df.to_markdown(index=False))
        f.write("\n")

    print(f"✓ Saved: {outfile}")
    return outfile


def main():
    print("=" * 80)
    print("GENERATING SAMPLE SIZE HISTOGRAMS")
    print("=" * 80)
    print()

    # Generate technology histogram
    tech_hist_df, bin_order = generate_technology_histogram()
    print()
    print("Technology Histogram Preview (first 10):")
    print("-" * 80)

    # Preview
    for idx, row in tech_hist_df.head(10).iterrows():
        tech = row['Technology']
        print(f"\n{tech} (Total: {row['Total']})")
        for bin_name in bin_order:
            count = row[bin_name]
            if count > 0:
                bar = row[f'{bin_name}_bar']
                print(f"  {bin_name:10s}: {count:2d} {bar}")

    print()
    print("-" * 80)

    # Save technology histogram
    tech_file = save_technology_histogram(tech_hist_df, bin_order)
    print()

    # Generate material histogram
    mat_hist_df, bin_order = generate_material_histogram()
    print()
    print("Material Histogram Preview (top 10 by total n):")
    print("-" * 80)

    # Preview
    for idx, row in mat_hist_df.head(10).iterrows():
        material = row['Material']
        print(f"\n{material} (Total: {row['Total']})")
        for bin_name in bin_order:
            count = row[bin_name]
            if count > 0:
                bar = row[f'{bin_name}_bar']
                print(f"  {bin_name:10s}: {count:2d} {bar}")

    print()
    print("-" * 80)

    # Save material histogram
    mat_file = save_material_histogram(mat_hist_df, bin_order)
    print()

    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print()
    print("Generated:")
    print(f"  - {tech_file}")
    print(f"  - {mat_file}")
    print()


if __name__ == '__main__':
    main()
