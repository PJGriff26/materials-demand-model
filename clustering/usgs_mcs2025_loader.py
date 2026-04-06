"""
Load supply-chain data from raw USGS MCS 2025 CSV files and OECD CRC 2026.

Replaces the manually-compiled risk_charts_inputs.xlsx with citable,
machine-readable raw sources:
  - USGS MCS 2025 Data Release (DOI: 10.5066/P13XCP3R)
  - OECD Country Risk Classifications (January 2026)

Returns data in the SAME dict-of-DataFrames format that the old
load_risk_data() provided, so downstream code needs no changes.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from config import DATA_DIR, DEMAND_TO_RISK


# ── Paths ─────────────────────────────────────────────────────────────────────

MCS2025_DIR = DATA_DIR / "usgs_mcs_2025"
SALIENT_DIR = MCS2025_DIR / "salient_commodity"
WORLD_DATA_CSV = MCS2025_DIR / "world_data" / "MCS2025_World_Data.csv"
NIR_FIGURE_CSV = MCS2025_DIR / "industry_trends" / "MCS2025_Fig2_Net_Import_Reliance.csv"
OECD_CRC_CSV = DATA_DIR / "oecd_crc" / "oecd_crc_2026.csv"

# Map our 19 risk material names → MCS 2025 salient CSV prefixes
RISK_TO_SALIENT_PREFIX = {
    "Aluminum": "alumi",
    "Boron": "boron",
    "Cement": "cemen",
    "Chromium": "chrom",
    "Copper": "coppe",
    "Lead": "lead",
    "Magnesium": "mgmet",
    "Manganese": "manga",
    "Molybdenum": "molyb",
    "Nickel": "nicke",
    "Niobium": "niobi",
    "Rare Earths": "rareee",
    "Silicon": "simet",
    "Silver": "silve",
    "Steel": "feste",
    "Tin": "tin",
    "Vanadium": "vanad",
    "Yttrium": "yttri",
    "Zinc": "zinc",
}

# Thin-film materials also available
THIN_FILM_SALIENT_PREFIX = {
    "Cadmium": "cadmi",
    "Gallium": "galli",
    "Germanium": "germa",
    "Indium": "indiu",
    "Selenium": "selen",
    "Tellurium": "tellu",
}

# Map risk material names → World Data commodity names
RISK_TO_WORLD_COMMODITY = {
    "Aluminum": "Aluminum",
    "Boron": "Boron",
    "Cement": "Cement",
    "Chromium": "Chromium",
    "Copper": "Copper",
    "Lead": "Lead",
    "Magnesium": "Magnesium metal",
    "Manganese": "Manganese",
    "Molybdenum": "Molybdenum",
    "Nickel": "Nickel",
    "Niobium": "Niobium",
    "Rare Earths": "Rare earths",
    "Silicon": "Silicon",
    "Silver": "Silver",
    "Steel": "Iron and Steel",
    "Tin": "Tin",
    "Vanadium": "Vanadium",
    "Zinc": "Zinc",
}

# NIR column name varies by commodity — map to a canonical column
# Some have NIR_pct, others NIR_Metal_pct, NIR_Refined_pct, etc.
# We pick the most relevant one for each material.
_NIR_COL_OVERRIDES = {
    "Lead": "NIR_Metal_pct",
    "Nickel": "NIR_ct",          # reported as percentage despite name
    "Rare Earths": None,         # complex — use NIR figure data instead
    "Silicon": "NIR_FeSi-Si_pct",  # combined ferrosilicon + silicon metal
    "Tin": "NIR_Refined_pct",
    "Zinc": "NIR_Refined_pct",   # refined zinc NIR, not ores (US exports ore)
}

# Per-material production column preferences.
# Default: first column starting with "USprod". Override here when the
# default picks the wrong form of production (e.g., mine vs refinery).
_PROD_COL_OVERRIDES = {
    # Aluminum: primary smelter production only (excludes secondary/recycled)
    "Aluminum": ["USprod_Primary_kt"],
    # Copper: use refinery production (primary + secondary)
    "Copper": ["USprod_Refinery-primary_kt", "USprod_Refinery-secondary_kt"],
    # Lead: mine production (consistent with other mine-based commodities)
    "Lead": ["USprod_Mine_kt"],
    # Nickel: mine production only (excludes secondary)
    "Nickel": ["USprod_Mine_t"],
    # Steel: raw steel production in mmt — convert to kt (* 1000)
    "Steel": ["USprod_Steel_mmt"],
    # Zinc: use refined production
    "Zinc": ["USprod_Refined_kt"],
}

# Unit conversion factors: multiply CSV value by this to get kt
# Default is 1.0 (CSV already in kt). Override for _t (tonnes) and _mmt.
_UNIT_TO_KT = {
    "_t": 0.001,    # tonnes → kt
    "_mmt": 1000.0,  # million metric tonnes → kt
    "_kt": 1.0,
}


# ── Salient data loading ──────────────────────────────────────────────────────

def _load_salient(material, prefix):
    """Load a single commodity's salient CSV. Returns DataFrame or None."""
    path = SALIENT_DIR / f"mcs2025-{prefix}_salient.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping {material}")
        return None
    return pd.read_csv(path)


def _extract_nir(df, material):
    """
    Extract net import reliance (%) from a salient DataFrame.
    Returns list of (year, nir_pct) tuples.
    """
    if df is None:
        return []

    # Find the right NIR column
    override = _NIR_COL_OVERRIDES.get(material)
    if override is not None:
        if override in df.columns:
            nir_col = override
        else:
            return []
    else:
        # Default: find first column containing "NIR" and "pct"
        candidates = [c for c in df.columns if "NIR" in c and "pct" in c.lower()]
        if not candidates:
            candidates = [c for c in df.columns if "NIR" in c]
        if not candidates:
            return []
        nir_col = candidates[0]

    results = []
    for _, row in df.iterrows():
        year = row.get("Year")
        val = row.get(nir_col)
        if pd.isna(year):
            continue
        s = str(val).strip()
        # Handle USGS text notation
        if s.upper() == "E" or s == "--" or s == "":
            nir = 0.0  # net exporter
        elif s.startswith("<"):
            nir = float(s[1:]) / 2
        elif s.startswith(">"):
            nir = (float(s[1:]) + 100) / 2
        else:
            try:
                nir = float(s)
            except ValueError:
                continue
        results.append((int(year), nir))
    return results


def _col_unit_factor(col_name):
    """Determine the kt conversion factor from a column name suffix."""
    for suffix, factor in _UNIT_TO_KT.items():
        if col_name.endswith(suffix):
            return factor
    # No recognized suffix — check for common patterns
    if "_t" in col_name and "_kt" not in col_name and "_mmt" not in col_name:
        return 0.001  # tonnes → kt
    return 1.0  # assume kt by default


def _extract_production(df, material):
    """
    Extract US production (kt) from a salient DataFrame.
    Returns list of (year, production_kt) tuples.

    Uses _PROD_COL_OVERRIDES to pick the right columns per material
    and _UNIT_TO_KT for unit conversion.
    """
    if df is None:
        return []

    override = _PROD_COL_OVERRIDES.get(material)
    if override:
        prod_cols = [c for c in override if c in df.columns]
    else:
        prod_cols = [c for c in df.columns if c.startswith("USprod")]
        if not prod_cols:
            prod_cols = [c for c in df.columns
                         if "prod" in c.lower() and "Price" not in c
                         and "DataSource" not in c and "Commodity" not in c]
    if not prod_cols:
        return []

    results = []
    for _, row in df.iterrows():
        year = row.get("Year")
        if pd.isna(year):
            continue
        total = 0.0
        any_valid = False
        for col in prod_cols:
            val_str = str(row.get(col, "")).replace(",", "").strip()
            if val_str in ("W", "XX", "--", "", "nan"):
                continue  # withheld or not available
            try:
                val = float(val_str)
                factor = _col_unit_factor(col)
                total += val * factor
                any_valid = True
            except (ValueError, TypeError):
                continue
        if any_valid:
            results.append((int(year), total))
    return results


def _extract_trade(df, material):
    """
    Extract US imports, exports, consumption from a salient DataFrame.
    Returns list of (year, imports_kt, exports_kt, consumption_kt) tuples.
    All values converted to kt using column-name unit suffixes.
    """
    if df is None:
        return []

    # Find import columns (may be multiple: crude, scrap, refined)
    import_cols = [c for c in df.columns if c.startswith("Imports")]
    export_cols = [c for c in df.columns if c.startswith("Exports")]
    # Prefer apparent consumption
    consump_cols = [c for c in df.columns
                    if c.startswith("Consump") and ("Apprnt" in c or "Apparent" in c)]
    if not consump_cols:
        consump_cols = [c for c in df.columns
                        if c.startswith("Consump") and "Total" in c]
    if not consump_cols:
        consump_cols = [c for c in df.columns if c.startswith("Consump")]

    results = []
    for _, row in df.iterrows():
        year = row.get("Year")
        if pd.isna(year):
            continue

        def _sum_cols_kt(cols):
            total = 0.0
            for c in cols:
                v = str(row.get(c, "")).replace(",", "").strip()
                if v.startswith("Less than"):
                    total += 0.25 * _col_unit_factor(c)
                elif v in ("W", "XX", "--", "", "nan"):
                    continue
                else:
                    try:
                        total += float(v) * _col_unit_factor(c)
                    except (ValueError, TypeError):
                        pass
            return total

        imports = _sum_cols_kt(import_cols)
        exports = _sum_cols_kt(export_cols)
        consumption = _sum_cols_kt(consump_cols) if consump_cols else 0.0

        results.append((int(year), imports, exports, consumption))
    return results


# ── Build the legacy-format sheets ────────────────────────────────────────────

def _build_aggregate_sheet():
    """
    Build a DataFrame matching the old 'aggregate' sheet format:
    columns: material, year, production, import, export, consumption, net_import
    """
    records = []
    for material, prefix in RISK_TO_SALIENT_PREFIX.items():
        df = _load_salient(material, prefix)
        if df is None:
            continue

        prod_data = dict(_extract_production(df, material))
        trade_data = _extract_trade(df, material)

        for year, imports, exports, consumption in trade_data:
            production = prod_data.get(year, 0.0)
            net_import = imports - exports
            records.append({
                "material": material,
                "year": year,
                "production": production,
                "import": imports,
                "export": exports,
                "consumption": consumption if consumption > 0 else (production + net_import),
                "net_import": net_import,
            })

    return pd.DataFrame(records)


def _build_import_dependency_sheet():
    """
    Build a DataFrame matching the old 'import_dependency' sheet format:
    columns: material, <year1>, <year2>, ...
    Values are NIR percentages (0-100) or "E" for net exporters.
    """
    all_years = set()
    mat_nir = {}

    for material, prefix in RISK_TO_SALIENT_PREFIX.items():
        df = _load_salient(material, prefix)
        nir_data = _extract_nir(df, material)
        if nir_data:
            mat_nir[material] = dict(nir_data)
            all_years.update(y for y, _ in nir_data)

    # Also handle Rare Earths from NIR figure data if salient didn't work
    if "Rare Earths" not in mat_nir or not mat_nir["Rare Earths"]:
        if NIR_FIGURE_CSV.exists():
            nir_fig = pd.read_csv(NIR_FIGURE_CSV)
            re_row = nir_fig[nir_fig["Commodity"].str.contains("RARE EARTH", case=False, na=False)]
            if not re_row.empty:
                nir_str = str(re_row.iloc[0]["Net_Import_Reliance_pct_2024"]).strip()
                try:
                    val = float(nir_str)
                except ValueError:
                    if nir_str.startswith(">"):
                        val = (float(nir_str[1:]) + 100) / 2
                    elif nir_str.startswith("<"):
                        val = float(nir_str[1:]) / 2
                    else:
                        val = 80.0  # known approximate
                mat_nir["Rare Earths"] = {2024: val}
                all_years.add(2024)

    years = sorted(all_years)
    rows = []
    for material in RISK_TO_SALIENT_PREFIX:
        row = {"material": material}
        nirs = mat_nir.get(material, {})
        for y in years:
            v = nirs.get(y)
            if v is not None:
                if v == 0:
                    row[y] = "E"
                else:
                    row[y] = v
            else:
                row[y] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def _build_production_sheet():
    """
    Build a DataFrame matching the old 'production' sheet format:
    Global and per-country production for each material.
    Source: MCS2025_World_Data.csv
    """
    if not WORLD_DATA_CSV.exists():
        print(f"  WARNING: {WORLD_DATA_CSV} not found")
        return pd.DataFrame()

    wdf = pd.read_csv(WORLD_DATA_CSV)

    # The world data has columns: COMMODITY, COUNTRY, TYPE, PROD_2023, PROD_EST_ 2024, etc.
    # Build a pivot: rows = countries, paired columns per material (2023, 2024)
    records = []
    for risk_name, world_name in RISK_TO_WORLD_COMMODITY.items():
        sub = wdf[wdf["COMMODITY"].str.strip() == world_name]
        if sub.empty:
            continue
        # Take the first TYPE for each commodity (primary production)
        first_type = sub["TYPE"].iloc[0]
        sub = sub[sub["TYPE"] == first_type]

        for _, row in sub.iterrows():
            country = row["COUNTRY"]
            prod_2023 = pd.to_numeric(row.get("PROD_2023"), errors="coerce")
            prod_2024 = pd.to_numeric(row.get("PROD_EST_ 2024"), errors="coerce")
            records.append({
                "material": risk_name,
                "country": country,
                "production_2023": prod_2023,
                "production_2024": prod_2024,
            })

    return pd.DataFrame(records)


def _build_reserves_sheet():
    """
    Build a DataFrame matching the old 'reserves' sheet format:
    rows = countries (with 'Global' and 'United States'),
    columns = material names, values = reserves.
    Source: MCS2025_World_Data.csv (RESERVES_2024 column)
    """
    if not WORLD_DATA_CSV.exists():
        return pd.DataFrame()

    wdf = pd.read_csv(WORLD_DATA_CSV)

    # Collect reserves by country per material
    all_countries = set()
    mat_reserves = {}

    def _parse_reserve_val(raw):
        """Parse reserve values, handling '>' prefix and commas."""
        s = str(raw).strip().replace(",", "")
        if s.startswith(">"):
            s = s[1:]
        try:
            return float(s)
        except (ValueError, TypeError):
            return np.nan

    # World data is in metric tons; old xlsx was in kt.
    # Determine unit from UNIT_MEAS column; convert to kt.
    for risk_name, world_name in RISK_TO_WORLD_COMMODITY.items():
        sub = wdf[wdf["COMMODITY"].str.strip() == world_name]
        if sub.empty:
            continue

        # Determine reserve unit. USGS World Data CSV uses UNIT_MEAS for
        # production. Reserves USUALLY share the same unit, but some
        # commodities have a RESERVE_NOTES field stating the reserves use
        # a different unit (e.g., "Reserve data is thousand metric tons").
        # The XML metadata confirms RESERVE_NOTES overrides UNIT_MEAS.
        unit = sub["UNIT_MEAS"].iloc[0] if "UNIT_MEAS" in sub.columns else ""
        unit_str = str(unit).lower()

        # Check if any row for this commodity has a RESERVE_NOTES override
        has_kt_note = False
        if "RESERVE_NOTES" in sub.columns:
            notes = sub["RESERVE_NOTES"].dropna().astype(str)
            has_kt_note = notes.str.contains("thousand metric tons", case=False).any()

        if "thousand" in unit_str or has_kt_note:
            unit_factor = 1.0   # reserves already in kt
        elif "kilogram" in unit_str:
            unit_factor = 1e-6  # kg → kt
        else:
            unit_factor = 0.001  # metric tons → kt

        # Take rows with reserves data
        sub_res = sub[sub["RESERVES_2024"].notna()]
        if sub_res.empty:
            continue

        mat_reserves[risk_name] = {}
        for _, row in sub_res.iterrows():
            country = row["COUNTRY"]
            val = _parse_reserve_val(row["RESERVES_2024"])
            if pd.notna(val):
                val_kt = val * unit_factor
                if country == "United States":
                    mat_reserves[risk_name]["United States"] = val_kt
                else:
                    mat_reserves[risk_name][country] = val_kt
                all_countries.add(country)

    # Compute Global totals from "World total" rows
    for risk_name in mat_reserves:
        world_name = RISK_TO_WORLD_COMMODITY.get(risk_name)
        if not world_name:
            continue
        sub = wdf[wdf["COMMODITY"].str.strip() == world_name]
        if sub.empty:
            continue

        # Recompute unit factor using same RESERVE_NOTES logic
        unit = sub["UNIT_MEAS"].iloc[0] if "UNIT_MEAS" in sub.columns else ""
        has_kt_note = False
        if "RESERVE_NOTES" in sub.columns:
            notes = sub["RESERVE_NOTES"].dropna().astype(str)
            has_kt_note = notes.str.contains("thousand metric tons", case=False).any()
        if "thousand" in str(unit).lower() or has_kt_note:
            uf = 1.0
        elif "kilogram" in str(unit).lower():
            uf = 1e-6
        else:
            uf = 0.001

        world_rows = sub[sub["COUNTRY"].str.contains("World|Total", case=False, na=False)]
        if not world_rows.empty:
            val = _parse_reserve_val(world_rows.iloc[0]["RESERVES_2024"])
            if pd.notna(val):
                mat_reserves[risk_name]["Global"] = val * uf

        # If no global row, sum all country reserves (already converted)
        if "Global" not in mat_reserves[risk_name]:
            total = sum(v for k, v in mat_reserves[risk_name].items()
                        if k != "United States")
            us = mat_reserves[risk_name].get("United States", 0)
            total += us
            if total > 0:
                mat_reserves[risk_name]["Global"] = total

    # Build wide-format DataFrame (matching old format)
    materials = sorted(mat_reserves.keys())
    countries = ["Global", "United States"] + sorted(
        c for c in all_countries if c not in ("Global", "United States")
    )

    rows = []
    for country in countries:
        row = {"Unnamed: 0": country}
        for mat in materials:
            row[mat] = mat_reserves.get(mat, {}).get(country, np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def _build_import_shares_sheet():
    """
    Build a DataFrame matching the old 'import_shares' sheet format:
    columns: material, country, share

    Primary source: U.S. Census Bureau International Trade API
        (machine-readable bilateral trade data by HTS code).
    Fallback: Legacy hand-compiled risk_charts_inputs.xlsx.

    The Census API provides actual import values by partner country,
    from which percentage shares are computed. This replaces the hand-
    compiled data that was extracted from USGS MCS publication PDFs.
    """
    # Primary: Census Bureau API via census_import_shares module
    try:
        from census_import_shares import fetch_import_shares, CACHE_FILE
        shares = fetch_import_shares(use_cache=True)
        if not shares.empty and shares["material"].nunique() >= 10:
            print(f"  Import shares: {shares['material'].nunique()} materials "
                  f"from Census Bureau data")
            return shares
        print("  WARNING: Census data incomplete, falling back to legacy xlsx.")
    except Exception as e:
        print(f"  WARNING: Census import shares unavailable ({e}), "
              f"falling back to legacy xlsx.")

    # Fallback: old hand-compiled xlsx
    old_xlsx = DATA_DIR / "supply_chain" / "risk_charts_inputs.xlsx"
    if old_xlsx.exists():
        try:
            return pd.read_excel(old_xlsx, sheet_name="import_shares")
        except Exception:
            pass

    # If nothing available, return empty with warning
    print("  WARNING: No import share data available.")
    return pd.DataFrame(columns=["material", "country", "share"])


def _build_crc_sheet():
    """
    Build a DataFrame matching the old 'crc' sheet format:
    columns: country, crc
    Source: OECD CRC January 2026 (parsed from PDF → CSV)
    """
    if not OECD_CRC_CSV.exists():
        print(f"  WARNING: {OECD_CRC_CSV} not found")
        return pd.DataFrame(columns=["country", "crc"])

    crc = pd.read_csv(OECD_CRC_CSV)

    # Map CRC values to match old format:
    # Old format used: "OECD" for 0, numeric 1-7, "China" handled separately
    def _map_crc(row):
        if row["crc"] == 0:
            return "OECD"
        return row["crc"]

    crc["crc_mapped"] = crc.apply(_map_crc, axis=1)

    # Rename some countries to match old format
    country_renames = {
        "China (People's Republic of)": "China",
        "Korea": "Republic of Korea",
        "Türkiye": "Turkey",
        "Viet Nam": "Vietnam",
        "Congo (Kinshasa)": "Democratic Republic of the Congo",
        "Congo (Brazzaville)": "Congo",
        "Côte d'Ivoire": "Cote d'Ivoire",
    }
    crc["country"] = crc["country"].replace(country_renames)

    result = crc[["country", "crc_mapped"]].rename(columns={"crc_mapped": "crc"})

    # Add extra columns to match old format (the old sheet had some extra cols)
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def load_risk_data_mcs2025():
    """
    Load supply-chain risk data from raw USGS MCS 2025 CSVs and OECD CRC.

    Returns a dict of DataFrames with the SAME keys and structure as the
    old load_risk_data() function that read from risk_charts_inputs.xlsx:
      - aggregate: material × year trade data
      - import_dependency: material × year NIR percentages
      - production: global production by country
      - reserves: global/US reserves by country
      - import_shares: import shares by country (from old xlsx as fallback)
      - crc: country risk classifications
    """
    print("  Loading supply-chain data from USGS MCS 2025 raw CSVs...")

    sheets = {
        "aggregate": _build_aggregate_sheet(),
        "import_dependency": _build_import_dependency_sheet(),
        "production": _build_production_sheet(),
        "reserves": _build_reserves_sheet(),
        "import_shares": _build_import_shares_sheet(),
        "crc": _build_crc_sheet(),
    }

    # Report
    for name, df in sheets.items():
        print(f"    {name}: {df.shape[0]} rows × {df.shape[1]} cols")

    return sheets


def load_thin_film_data_mcs2025():
    """
    Load thin-film material data from MCS 2025 salient CSVs.

    Returns DataFrame with columns: material, production_t, nir_pct
    (same format as old load_usgs_2023_thin_film()).
    """
    records = []
    for material, prefix in THIN_FILM_SALIENT_PREFIX.items():
        df = _load_salient(material, prefix)
        if df is None:
            continue

        prod_data = _extract_production(df, material)
        nir_data = _extract_nir(df, material)

        avg_prod = np.mean([p for _, p in prod_data]) if prod_data else 0.0
        avg_nir = np.mean([n / 100.0 for _, n in nir_data]) if nir_data else 1.0

        records.append({
            "material": material,
            "production_t": avg_prod,
            "nir_pct": avg_nir,
        })

    return pd.DataFrame(records).set_index("material") if records else pd.DataFrame()
