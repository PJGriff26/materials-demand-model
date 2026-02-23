# feature_engineering.py
"""
Feature engineering for clustering analysis.
Creates scenario-level and material-level feature matrices from
the Monte Carlo demand output and risk/supply-chain data.
"""

import pandas as pd
import numpy as np
from config import (
    DEMAND_FILE, NREL_SCENARIOS_FILE, USGS_SUPPLY_CHAIN_FILE,
    RISK_INPUTS_FILE, USGS_2023_DIR,
    USGS_TO_DEMAND, DEMAND_TO_RISK, USGS_2023_FILES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_demand_data():
    """Load the MC demand output."""
    return pd.read_csv(DEMAND_FILE)


def load_nrel_data():
    """Load NREL StdScen24 capacity data (3 header rows to skip)."""
    return pd.read_csv(NREL_SCENARIOS_FILE, skiprows=3)


def load_usgs_data():
    """Load USGS supply chain data and map commodity names to demand names."""
    usgs = pd.read_csv(USGS_SUPPLY_CHAIN_FILE)
    usgs["material"] = usgs["Commodity"].map(USGS_TO_DEMAND)
    return usgs


def load_risk_data():
    """
    Load the risk_charts_inputs.xlsx workbook.
    Returns dict of DataFrames: aggregate, import_dependency, production,
    reserves, import_shares, crc.
    """
    if not RISK_INPUTS_FILE.exists():
        print(f"  WARNING: {RISK_INPUTS_FILE} not found — risk features will be empty")
        return None

    sheets = {}
    for name in ["aggregate", "import_dependency", "production", "reserves",
                  "import_shares", "crc"]:
        sheets[name] = pd.read_excel(RISK_INPUTS_FILE, sheet_name=name)
    return sheets


def load_usgs_2023_thin_film():
    """
    Parse USGS 2023 individual CSVs for thin-film materials
    (Cadmium, Gallium, Germanium, Indium, Selenium, Tellurium).

    Returns a DataFrame with columns: material, production_t, nir_pct
    (average across available years).
    """
    records = []
    for material, filename in USGS_2023_FILES.items():
        path = USGS_2023_DIR / filename
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping {material}")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")

        # Extract NIR (net import reliance %)
        # USGS reports NIR as text ranges: "<75", ">50", "100", etc.
        # Interpretation: "<X" → X/2 (midpoint to 0), ">X" → (X+100)/2 (midpoint to 100)
        nir_col = [c for c in df.columns if "NIR" in c or "nir" in c]
        nir_vals = []
        if nir_col:
            for v in df[nir_col[0]]:
                s = str(v).strip()
                try:
                    if s.startswith("<"):
                        # "<75" → midpoint between 0 and 75 = 37.5
                        num = float(s[1:])
                        nir_vals.append(num / 2)
                    elif s.startswith(">"):
                        # ">50" → midpoint between 50 and 100 = 75
                        num = float(s[1:])
                        nir_vals.append((num + 100) / 2)
                    else:
                        nir_vals.append(float(s))
                except ValueError:
                    pass
        avg_nir = np.mean(nir_vals) / 100.0 if nir_vals else 1.0

        # Extract US production — look for columns starting with "USprod"
        prod_cols = [c for c in df.columns if c.startswith("USprod")]
        avg_prod = 0.0
        if prod_cols:
            for pc in prod_cols:
                vals = pd.to_numeric(df[pc], errors="coerce").fillna(0)
                avg_prod += vals.mean()
            # Determine unit from column name
            if any("_kg" in c for c in prod_cols):
                avg_prod /= 1000.0  # kg → t
            # else already in tonnes

        records.append({
            "material": material,
            "production_t": avg_prod,
            "nir_pct": avg_nir,
        })

    return pd.DataFrame(records).set_index("material") if records else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# Risk / supply-chain helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_production_series(risk_data, thin_film_data):
    """
    Build a Series of average annual US production (tonnes) per demand-material
    name, combining:
      1. risk_charts_inputs.xlsx → aggregate sheet (19 materials)
      2. USGS 2023 thin-film CSVs (6 materials)
    """
    prod = pd.Series(dtype=float, name="production_t")

    # --- From aggregate sheet ---
    if risk_data is not None:
        agg = risk_data["aggregate"].copy()
        # Coerce non-numeric entries (e.g., '-----') to NaN
        for col in ["production", "import", "export", "consumption", "net_import"]:
            if col in agg.columns:
                agg[col] = pd.to_numeric(agg[col], errors="coerce")
        # Average production across years per material
        avg_prod = agg.groupby("material")["production"].mean()
        # Map risk material names → demand material names
        for demand_name, risk_name in DEMAND_TO_RISK.items():
            if risk_name in avg_prod.index and demand_name not in prod.index:
                prod[demand_name] = avg_prod[risk_name]

    # --- From thin-film CSVs ---
    if thin_film_data is not None and not thin_film_data.empty:
        for mat in thin_film_data.index:
            prod[mat] = thin_film_data.loc[mat, "production_t"]

    return prod


def _build_import_dependency_series(risk_data, thin_film_data):
    """
    Build a Series of import dependency (0-1 fraction) per demand-material name.

    Data sources (in priority order):
    1. aggregate sheet: Calculate NIR = (imports - exports) / consumption
       - Most accurate, handles net exporters (negative NIR → 0)
    2. import_dependency sheet: Use reported values
       - "E" = net exporter → 0.0 (not NaN)
    3. USGS 2023 thin-film CSVs: Use nir_pct values
    """
    dep = pd.Series(dtype=float, name="import_dependency")

    if risk_data is not None:
        # --- Priority 1: Calculate from aggregate trade data ---
        if "aggregate" in risk_data:
            agg = risk_data["aggregate"].copy()
            # Coerce non-numeric entries to NaN
            for col in ["import", "export", "consumption"]:
                if col in agg.columns:
                    agg[col] = pd.to_numeric(agg[col], errors="coerce")

            # Calculate NIR per year, then average
            agg["nir"] = (agg["import"] - agg["export"]) / agg["consumption"]
            avg_nir = agg.groupby("material")["nir"].mean()

            for demand_name, risk_name in DEMAND_TO_RISK.items():
                if risk_name in avg_nir.index:
                    nir_val = avg_nir[risk_name]
                    if pd.notna(nir_val):
                        # Clamp: net exporters → 0, max 1.0
                        dep[demand_name] = max(0.0, min(1.0, nir_val))

        # --- Priority 2: Fallback to import_dependency sheet ---
        if "import_dependency" in risk_data:
            imp_df = risk_data["import_dependency"]
            year_cols = [c for c in imp_df.columns if c != "material"]
            for _, row in imp_df.iterrows():
                risk_name = row["material"]
                vals = []
                is_net_exporter = False
                for yc in year_cols:
                    val = row[yc]
                    if str(val).strip().upper() == "E":
                        is_net_exporter = True
                    else:
                        try:
                            vals.append(float(val))
                        except (ValueError, TypeError):
                            pass

                # Determine NIR value
                if is_net_exporter and not vals:
                    # All years marked "E" → net exporter
                    avg = 0.0
                elif vals:
                    avg = np.mean(vals) / 100.0
                else:
                    avg = np.nan

                # Map to demand names (only if not already set from aggregate)
                for demand_name, rn in DEMAND_TO_RISK.items():
                    if rn == risk_name and demand_name not in dep.index:
                        if pd.notna(avg):
                            dep[demand_name] = max(0.0, min(1.0, avg))

    # --- Priority 3: Thin-film materials from USGS 2023 CSVs ---
    if thin_film_data is not None and not thin_film_data.empty:
        for mat in thin_film_data.index:
            if mat not in dep.index:  # Don't overwrite if already set
                dep[mat] = thin_film_data.loc[mat, "nir_pct"]

    return dep


def _build_reserves_series(risk_data):
    """
    Build a Series of global reserves (kt) per demand-material name.
    Source: reserves sheet, row 0 = 'Global'.
    """
    res = pd.Series(dtype=float, name="reserves_kt")
    if risk_data is None:
        return res
    reserves_df = risk_data["reserves"]
    global_row = reserves_df[reserves_df["Unnamed: 0"] == "Global"]
    if global_row.empty:
        return res
    for demand_name, risk_name in DEMAND_TO_RISK.items():
        if risk_name in global_row.columns:
            val = global_row[risk_name].values[0]
            try:
                res[demand_name] = float(val)
            except (ValueError, TypeError):
                pass
    return res


def _build_domestic_reserves_series(risk_data):
    """
    Build a Series of US domestic reserves (kt) per demand-material name.
    Source: reserves sheet, row where Unnamed: 0 == 'United States'.
    """
    res = pd.Series(dtype=float, name="domestic_reserves_kt")
    if risk_data is None:
        return res
    reserves_df = risk_data["reserves"]
    us_row = reserves_df[reserves_df["Unnamed: 0"] == "United States"]
    if us_row.empty:
        return res
    for demand_name, risk_name in DEMAND_TO_RISK.items():
        if risk_name in us_row.columns:
            val = us_row[risk_name].values[0]
            try:
                res[demand_name] = float(val)
            except (ValueError, TypeError):
                pass
    return res


def _build_reserves_by_crc(risk_data):
    """
    For each material, compute the fraction of global reserves held in
    high-risk countries (CRC 5-7 + China) vs. OECD/US.

    Returns dict of Series:
      - reserves_high_risk_frac: fraction in CRC 5-7 + China
      - reserves_oecd_frac: fraction in OECD + US
      - reserves_china_frac: fraction in China
    """
    out = {
        "reserves_high_risk_frac": pd.Series(dtype=float),
        "reserves_oecd_frac": pd.Series(dtype=float),
        "reserves_china_frac": pd.Series(dtype=float),
    }
    if risk_data is None:
        return out

    reserves_df = risk_data["reserves"]
    crc_map = risk_data["crc"].iloc[:, :2].copy()
    crc_map.columns = ["country", "crc"]

    # Get material columns (everything except Unnamed: 0)
    mat_cols = [c for c in reserves_df.columns if c != "Unnamed: 0"]

    # Skip special rows
    skip = {"Global", "United States", "Other"}
    country_res = reserves_df[~reserves_df["Unnamed: 0"].isin(skip)].copy()
    country_res = country_res.rename(columns={"Unnamed: 0": "country"})

    # Merge CRC
    country_res = country_res.merge(crc_map, on="country", how="left")
    country_res.loc[country_res["country"] == "China", "crc"] = "China"

    # Get global totals
    global_row = reserves_df[reserves_df["Unnamed: 0"] == "Global"]

    for risk_name in mat_cols:
        # Coerce to numeric
        country_res[risk_name] = pd.to_numeric(country_res[risk_name], errors="coerce").fillna(0)
        global_val = pd.to_numeric(global_row[risk_name].values[0], errors="coerce")
        if pd.isna(global_val) or global_val == 0:
            continue

        # Compute CRC group totals
        high_risk_crcs = {5, 6, 7, "China"}
        oecd_crcs = {"OECD"}

        high_risk = country_res[country_res["crc"].isin(high_risk_crcs)][risk_name].sum()
        oecd = country_res[country_res["crc"].isin(oecd_crcs)][risk_name].sum()
        # Add US to OECD
        us_row = reserves_df[reserves_df["Unnamed: 0"] == "United States"]
        us_val = pd.to_numeric(us_row[risk_name].values[0], errors="coerce")
        if not pd.isna(us_val):
            oecd += us_val
        china = country_res[country_res["crc"] == "China"][risk_name].sum()

        # Map to demand names
        for demand_name, rn in DEMAND_TO_RISK.items():
            if rn == risk_name:
                out["reserves_high_risk_frac"][demand_name] = high_risk / global_val
                out["reserves_oecd_frac"][demand_name] = oecd / global_val
                out["reserves_china_frac"][demand_name] = china / global_val

    return out


def _build_crc_weighted_risk(risk_data):
    """
    CRC-weighted import risk per material.
    = sum(import_share_i * crc_weight_i) where crc_weight maps:
      OECD→1, 1→2, 2→3, ..., 7→8, China→7, US→0, Undefined→5
    Higher = riskier supply chain.
    Returns Series indexed by risk material name (0-8 scale).
    """
    crc_risk = pd.Series(dtype=float, name="crc_weighted_risk")
    if risk_data is None:
        return crc_risk

    crc_weights = {
        "OECD": 1, "United States": 0,
        1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8,
        "China": 7, "Undefined": 5,
    }

    imp_shares = risk_data["import_shares"]  # material, country, share
    crc_map = risk_data["crc"].iloc[:, :2]
    crc_map.columns = ["country", "crc"]

    merged = imp_shares.merge(crc_map, on="country", how="left")
    # Override China
    merged.loc[merged["country"] == "China", "crc"] = "China"

    for mat, grp in merged.groupby("material"):
        weighted = 0.0
        total_share = 0.0
        for _, row in grp.iterrows():
            w = crc_weights.get(row["crc"], 5)
            s = row["share"] if pd.notna(row["share"]) else 0
            weighted += w * s
            total_share += s
        if total_share > 0:
            crc_risk[mat] = weighted / total_share

    return crc_risk


def _build_crc_sourcing_breakdown(risk_data):
    """
    For each material, compute import sourcing fractions by CRC group:
      - import_china_frac: % of imports from China
      - import_high_risk_frac: % from CRC 5-7 + China
      - import_oecd_frac: % from OECD countries
      - import_hhi: Herfindahl-Hirschman Index of import concentration

    Returns dict of Series indexed by risk material name.
    """
    out = {
        "import_china_frac": pd.Series(dtype=float),
        "import_high_risk_frac": pd.Series(dtype=float),
        "import_oecd_frac": pd.Series(dtype=float),
        "import_hhi": pd.Series(dtype=float),
    }
    if risk_data is None:
        return out

    imp_shares = risk_data["import_shares"]
    crc_map = risk_data["crc"].iloc[:, :2].copy()
    crc_map.columns = ["country", "crc"]

    merged = imp_shares.merge(crc_map, on="country", how="left")
    merged.loc[merged["country"] == "China", "crc"] = "China"

    high_risk_crcs = {5, 6, 7, "China"}
    oecd_crcs = {"OECD"}

    for mat, grp in merged.groupby("material"):
        total = grp["share"].sum()
        if total == 0:
            continue
        shares = grp["share"] / total  # normalize to fractions

        china = grp.loc[grp["crc"] == "China", "share"].sum() / total
        high_risk = grp.loc[grp["crc"].isin(high_risk_crcs), "share"].sum() / total
        oecd = grp.loc[grp["crc"].isin(oecd_crcs), "share"].sum() / total
        hhi = (shares ** 2).sum()  # 0-1 scale, higher = more concentrated

        out["import_china_frac"][mat] = china
        out["import_high_risk_frac"][mat] = high_risk
        out["import_oecd_frac"][mat] = oecd
        out["import_hhi"][mat] = hhi

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario features
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_scenario_features(demand, nrel, risk_data=None, thin_film_data=None):
    """
    Construct features for each of the 61 scenarios.

    Parameters
    ----------
    demand : DataFrame
        MC demand output (scenario, year, material, mean, std, p2, p5, ..., p95, p97).
    nrel : DataFrame
        NREL StdScen24 capacity data.
    risk_data : dict, optional
        Risk sheets from risk_charts_inputs.xlsx.
    thin_film_data : DataFrame, optional
        USGS 2023 thin-film material data.

    Returns
    -------
    DataFrame with scenario index and numeric feature columns.
    """
    years = sorted(demand["year"].unique())

    # Pivot: for each scenario+year, total demand across all materials
    total_by_sy = (
        demand.groupby(["scenario", "year"])["mean"]
        .sum()
        .reset_index()
        .rename(columns={"mean": "total_demand"})
    )
    scen_year = total_by_sy.pivot(
        index="scenario", columns="year", values="total_demand"
    ).fillna(0)

    feats = pd.DataFrame(index=scen_year.index)

    # 1. Total cumulative demand (sum across all years)
    feats["total_cumulative_demand"] = scen_year.sum(axis=1)

    # 2. Peak demand (max across years)
    feats["peak_demand"] = scen_year.max(axis=1)

    # 3. Mean demand 2029-2035 (early build-out period)
    early_years = [y for y in years if 2029 <= y <= 2035]
    feats["mean_demand_early"] = scen_year[early_years].mean(axis=1)

    # 4. Year of peak demand
    feats["year_of_peak"] = scen_year.idxmax(axis=1).astype(float)

    # 5. Demand growth rate: linear slope over years
    year_arr = np.array(years, dtype=float)
    def _slope(row):
        vals = row.values.astype(float)
        if vals.std() == 0:
            return 0.0
        return np.polyfit(year_arr, vals, 1)[0]
    feats["demand_slope"] = scen_year.apply(_slope, axis=1)

    # 6. Temporal concentration: early-half / late-half ratio
    mid = years[len(years) // 2]
    early = [y for y in years if y <= mid]
    late = [y for y in years if y > mid]
    early_sum = scen_year[early].sum(axis=1)
    late_sum = scen_year[late].sum(axis=1)
    feats["temporal_concentration"] = early_sum / (late_sum + 1)

    # 7. Mean CV across materials for this scenario
    cv_by_sm = demand.copy()
    cv_by_sm["cv"] = cv_by_sm["std"] / (cv_by_sm["mean"] + 1e-12)
    feats["mean_cv"] = cv_by_sm.groupby("scenario")["cv"].mean()

    # 8. Mean confidence interval width (p97 - p2) / mean  [95% CI: p2.5 to p97.5]
    demand_copy = demand.copy()
    demand_copy["ci_width"] = (demand_copy["p97"] - demand_copy["p2"]) / (demand_copy["mean"] + 1e-12)
    feats["mean_ci_width"] = demand_copy.groupby("scenario")["ci_width"].mean()

    # 9-11. Technology mix fractions from NREL at 2035
    nrel_2035 = nrel[nrel["t"] == 2035].set_index("scenario")
    mw_cols = [c for c in nrel.columns if c.endswith("_MW")]
    total_cap = nrel_2035[mw_cols].sum(axis=1)

    solar_cols = [c for c in mw_cols if "pv" in c or "csp" in c]
    wind_cols = [c for c in mw_cols if "wind" in c]
    storage_cols = [c for c in mw_cols if "battery" in c or "pumped" in c]

    feats["solar_fraction_2035"] = nrel_2035[solar_cols].sum(axis=1) / (total_cap + 1)
    feats["wind_fraction_2035"] = nrel_2035[wind_cols].sum(axis=1) / (total_cap + 1)
    feats["storage_fraction_2035"] = nrel_2035[storage_cols].sum(axis=1) / (total_cap + 1)

    # 12. Number of materials with demand > 0
    n_active = (
        demand[demand["mean"] > 0]
        .groupby("scenario")["material"]
        .nunique()
    )
    feats["n_active_materials"] = n_active

    # ── Supply-chain stress features (13–17) ─────────────────────────────
    # Quantify how much each scenario stresses production capacity and
    # supply-chain risk, using material-level risk data.

    us_production = _build_production_series(risk_data, thin_film_data)
    import_dep = _build_import_dependency_series(risk_data, thin_film_data)
    crc_risk = _build_crc_weighted_risk(risk_data)

    # Map CRC risk to demand material names
    crc_mapped = pd.Series(dtype=float)
    for demand_name, risk_name in DEMAND_TO_RISK.items():
        if risk_name in crc_risk.index:
            crc_mapped[demand_name] = crc_risk[risk_name]

    all_materials = sorted(demand["material"].unique())
    scenarios = sorted(demand["scenario"].unique())

    # Material-level risk vectors aligned to all_materials
    prod_vec = us_production.reindex(all_materials).fillna(0)
    dep_vec = import_dep.reindex(all_materials).fillna(1.0)
    crc_vec = crc_mapped.reindex(all_materials).fillna(5.0)
    crc_norm = crc_vec / 8.0  # normalize to 0-1

    # Per-scenario, per-year demand by material
    demand_pivot = demand.pivot_table(
        index=["scenario", "year"], columns="material", values="mean",
    ).fillna(0)

    stress_records = []
    for scenario in scenarios:
        scen_data = demand_pivot.loc[scenario]  # year × material

        yearly_stress = []
        yearly_exceedance = []
        yearly_import_demand = []

        for year in scen_data.index:
            row = scen_data.loc[year].reindex(all_materials).fillna(0)

            # Supply chain stress index:
            # demand-weighted average of (import_dep × CRC_risk/8)
            # Captures: high demand for high-risk, highly-imported materials
            risk_weight = dep_vec * crc_norm
            total_demand_year = row.sum()
            if total_demand_year > 0:
                weighted_risk = (row * risk_weight).sum() / total_demand_year
            else:
                weighted_risk = 0.0

            # Materials exceeding US production
            n_exceeding = int(((row > prod_vec) & (prod_vec > 0)).sum())

            # Import-exposed demand: sum(demand × import_dependency)
            import_demand = (row * dep_vec).sum()

            yearly_stress.append(weighted_risk)
            yearly_exceedance.append(n_exceeding)
            yearly_import_demand.append(import_demand)

        stress_records.append({
            "scenario": scenario,
            # 13. Mean supply-chain stress index (0-1, higher = riskier)
            "supply_chain_stress": np.mean(yearly_stress),
            # 14. Peak supply-chain stress (max across years)
            "peak_supply_chain_stress": np.max(yearly_stress),
            # 15. Mean materials exceeding US production per year
            "mean_n_exceeding_production": np.mean(yearly_exceedance),
            # 16. Peak materials exceeding US production (worst year)
            "peak_n_exceeding_production": np.max(yearly_exceedance),
            # 17. Total import-exposed demand (cumulative demand × import_dep)
            "total_import_exposed_demand": np.sum(yearly_import_demand),
        })

    stress_df = pd.DataFrame(stress_records).set_index("scenario")
    for col in stress_df.columns:
        feats[col] = stress_df[col]

    n_with_stress = (feats["supply_chain_stress"] > 0).sum()
    print(f"  Supply-chain stress features computed for {n_with_stress}/{len(feats)} scenarios")

    feats = feats.fillna(0)
    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# Material features
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_material_features(demand, risk_data, thin_film_data):
    """
    Construct features for each of the 31 materials, integrating
    risk/supply-chain data from risk_charts_inputs.xlsx and USGS 2023.

    Returns
    -------
    DataFrame with material index and numeric feature columns.
    """
    years = sorted(demand["year"].unique())
    all_materials = sorted(demand["material"].unique())

    # Build supply-chain series from risk data
    us_production = _build_production_series(risk_data, thin_film_data)
    import_dep = _build_import_dependency_series(risk_data, thin_film_data)
    global_reserves = _build_reserves_series(risk_data)
    domestic_reserves = _build_domestic_reserves_series(risk_data)
    crc_risk = _build_crc_weighted_risk(risk_data)
    reserves_crc = _build_reserves_by_crc(risk_data)
    sourcing = _build_crc_sourcing_breakdown(risk_data)

    # Map CRC risk from risk-material names to demand-material names
    crc_mapped = pd.Series(dtype=float)
    for demand_name, risk_name in DEMAND_TO_RISK.items():
        if risk_name in crc_risk.index:
            crc_mapped[demand_name] = crc_risk[risk_name]

    feats = pd.DataFrame(index=all_materials)
    feats.index.name = "material"

    # ── Demand-derived features ───────────────────────────────────────────

    # 1. Mean demand across all scenarios and years
    feats["mean_demand"] = demand.groupby("material")["mean"].mean()

    # 2. Peak demand (max across all scenarios and years)
    feats["peak_demand"] = demand.groupby("material")["mean"].max()

    # 3. Scenario CV: std of scenario-level total demands / mean
    scen_totals = (
        demand.groupby(["material", "scenario"])["mean"]
        .sum()
        .reset_index()
    )
    scen_cv = scen_totals.groupby("material")["mean"].agg(
        lambda x: x.std() / (x.mean() + 1e-12)
    )
    feats["scenario_cv"] = scen_cv

    # 4. Mean CI width (p97-p2)/mean across all scenarios/years  [95% CI: p2.5 to p97.5]
    demand_copy = demand.copy()
    demand_copy["ci_rel"] = (demand_copy["p97"] - demand_copy["p2"]) / (demand_copy["mean"] + 1e-12)
    feats["mean_ci_width"] = demand_copy.groupby("material")["ci_rel"].mean()

    # 5. Demand volatility: std across years (averaged over scenarios)
    year_pivot = demand.pivot_table(
        index=["material", "scenario"], columns="year", values="mean"
    ).fillna(0)
    vol = year_pivot.std(axis=1).groupby("material").mean()
    feats["demand_volatility"] = vol

    # 6. Demand growth slope across years
    year_arr = np.array(years, dtype=float)
    mat_year = demand.groupby(["material", "year"])["mean"].mean().unstack(fill_value=0)
    def _slope(row):
        vals = row.values.astype(float)
        if vals.std() == 0:
            return 0.0
        return np.polyfit(year_arr, vals, 1)[0]
    feats["demand_slope"] = mat_year.apply(_slope, axis=1)

    # ── Supply-chain / risk features ──────────────────────────────────────

    # 7. US domestic production (tonnes)
    feats["domestic_production"] = us_production.reindex(feats.index).fillna(0)

    # 8. Import dependency (0–1 fraction, 1 = fully imported)
    feats["import_dependency"] = import_dep.reindex(feats.index).fillna(1.0)

    # 9. CRC-weighted supply risk (0–8 scale)
    feats["crc_weighted_risk"] = crc_mapped.reindex(feats.index).fillna(5.0)

    # 10. Mean capacity ratio: mean scenario demand / US production
    mat_annual_mean = scen_totals.groupby("material")["mean"].mean()
    feats["mean_capacity_ratio"] = (
        mat_annual_mean / us_production
    ).reindex(feats.index).fillna(0)

    # 11. Max capacity ratio
    feats["max_capacity_ratio"] = (
        scen_totals.groupby("material")["mean"].max() / us_production
    ).reindex(feats.index).fillna(0)

    # 12. Exceedance frequency: fraction of scenarios where demand > production
    exceed = scen_totals.merge(
        us_production.rename("production").reset_index().rename(
            columns={"index": "material"}
        ),
        on="material", how="left",
    )
    exceed["exceeds"] = exceed["mean"] > exceed["production"]
    feats["exceedance_frequency"] = (
        exceed.groupby("material")["exceeds"].mean()
    ).reindex(feats.index).fillna(0)

    # ── Reserve Adequacy Features (based on cumulative 2026-2050 demand) ────
    #
    # These features compare total energy transition demand to known reserves,
    # answering: "What fraction of reserves does the transition consume?"

    # 13. Cumulative demand: total demand 2026-2050 (median across scenarios)
    #     Sum annual demand for each scenario, then take median across scenarios
    scenario_cumulative = year_pivot.sum(axis=1).groupby("material").median()
    feats["cumulative_demand"] = scenario_cumulative.reindex(feats.index).fillna(0)

    # 14. Reserve consumption %: (cumulative_demand / global_reserves) × 100
    #     Interpretation: "X% of known global reserves consumed by energy transition"
    #     Values >100% indicate reserves are insufficient
    global_res = global_reserves.reindex(feats.index).fillna(0)
    feats["reserve_consumption_pct"] = (
        feats["cumulative_demand"] / (global_res + 1e-12) * 100
    )
    feats.loc[global_res == 0, "reserve_consumption_pct"] = 0

    # 15. Domestic reserve coverage: domestic_reserves / cumulative_demand
    #     Interpretation: "Fraction of transition demand covered by US reserves"
    #     Values <1 indicate need for imports; values >1 indicate domestic sufficiency
    domestic_res = domestic_reserves.reindex(feats.index).fillna(0)
    feats["domestic_reserve_coverage"] = (
        domestic_res / (feats["cumulative_demand"] + 1e-12)
    )
    feats.loc[feats["cumulative_demand"] == 0, "domestic_reserve_coverage"] = 0

    # 16. Global reserve coverage: global_reserves / cumulative_demand
    #     Interpretation: "How many times over can global reserves meet transition demand"
    #     Values <1 indicate global shortage risk
    feats["global_reserve_coverage"] = (
        global_res / (feats["cumulative_demand"] + 1e-12)
    )
    feats.loc[feats["cumulative_demand"] == 0, "global_reserve_coverage"] = 0

    # ── Reserves-by-CRC features (Fig. 4 style) ─────────────────────────

    # 17. Fraction of global reserves in high-risk countries (CRC 5-7 + China)
    feats["reserves_high_risk_frac"] = (
        reserves_crc["reserves_high_risk_frac"].reindex(feats.index).fillna(0)
    )

    # 18. Fraction of global reserves in OECD + US
    feats["reserves_oecd_frac"] = (
        reserves_crc["reserves_oecd_frac"].reindex(feats.index).fillna(0)
    )

    # 19. Fraction of global reserves in China
    feats["reserves_china_frac"] = (
        reserves_crc["reserves_china_frac"].reindex(feats.index).fillna(0)
    )

    # ── CRC sourcing breakdown features (Fig. 3 style) ──────────────────

    # 20. Fraction of imports from China
    feats["import_china_frac"] = (
        sourcing["import_china_frac"].reindex(feats.index).fillna(0)
    )

    # 21. Fraction of imports from high-risk countries (CRC 5-7 + China)
    feats["import_high_risk_frac"] = (
        sourcing["import_high_risk_frac"].reindex(feats.index).fillna(0)
    )

    # 22. Fraction of imports from OECD countries
    feats["import_oecd_frac"] = (
        sourcing["import_oecd_frac"].reindex(feats.index).fillna(0)
    )

    # 23. Import concentration (HHI of country shares, 0-1)
    feats["import_hhi"] = (
        sourcing["import_hhi"].reindex(feats.index).fillna(0)
    )

    feats = feats.replace([np.inf, -np.inf], 0).fillna(0)

    # Report coverage
    n_with_prod = (feats["domestic_production"] > 0).sum()
    n_with_dep = (feats["import_dependency"] < 1.0).sum()
    n_with_crc = feats.index.isin(crc_mapped.index).sum()
    print(f"  Supply-chain coverage: production={n_with_prod}/31, "
          f"import_dep={n_with_dep}/31, CRC_risk={n_with_crc}/31")

    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# CLI test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading data...")
    demand = load_demand_data()
    nrel = load_nrel_data()
    risk_data = load_risk_data()
    thin_film = load_usgs_2023_thin_film()

    print("\nEngineering scenario features...")
    sf = engineer_scenario_features(demand, nrel)
    print(f"  {sf.shape[0]} scenarios × {sf.shape[1]} features")
    print(sf.describe().round(2))

    print("\nEngineering material features...")
    mf = engineer_material_features(demand, risk_data, thin_film)
    print(f"  {mf.shape[0]} materials × {mf.shape[1]} features")
    print(mf.describe().round(2))

    print("\nFeature engineering complete!")
