"""
Technology Mapping Configuration
=================================

This file maps Standard Scenarios capacity technologies to NREL material intensity technologies.

INSTRUCTIONS FOR EDITING:
1. Each capacity technology (left side) maps to one or more intensity technologies (right side)
2. For multiple mappings, specify weights that sum to 1.0
3. If a technology is missing, it will be skipped (logged as warning)

Format:
    CAPACITY_TECH: {
        'intensity_tech_1': weight1,
        'intensity_tech_2': weight2,
        ...
    }

Example:
    'upv': {
        'utility-scale solar pv': 0.6,  # 60% mono/poly-Si
        'CIGS': 0.2,                     # 20% thin film CIGS
        'CdTe': 0.2                      # 20% thin film CdTe
    }
"""


# ============================================================================
# INTENSITY DATA TECHNOLOGY CONSOLIDATION
# ============================================================================
# The raw intensity_data.csv splits some technologies into separate cell-specific
# and balance-of-system (BOS) entries. This mapping consolidates them during
# preprocessing so that each technology has complete material coverage.
#
# Background:
#   - 'CDTE' (15 rows): BOS materials (Aluminum, Cement, Copper, Glass, Steel)
#     for CdTe thin-film panels. Separate from 'CdTe' (112 rows) which has
#     cell-specific materials (Tellurium, Cadmium, etc.).
#   - 'ASIGE' (15 rows): BOS materials for a-Si/Germanium panels.
#     Identical values to CDTE — same generic thin-film BOS source.
#   - 'CIGS' already includes both cell and BOS materials (no consolidation needed).
#   - 'utility-scale solar pv' already includes both (no consolidation needed).
#
# Without consolidation, the CdTe fraction of UPV is missing BOS material demand
# (~15% underestimate of structural materials from solar PV).

TECHNOLOGY_CONSOLIDATION = {
    'CDTE': 'CdTe',     # BOS materials → merge into CdTe
    'ASIGE': 'a-Si',    # BOS materials → merge into a-Si
}


TECHNOLOGY_MAPPING = {
    # ============================================================================
    # SOLAR TECHNOLOGIES
    # ============================================================================

    'upv': {
        # Utility-scale PV: mix of crystalline Si and thin film
        #
        # Market share sources (2024):
        #   - Fraunhofer ISE Photovoltaics Report: c-Si ~95-98% of global production
        #   - IEA-PVPS: CdTe ~2-5% global, ~22% of US utility-scale (First Solar)
        #   - CIGS: <1% global (last major manufacturer ceased production 2021)
        #   - a-Si: negligible (<0.1%)
        #
        # US utility-scale context (DOE/EIA, through 2023):
        #   - Thin film accounted for ~24% of US utility-scale deployments
        #   - CdTe (First Solar) represents ~97% of US thin-film capacity
        #
        # Weights below reflect a blended estimate acknowledging:
        #   1. Global market heavily favors c-Si (~95%)
        #   2. US utility-scale has higher thin-film penetration (~24%)
        #   3. CdTe dominates thin film; CIGS is marginal
        'utility-scale solar pv': 0.90,  # Crystalline Si (mono/poly)
        'CdTe': 0.07,                    # Thin film — First Solar
        'CIGS': 0.03,                     # Thin film — marginal market share
    },
    
    'distpv': {
        # Distributed PV: mostly rooftop, primarily crystalline Si
        'Solar Distributed': 1.0
    },
    
    'csp': {
        # Concentrated Solar Power
        'CSP': 1.0
    },
    
    # ============================================================================
    # WIND TECHNOLOGIES
    # ============================================================================
    
    'wind_onshore': {
        'onshore wind': 1.0
    },
    
    'wind_offshore': {
        'offshore wind': 1.0
    },
    
    # ============================================================================
    # FOSSIL FUEL TECHNOLOGIES
    # ============================================================================
    
    'coal': {
        'Coal': 1.0
    },
    
    'coal_ccs': {
        'Coal CCS': 1.0
    },
    
    'gas_cc': {
        # Natural Gas Combined Cycle
        'NGCC': 1.0
    },
    
    'gas_ct': {
        # Natural Gas Combustion Turbine
        'NGGT': 1.0
    },
    
    'gas_cc_ccs': {
        # Natural Gas Combined Cycle with CCS
        'Gas CCS': 1.0
    },
    
    'h2-ct': {
        # Hydrogen Combustion Turbine - use gas turbine as proxy
        'NGGT': 1.0
    },
    
    # ============================================================================
    # BIOMASS TECHNOLOGIES
    # ============================================================================
    
    'bio': {
        'Biomass': 1.0
    },
    
    'bio-ccs': {
        'Bio CCS': 1.0
    },
    
    # ============================================================================
    # NUCLEAR TECHNOLOGIES
    # ============================================================================
    
    'nuclear': {
        'Nuclear New': 1.0
    },
    
    'nuclear_smr': {
        # Small Modular Reactors - use Nuclear New as proxy
        'Nuclear New': 1.0
    },
    
    # ============================================================================
    # HYDRO AND GEOTHERMAL
    # ============================================================================
    
    'hydro': {
        'Hydro': 1.0
    },
    
    'pumped-hydro': {
        # Pumped storage - use regular hydro as proxy
        'Hydro': 1.0
    },
    
    'geo': {
        'Geothermal': 1.0
    },
    
    # ============================================================================
    # STORAGE AND OTHER TECHNOLOGIES
    # ============================================================================
    
    # Note: Battery technologies don't have intensity data in current dataset
    # These will be skipped with warnings
    'battery_4': {},   # 4-hour battery storage
    'battery_8': {},   # 8-hour battery storage
    
    # Note: These technologies don't have intensity data
    'o-g-s': {},       # Oil-gas-steam - deprecated technology
    'dac': {},         # Direct Air Capture - no intensity data yet
    'electrolyzer': {} # Electrolyzer - no intensity data yet
}


# ============================================================================
# TECHNOLOGY LIFETIMES (years)
# ============================================================================
# Used for stock-flow model to calculate retirements

TECHNOLOGY_LIFETIMES = {
    # Solar
    'upv': 30,
    'distpv': 30,
    'csp': 30,
    
    # Wind
    'wind_onshore': 25,
    'wind_offshore': 25,
    
    # Fossil
    'coal': 40,
    'coal_ccs': 40,
    'gas_cc': 30,
    'gas_ct': 30,
    'gas_cc_ccs': 30,
    'h2-ct': 30,
    
    # Biomass
    'bio': 30,
    'bio-ccs': 30,
    
    # Nuclear
    'nuclear': 60,
    'nuclear_smr': 60,
    
    # Hydro and Geo
    'hydro': 80,
    'pumped-hydro': 80,
    'geo': 30,
    
    # Storage and Other
    'battery_4': 15,
    'battery_8': 15,
    'o-g-s': 30,
    'dac': 30,
    'electrolyzer': 20
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_mapping():
    """Validate that all weights sum to 1.0"""
    errors = []
    warnings = []
    
    for cap_tech, intensity_mapping in TECHNOLOGY_MAPPING.items():
        if not intensity_mapping:
            warnings.append(f"  - {cap_tech}: No mapping (will be skipped)")
            continue
            
        total_weight = sum(intensity_mapping.values())
        if abs(total_weight - 1.0) > 0.001:
            errors.append(
                f"  - {cap_tech}: Weights sum to {total_weight:.3f}, not 1.0"
            )
    
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(w)
    
    if errors:
        print("\nERRORS:")
        for e in errors:
            print(e)
        raise ValueError("Technology mapping validation failed")
    
    print("\n✓ Technology mapping validation passed")
    return True


def get_intensity_technologies(capacity_tech):
    """
    Get the intensity technology mappings for a capacity technology.
    
    Returns:
        dict: {intensity_tech: weight}
    """
    return TECHNOLOGY_MAPPING.get(capacity_tech, {})


def get_lifetime(capacity_tech):
    """Get the lifetime (years) for a capacity technology"""
    return TECHNOLOGY_LIFETIMES.get(capacity_tech, 30)  # Default 30 years


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("TECHNOLOGY MAPPING VALIDATION")
    print("="*80)
    
    validate_mapping()
    
    print("\n" + "="*80)
    print("EXAMPLE MAPPINGS")
    print("="*80)
    
    # Example: UPV mapping
    print("\nUtility PV (upv):")
    for intensity_tech, weight in get_intensity_technologies('upv').items():
        print(f"  → {intensity_tech}: {weight*100:.1f}%")
    print(f"  Lifetime: {get_lifetime('upv')} years")
    
    # Example: Wind mapping
    print("\nOnshore Wind (wind_onshore):")
    for intensity_tech, weight in get_intensity_technologies('wind_onshore').items():
        print(f"  → {intensity_tech}: {weight*100:.1f}%")
    print(f"  Lifetime: {get_lifetime('wind_onshore')} years")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    total = len(TECHNOLOGY_MAPPING)
    mapped = sum(1 for v in TECHNOLOGY_MAPPING.values() if v)
    unmapped = total - mapped
    print(f"Total capacity technologies: {total}")
    print(f"  Mapped: {mapped}")
    print(f"  Unmapped: {unmapped}")
