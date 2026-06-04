"""
IME Vessel-to-Cargo Matching Engine
====================================
Matches open vessels to cargo requirements by:
- Region/port proximity
- DWT compatibility
- Date overlap (laycan vs open date)
- Cargo type suitability
"""

import sqlite3
import os
import re
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'ime.db')

# Port-to-region mapping
PORT_REGIONS = {
    # Pacific / East Asia
    'china': 'East Asia', 'guangzhou': 'East Asia', 'tianjin': 'East Asia',
    'shanghai': 'East Asia', 'qingdao': 'East Asia', 'dalian': 'East Asia',
    'xiamen': 'East Asia', 'beilun': 'East Asia',
    'japan': 'East Asia', 'korea': 'East Asia', 'busan': 'East Asia',
    'taiwan': 'East Asia', 'hong kong': 'East Asia',
    # Southeast Asia
    'singapore': 'Southeast Asia', 'malaysia': 'Southeast Asia',
    'port kelang': 'Southeast Asia', 'indonesia': 'Southeast Asia',
    'vietnam': 'Southeast Asia', 'vung ang': 'Southeast Asia',
    'thailand': 'Southeast Asia', 'koh si chang': 'Southeast Asia',
    'philippines': 'Southeast Asia', 'manila': 'Southeast Asia',
    # South Asia
    'india': 'South Asia', 'kandla': 'South Asia', 'mundra': 'South Asia',
    'krishnapatnam': 'South Asia', 'chennai': 'South Asia', 'mumbai': 'South Asia',
    'bangladesh': 'South Asia', 'chittagong': 'South Asia',
    'pakistan': 'South Asia', 'karachi': 'South Asia',
    # Middle East / Indian Ocean
    'oman': 'Middle East', 'sohar': 'Middle East', 'muscat': 'Middle East',
    'uae': 'Middle East', 'dubai': 'Middle East',
    'saudi': 'Middle East', 'jeddah': 'Middle East',
    'iraq': 'Middle East', 'bandar': 'Middle East',
    'qatar': 'Middle East', 'doha': 'Middle East',
    'aqaba': 'Middle East', 'jordan': 'Middle East',
    # Europe
    'rotterdam': 'Europe', 'netherlands': 'Europe',
    'bilbao': 'Europe', 'spain': 'Europe',
    'med': 'Mediterranean', 'mediterranean': 'Mediterranean',
    'egypt': 'Mediterranean', 'suez': 'Mediterranean',
    'turkey': 'Mediterranean', 'iskenderun': 'Mediterranean',
    # Africa
    'tanzania': 'East Africa', 'dar es salaam': 'East Africa',
    'kenya': 'East Africa', 'mombasa': 'East Africa',
    'south africa': 'Southern Africa', 'durban': 'Southern Africa',
    'west africa': 'West Africa', 'nigeria': 'West Africa',
    # Americas
    'brazil': 'South America', 'santos': 'South America',
    'ecsa': 'South America',
    'gulf': 'US Gulf', 'houston': 'US Gulf',
    'vancouver': 'North Pacific', 'australia': 'Australia/Pacific',
    'newcastle': 'Australia/Pacific', 'port hedland': 'Australia/Pacific',
    'dampier': 'Australia/Pacific',
}

DWT_RANGES = {
    'handysize': (25000, 40000),
    'handymax': (40000, 50000),
    'supramax': (50000, 65000),
    'ultramax': (60000, 67000),
    'smx': (50000, 65000),
    'umx': (60000, 67000),
    'supra': (50000, 65000),
    'ultra': (60000, 67000),
    'panamax': (65000, 85000),
    'kamsarmax': (78000, 82000),
    'post-panamax': (85000, 105000),
}


def get_region(port_or_text):
    if not port_or_text:
        return 'Unknown'
    text = str(port_or_text).lower()
    for key, region in PORT_REGIONS.items():
        if key in text:
            return region
    return 'Global'


def parse_dwt(dwt_str):
    if not dwt_str:
        return None
    # Handle "93K" style
    m = re.search(r'(\d+\.?\d*)K', str(dwt_str), re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1000
    m = re.search(r'[\d,]+', str(dwt_str).replace(',', ''))
    if m:
        return float(m.group(0).replace(',', ''))
    return None


def parse_size_pref(pref_str):
    """Return (min_dwt, max_dwt) from vessel size preference."""
    if not pref_str:
        return (25000, 120000)
    text = str(pref_str).lower()
    min_dwt, max_dwt = 25000, 120000
    for name, (lo, hi) in DWT_RANGES.items():
        if name in text:
            min_dwt = min(min_dwt, lo)
            max_dwt = max(max_dwt, hi)
    if min_dwt == 25000 and max_dwt == 120000:
        return (25000, 120000)
    return (min_dwt - 5000, max_dwt + 5000)


def score_match(vessel, cargo, cargo_type):
    """
    Score a vessel-cargo pair.
    Returns (score 0-100, reasons list, match_type)
    """
    score = 0
    reasons = []

    # --- Region match ---
    vessel_region = get_region(vessel.get('open_port', ''))

    if cargo_type == 'cargo_vc':
        cargo_region = get_region(cargo.get('loading_port', ''))
    else:  # cargo_tc
        cargo_region = get_region(cargo.get('delivery_port', ''))

    if vessel_region != 'Unknown' and vessel_region != 'Global' and vessel_region == cargo_region:
        score += 40
        reasons.append(f'Same region: {vessel_region}')
    elif vessel_region == 'Global' or cargo_region == 'Global':
        score += 15
        reasons.append('Global positioning')
    else:
        # Neighboring regions bonus
        neighbors = {
            'East Asia': ['Southeast Asia'],
            'Southeast Asia': ['East Asia', 'South Asia'],
            'South Asia': ['Southeast Asia', 'Middle East'],
            'Middle East': ['South Asia', 'East Africa'],
            'Europe': ['Mediterranean'],
            'Mediterranean': ['Europe', 'Middle East'],
        }
        if cargo_region in neighbors.get(vessel_region, []):
            score += 20
            reasons.append(f'Adjacent regions: {vessel_region} ↔ {cargo_region}')

    # --- DWT match ---
    vessel_dwt = parse_dwt(vessel.get('vessel_size_dwt'))

    if cargo_type == 'cargo_vc':
        # VC: match by cargo quantity
        qty_str = cargo.get('quantity', '')
        qty = parse_dwt(qty_str)
        if vessel_dwt and qty:
            if vessel_dwt * 0.8 <= qty <= vessel_dwt * 1.1:
                score += 30
                reasons.append(f'Optimal DWT match ({int(vessel_dwt):,} DWT for {int(qty):,} MT cargo)')
            elif vessel_dwt * 0.5 <= qty <= vessel_dwt * 1.3:
                score += 15
                reasons.append(f'DWT compatible ({int(vessel_dwt):,} DWT)')
        elif vessel_dwt:
            score += 10
            reasons.append(f'Vessel DWT: {int(vessel_dwt):,}')
    else:
        # TC: match by size preference
        size_pref = cargo.get('vessel_size_pref', '')
        min_dwt, max_dwt = parse_size_pref(size_pref)
        if vessel_dwt:
            if min_dwt <= vessel_dwt <= max_dwt:
                score += 30
                reasons.append(f'DWT within preferred range ({size_pref or "Any size"})')
            elif abs(vessel_dwt - min_dwt) < 10000 or abs(vessel_dwt - max_dwt) < 10000:
                score += 15
                reasons.append(f'DWT close to preferred range')

    # --- Date/Laycan overlap ---
    laycan = cargo.get('laycan', '')
    open_date = vessel.get('open_date', '')
    if laycan and open_date:
        score += 20
        reasons.append('Date overlap possible')
    elif laycan or open_date:
        score += 5

    # --- Cargo type suitability ---
    cargo_name = str(cargo.get('cargo_name', '') or cargo.get('cargo_type', '')).lower()
    vessel_type = str(vessel.get('vessel_type', 'Bulk Carrier')).lower()
    if 'bulk' in vessel_type or 'bulk' in cargo_name or 'dry' in cargo_name:
        score += 10
        reasons.append('Cargo type compatible (Dry Bulk)')

    return min(score, 100), reasons


def get_vessel_cargo_matches():
    """Generate vessel-cargo match opportunities."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("SELECT * FROM tonnage ORDER BY id DESC LIMIT 30")
        vessels = [dict(r) for r in c.fetchall()]

        c.execute("SELECT * FROM cargo_vc ORDER BY id DESC LIMIT 30")
        cargoes_vc = [dict(r) for r in c.fetchall()]

        c.execute("SELECT * FROM cargo_tc ORDER BY id DESC LIMIT 30")
        cargoes_tc = [dict(r) for r in c.fetchall()]

        conn.close()
    except:
        return {'matches': [], 'stats': {}, 'vessels': [], 'cargoes_vc': [], 'cargoes_tc': []}

    matches = []

    # Match vessels against VC cargoes
    for vessel in vessels:
        for cargo in cargoes_vc:
            score, reasons = score_match(vessel, cargo, 'cargo_vc')
            if score >= 30:
                matches.append({
                    'type': 'VC',
                    'type_label': 'Voyage Charter',
                    'vessel_name': vessel.get('vessel_name', 'Unknown Vessel'),
                    'vessel_dwt': vessel.get('vessel_size_dwt', 'N/A'),
                    'vessel_port': vessel.get('open_port', 'N/A'),
                    'vessel_date': vessel.get('open_date', 'N/A'),
                    'cargo_name': cargo.get('cargo_name', 'N/A'),
                    'cargo_qty': cargo.get('quantity', 'N/A'),
                    'load_port': cargo.get('loading_port', 'N/A'),
                    'disch_port': cargo.get('discharge_port', 'N/A'),
                    'laycan': cargo.get('laycan', 'N/A'),
                    'score': score,
                    'reasons': reasons,
                    'vessel_region': get_region(vessel.get('open_port', '')),
                    'cargo_region': get_region(cargo.get('loading_port', '')),
                    'commission': cargo.get('commission_pct', 'N/A'),
                })

    # Match vessels against TC cargoes
    for vessel in vessels:
        for cargo in cargoes_tc:
            score, reasons = score_match(vessel, cargo, 'cargo_tc')
            if score >= 30:
                matches.append({
                    'type': 'TC',
                    'type_label': 'Time Charter',
                    'vessel_name': vessel.get('vessel_name', 'Unknown Vessel'),
                    'vessel_dwt': vessel.get('vessel_size_dwt', 'N/A'),
                    'vessel_port': vessel.get('open_port', 'N/A'),
                    'vessel_date': vessel.get('open_date', 'N/A'),
                    'cargo_name': cargo.get('cargo_name', 'N/A'),
                    'cargo_qty': cargo.get('duration', 'N/A'),
                    'load_port': cargo.get('delivery_port', 'N/A'),
                    'disch_port': cargo.get('redelivery_port', 'N/A'),
                    'laycan': cargo.get('laycan', 'N/A'),
                    'score': score,
                    'reasons': reasons,
                    'vessel_region': get_region(vessel.get('open_port', '')),
                    'cargo_region': get_region(cargo.get('delivery_port', '')),
                    'commission': cargo.get('commission_pct', 'N/A'),
                })

    # Sort by score descending
    matches.sort(key=lambda x: x['score'], reverse=True)

    stats = {
        'total_matches': len(matches),
        'high_confidence': len([m for m in matches if m['score'] >= 70]),
        'medium_confidence': len([m for m in matches if 50 <= m['score'] < 70]),
        'vc_matches': len([m for m in matches if m['type'] == 'VC']),
        'tc_matches': len([m for m in matches if m['type'] == 'TC']),
        'vessels_available': len(vessels),
        'cargoes_available': len(cargoes_vc) + len(cargoes_tc),
    }

    return {
        'matches': matches[:50],  # Top 50
        'stats': stats,
        'vessels': vessels,
        'cargoes_vc': cargoes_vc,
        'cargoes_tc': cargoes_tc,
    }
