"""
IME Email Parser Engine
=======================
Rule-based email classification and structured data extraction.
No external LLM APIs used - pure regex + keyword scoring.
"""

import re
import json
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────

TONNAGE_KEYWORDS = [
    r'\bopen\b', r'\bdwt\b', r'\bbuilt\b', r'\bflag\b', r'\bclass\b',
    r'\bloa\b', r'\bbeam\b', r'\bgrain\b', r'\bscrubber\b', r'\bbulk carrier\b',
    r'\bho/ha\b', r'\bcrane\b', r'\bspeed.*cons\b', r'\bmv\b', r'\bm/v\b',
    r'\bopen port\b', r'\bopen.*\d{1,2}.*\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
    r'\bsdbc\b', r'\bsdstbc\b', r'\bvessel.*open\b', r'\btons.*dwt\b'
]

CARGO_TC_KEYWORDS = [
    r'\bdelivery\b', r'\bredelivery\b', r'\btct\b', r'\btime charter\b',
    r'\bduration\b', r'\bdays wog\b', r'\b\d+[-–]\d+ (years|months|days)\b',
    r'\bdely\b', r'\bredel\b', r'\btry.*period\b', r'\badc\b',
    r'\baddcom\b', r'\bindex\b.*\bflat\b', r'\b1 tct\b'
]

CARGO_VC_KEYWORDS = [
    r'\bload port\b', r'\bload\b.*\bport\b', r'\bdischarge port\b', r'\bdischarge\b.*\bport\b',
    r'\blaycan\b', r'\blp\b', r'\bdp\b', r'\bpol\b', r'\bpod\b',
    r'\bpwwd\b', r'\bfios\b', r'\bfhinc\b', r'\bmolco\b', r'\bmoloo\b',
    r'\bcqd\b', r'\bcargo.*offer\b', r'\bfirm.*cargo\b', r'\burea\b',
    r'\biron.*ore\b', r'\bcoal\b', r'\bgrain\b.*\bcargo\b', r'\bmts\b.*\bload\b',
    r'\bpct.*ttl\b', r'\bttl.*comm\b'
]


def score_category(text: str, patterns: list) -> int:
    text_lower = text.lower()
    score = 0
    for pat in patterns:
        if re.search(pat, text_lower):
            score += 1
    return score


def classify_email(text: str) -> str:
    """Returns: 'tonnage' | 'cargo_vc' | 'cargo_tc'"""
    t_score = score_category(text, TONNAGE_KEYWORDS)
    vc_score = score_category(text, CARGO_VC_KEYWORDS)
    tc_score = score_category(text, CARGO_TC_KEYWORDS)

    # TC has unique markers - give it priority if detected
    if tc_score >= 2 and tc_score >= vc_score:
        return 'cargo_tc'
    if vc_score > t_score and vc_score >= 2:
        return 'cargo_vc'
    if t_score >= 2:
        return 'tonnage'

    # Fallback: highest score wins
    scores = {'tonnage': t_score, 'cargo_vc': vc_score, 'cargo_tc': tc_score}
    return max(scores, key=scores.get)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clean(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return re.sub(r'\s+', ' ', s.strip().strip('.,;:-').strip())


def find_account_name(text: str) -> Optional[str]:
    """Extract account/company name from email signature or header."""
    patterns = [
        r'ACC\s+([A-Z][A-Z\s&]+(?:LTD|LIMITED|INC|CORP|CO\.|COMPANY|DMCC|SHIPPING|MARITIME)?\.?)',
        r'A/C\s+([A-Z][A-Z\s&]+(?:LTD|LIMITED|INC|CORP|CO\.|COMPANY|DMCC|SHIPPING|MARITIME)?\.?)',
        r'(?:FROM|ON BEHALF OF|RE:?)\s*:?\s*([A-Z][A-Z\s&]+(?:LTD|LIMITED|INC|CORP|MARITIME|SHIPPING))',
        r'([A-Z][A-Z\s]+(?:MARITIME|SHIPPING|CHARTERING|OCEAAN|OCEAN)\s+(?:INC|LTD|CO|CORP|DMCC)\.?)',
        r'(?:BEST REGARDS|REGARDS)\s*\n+[^\n]+\n+([A-Z][A-Z\s]+(?:LTD|INC|MARITIME|DMCC|SHIPPING))',
        r'([A-Z][A-Z\s]+MARITIME\s+INC\.?)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = clean(m.group(1))
            if val and len(val) > 3:
                return val.title()

    # Try email domain extraction
    m = re.search(r'[\w.+-]+@([\w-]+)\.\w+', text)
    if m:
        domain = m.group(1).replace('-', ' ').title()
        return domain

    return None


def find_laycan(text: str) -> Optional[str]:
    patterns = [
        r'(?:LAYCAN|LC|LAY\s*CAN)\s*:?\s*(\d{1,2}[-–/]\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*\s*\d{0,4})',
        r'(?:LAYCAN|LC|LAY\s*CAN)\s*:?\s*(\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*\s*[-–]\s*\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*\s*\d{0,4})',
        r'(?:LAYCAN|LC|LAY\s*CAN)\s*:?\s*((?:EARLY|MID|LATE|END)\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*\s*\d{0,4})',
        r'(?:LAYCAN|LC)\s*:?\s*(\d{1,2}[-–]\d{1,2}\s+\w+\s*\d{4})',
        r'(?:LAYCAN)\s*:?\s*([A-Z][\w\s\-/]+\d{4})',
        r'(?:25|26|27|28|29|30|31|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24)\s*[-–]\s*(?:25|26|27|28|29|30|31|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24)\s+(?:JUNE|JULY|AUG|AUGUST|JUNE|MAY|APRIL|MARCH)\s*\d{0,4}',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return clean(m.group(1))
            except:
                return clean(m.group(0))
    return None


# ─────────────────────────────────────────────
# TONNAGE EXTRACTOR
# ─────────────────────────────────────────────

def extract_vessels_from_list(text: str) -> list:
    """Extract multiple vessels from a position list format."""
    vessels = []
    # Pattern: MV NAME DWT XXXXX OPEN PORT, COUNTRY O/A DATE
    list_pattern = r'(?:MV|M/V)\s+([A-Z][A-Z\s]+?)\s+DWT\s+([\d,\.]+)\s+(?:MT\s+)?OPEN\s+([A-Z][A-Z\s,\.]+?)\s+O/A\s+([\w\s,/\-]+?)(?=\n|$)'
    for m in re.finditer(list_pattern, text, re.IGNORECASE):
        name = clean(m.group(1))
        dwt = clean(m.group(2).replace(',', ''))
        port = clean(m.group(3))
        date = clean(m.group(4))
        if name:
            vessels.append({
                'vessel_name': name.title(),
                'vessel_size_dwt': dwt,
                'open_port': port.title() if port else None,
                'open_date': date,
            })
    return vessels


def extract_vessel_details(text: str) -> dict:
    """Extract detailed vessel specs from vessel particular block."""
    d = {}

    # Vessel name
    for pat in [
        r'M/V\s*:?\s*([A-Z][A-Z\s]+?)(?:\n|DWT|IMO|BUILT|\d{4})',
        r'MV\s+([A-Z][A-Z\s]+?)(?:\n|DWT|IMO|BUILT|\d{4})',
        r'(?:VESSEL NAME|VSL NAME)\s*:?\s*([A-Z][A-Z\s]+)',
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            d['vessel_name'] = clean(m.group(1)).title()
            break

    # DWT
    m = re.search(r'(?:DWT|DEAD\s*WEIGHT)\s*:?\s*([\d,\.]+)\s*(?:MT|MTDW|MTDWT)?', text, re.IGNORECASE)
    if m:
        d['vessel_size_dwt'] = clean(m.group(1).replace(',', ''))

    # Built year
    m = re.search(r'(?:BUILT|BLT|YEAR\s+BUILT)\s*:?\s*(\d{4})', text, re.IGNORECASE)
    if m:
        d['built_year'] = m.group(1)

    # Flag
    m = re.search(r'FLAG\s*:?\s*([A-Z][A-Z\s]+?)(?:\n|CLASS|BUILT|FLAG)', text, re.IGNORECASE)
    if m:
        d['flag'] = clean(m.group(1)).title()

    # Class
    m = re.search(r'CLASS(?:IFICATION)?\s*:?\s*([A-Z][A-Z\s]+?)(?:\n|FLAG|BUILT|LOA)', text, re.IGNORECASE)
    if m:
        d['classification'] = clean(m.group(1)).upper()

    # Vessel type
    for pat in [r'(BULK\s+CARRIER)', r'(TANKER)', r'(CONTAINER)', r'(GENERAL\s+CARGO)']:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            d['vessel_type'] = clean(m.group(1)).title()
            break
    if 'vessel_type' not in d:
        d['vessel_type'] = 'Bulk Carrier'  # default for this domain

    # Open port + date from "OPEN X, COUNTRY DATE" pattern
    m = re.search(
        r'(?:OPEN|OPENING)\s+([A-Z][A-Z\s,\.]+?)[,\s]+(?:O/A\s*|ABT\s*)?(\d{1,2}[-–\s]?\d{0,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*\s*\d{0,4}|\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*)',
        text, re.IGNORECASE
    )
    if m:
        d['open_port'] = clean(m.group(1)).title()
        d['open_date'] = clean(m.group(2))

    return d


def extract_tonnage(text: str) -> list:
    """Extract all vessel records from a tonnage email."""
    account = find_account_name(text)
    results = []

    # First try: position list format (multiple vessels in a list)
    listed = extract_vessels_from_list(text)

    # Also get detailed blocks
    # Split by vessel separator lines
    blocks = re.split(r'-{10,}|={10,}', text)

    # Try each block for vessel details
    detail_vessels = []
    for block in blocks:
        if re.search(r'\b(DWT|DEAD\s*WEIGHT|LOA|BUILT|FLAG)\b', block, re.IGNORECASE):
            d = extract_vessel_details(block)
            if d.get('vessel_name') or d.get('vessel_size_dwt'):
                detail_vessels.append(d)

    # Merge: listed vessels get enriched with detail blocks
    if listed:
        for lv in listed:
            # Find matching detail block
            enriched = dict(lv)
            for dv in detail_vessels:
                name_lv = (lv.get('vessel_name') or '').upper().replace(' ', '')
                name_dv = (dv.get('vessel_name') or '').upper().replace(' ', '')
                if name_lv and name_dv and (name_lv in name_dv or name_dv in name_lv):
                    enriched.update({k: v for k, v in dv.items() if v and not enriched.get(k)})
                    break
            enriched['account_name'] = account
            results.append(enriched)
    elif detail_vessels:
        for dv in detail_vessels:
            dv['account_name'] = account
            results.append(dv)
    else:
        # Single vessel fallback
        d = extract_vessel_details(text)
        d['account_name'] = account
        if d:
            results.append(d)

    # Fallback: simple SARONIC pattern
    saronic_pat = r'([A-Z][A-Z\s]+)\s*\((\d+K)[^)]*\)[^–\-]*[–\-]\s*OPEN\s+([A-Z][A-Z\s,]+),\s*([A-Z]+)\s+([\d\-–\s]+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*)'
    for m in re.finditer(saronic_pat, text, re.IGNORECASE):
        name = clean(m.group(1)).title()
        size = m.group(2)
        port = clean(m.group(3)).title()
        country = m.group(4).title()
        date = clean(m.group(5))
        results.append({
            'vessel_name': name,
            'vessel_size_dwt': size,
            'open_port': f"{port}, {country}",
            'open_date': date,
            'account_name': account,
            'vessel_type': 'Bulk Carrier',
        })

    return results if results else [{'account_name': account, 'raw_note': 'Could not parse vessel details'}]


# ─────────────────────────────────────────────
# CARGO VC EXTRACTOR
# ─────────────────────────────────────────────

def extract_cargo_vc(text: str) -> list:
    cargoes = []
    # Split into individual cargo offers by +++ or --- separators
    blocks = re.split(r'\+{5,}|[-]{20,}', text)

    for block in blocks:
        if not re.search(r'\b(cargo|load|discharge|laycan|lp|dp|pol|pod|mts|tons)\b', block, re.IGNORECASE):
            continue

        c = {}
        c['account_name'] = find_account_name(block) or find_account_name(text)

        # Cargo name / quantity
        for pat in [
            r'([\d,]+\s*(?:[-–]\s*[\d,]+)?\s*(?:MTS|MT|TONS?|TONNES?))\s+(?:\d+PCT\s+)?(?:MOLCO\s+|MOLOO\s+)?([A-Z][A-Z\s]+?)(?:\n|LP|LOAD|IN BULK)',
            r'CARGO\s*:?\s*([^\n]+)',
            r'(\d[\d,\s]+MTS?\s+[A-Z][A-Z\s]+)',
        ]:
            m = re.search(pat, block, re.IGNORECASE)
            if m:
                try:
                    c['cargo_name'] = clean(m.group(2)).title() if m.lastindex >= 2 else clean(m.group(1)).title()
                    c['quantity'] = clean(m.group(1)) if m.lastindex >= 2 else None
                except:
                    c['cargo_name'] = clean(m.group(1)).title()
                break

        # Loading port
        for pat in [
            r'LOAD(?:ING)?\s+PORT\s*:?\s*([A-Z][A-Z\s,\.]+?)(?:\n|DISCH|DISC|LOAD\s+RATE|LAYCAN)',
            r'LP\s*:?\s*([A-Z][A-Z\s,\.]+?)(?:\n|DP|DISCH)',
            r'POL\s*:?\s*([A-Z][A-Z\s,\.]+?)(?:\n|POD|DISCH)',
        ]:
            m = re.search(pat, block, re.IGNORECASE)
            if m:
                c['loading_port'] = clean(m.group(1)).title()
                break

        # Discharge port
        for pat in [
            r'DISCH(?:ARGE)?\s+PORT\s*:?\s*([A-Z][A-Z\s,\+\.]+?)(?:\n|LOAD\s+RATE|LAYCAN|COMM)',
            r'DP\s*:?\s*([A-Z][A-Z\s,\+\.]+?)(?:\n|LAYCAN|COMM)',
            r'POD\s*:?\s*([A-Z][A-Z\s,\+\.]+?)(?:\n|LAYCAN|COMM)',
        ]:
            m = re.search(pat, block, re.IGNORECASE)
            if m:
                c['discharge_port'] = clean(m.group(1)).title()
                break

        # Laycan
        c['laycan'] = find_laycan(block)

        # Cargo type
        cargo_types = ['BULK', 'BREAKBULK', 'LIQUID', 'DRY', 'GRAIN', 'COAL', 'IRON ORE',
                       'STEEL', 'FERTILIZER', 'UREA', 'CLINKER', 'SLAG', 'HRC']
        for ct in cargo_types:
            if re.search(ct, block, re.IGNORECASE):
                c['cargo_type'] = ct.title()
                break
        if 'cargo_type' not in c:
            c['cargo_type'] = 'Dry Bulk'

        # Commission
        m = re.search(r'(\d+\.?\d*)\s*(?:PCT|%)\s*(?:TTL|TOTAL|ADDCOM|ADC)', block, re.IGNORECASE)
        if m:
            c['commission_pct'] = m.group(1) + '%'

        if c.get('loading_port') or c.get('cargo_name') or c.get('laycan'):
            cargoes.append(c)

    return cargoes if cargoes else [{'account_name': find_account_name(text), 'raw_note': 'Cargo details unclear'}]


# ─────────────────────────────────────────────
# CARGO TC EXTRACTOR
# ─────────────────────────────────────────────

def extract_cargo_tc(text: str) -> list:
    cargoes = []
    # Split on +++ or --- block separators
    blocks = re.split(r'\+{5,}|[-]{30,}', text)

    for block in blocks:
        if not re.search(r'\b(delivery|dely|redelivery|redel|tct|duration|days wog|laycan|lc)\b', block, re.IGNORECASE):
            continue

        c = {}
        c['account_name'] = find_account_name(block) or find_account_name(text)

        # Cargo name
        for pat in [
            r'(?:WITH|WITH\s+)\s*(GRAIN|COAL|IRON\s+ORE|STEEL|CLINKER|FERTILIZER|UREA|GRAINS|GENS|STEELS|LAWFULS)',
            r'TCT\s+WITH\s+([A-Z][A-Z/\s]+?)(?:\n|\.|REDEL|REDELV)',
            r'CARGO\s*:?\s*([A-Z][A-Z\s]+?)(?:\n|DELY|DELIVERY)',
        ]:
            m = re.search(pat, block, re.IGNORECASE)
            if m:
                c['cargo_name'] = clean(m.group(1)).title()
                break

        # Delivery port
        for pat in [
            r'DELIV(?:ERY)?\s*(?:PORT)?\s*:?\s*([A-Z][A-Z\s,]+?)(?:\n|LC|LAYCAN|REDEL|REDELV|DURATION)',
            r'DELY\s+(?:TO\s+)?([A-Z][A-Z\s,\(\)]+?)(?:\n|LC|LAYCAN|REDEL|DURATION)',
            r'DELIVERY\s*:?\s*([A-Z][A-Z\s,]+?)(?:\n|LC|LAYCAN|REDEL)',
        ]:
            m = re.search(pat, block, re.IGNORECASE)
            if m:
                c['delivery_port'] = clean(m.group(1)).title()
                break

        # Redelivery port
        for pat in [
            r'REDELIV(?:ERY)?\s*(?:PORT)?\s*:?\s*([A-Z][A-Z\s,]+?)(?:\n|DURATION|COMM|3\.)',
            r'REDEL\s*:?\s*([A-Z][A-Z\s,]+?)(?:\n|DURATION|COMM|3\.)',
            r'REDELIVERY\s*:?\s*([A-Z][A-Z\s,]+?)(?:\n|DURATION|COMM)',
        ]:
            m = re.search(pat, block, re.IGNORECASE)
            if m:
                c['redelivery_port'] = clean(m.group(1)).title()
                break

        # Duration
        for pat in [
            r'DURATION\s*:?\s*([\w\s\-/]+?(?:DAYS|MONTHS|YEARS|YRS)\s*(?:WOG)?)',
            r'(\d+[-–]\d+\s*(?:DAYS|MONTHS|YEARS|YRS)\s*(?:WOG)?)',
            r'(ABT\s+\d+\s*(?:DAYS|MONTHS|YEARS|YRS)\s*(?:WOG)?)',
            r'(\d+[-–]\d+\s*YEARS)',
        ]:
            m = re.search(pat, block, re.IGNORECASE)
            if m:
                c['duration'] = clean(m.group(1))
                break

        # Laycan / LC
        for pat in [
            r'(?:LC|LAYCAN)\s*:?\s*(\d{1,2}[-–]\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*\s*\d{0,4})',
            r'(?:LC|LAYCAN)\s*:?\s*(\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*\s*[-–]\s*\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*\s*\d{0,4})',
            r'(?:LC)\s+(\d{1,2}[-–]\d{1,2}\s+\w+)',
            r'LAYCAN\s*:?\s*(\d{1,2}[-–]\d{1,2}\s+\w+\s*\d{0,4})',
        ]:
            m = re.search(pat, block, re.IGNORECASE)
            if m:
                c['laycan'] = clean(m.group(1))
                break

        # Cargo type
        cargo_types = ['GRAIN', 'GRAINS', 'CLINKER', 'COAL', 'STEEL', 'FERTILIZER',
                       'IRON ORE', 'GENS', 'GENERAL CARGO', 'LAWFULS']
        for ct in cargo_types:
            if re.search(r'\b' + ct + r'\b', block, re.IGNORECASE):
                c['cargo_type'] = ct.title()
                break
        if 'cargo_type' not in c:
            c['cargo_type'] = 'Dry Bulk'

        # Vessel size preference
        m = re.search(r'(SMX|UMX|SUPRA|ULTRA|HANDYMAX|PANAMAX|KAMSARMAX|HANDYSIZE)\s*[-–]?\s*(UMX|SMX|SUPRA|ULTRA)?', block, re.IGNORECASE)
        if m:
            c['vessel_size_pref'] = m.group(0).strip().upper()

        # Commission
        m = re.search(r'(\d+\.?\d*)\s*(?:PCT|%)\s*(?:ADDCOM|ADC|TTL|TOTAL)', block, re.IGNORECASE)
        if m:
            c['commission_pct'] = m.group(1) + '%'

        if c.get('delivery_port') or c.get('cargo_name') or c.get('duration') or c.get('laycan'):
            cargoes.append(c)

    return cargoes if cargoes else [{'account_name': find_account_name(text), 'raw_note': 'TC cargo details unclear'}]


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def parse_email(text: str, sender: str = '', subject: str = '') -> dict:
    """
    Main parse function.
    Returns dict with: category, records[], confidence, timestamp
    """
    combined = f"{subject}\n{text}"
    category = classify_email(combined)

    # Score for confidence display
    t = score_category(combined, TONNAGE_KEYWORDS)
    vc = score_category(combined, CARGO_VC_KEYWORDS)
    tc = score_category(combined, CARGO_TC_KEYWORDS)
    total = max(t + vc + tc, 1)

    scores = {'tonnage': t, 'cargo_vc': vc, 'cargo_tc': tc}
    top = scores[category]
    confidence = min(round((top / max(sum(scores.values()), 1)) * 100 + 20), 98)

    if category == 'tonnage':
        records = extract_tonnage(text)
    elif category == 'cargo_vc':
        records = extract_cargo_vc(text)
    else:
        records = extract_cargo_tc(text)

    return {
        'category': category,
        'category_label': {
            'tonnage': 'Tonnage (Open Vessels)',
            'cargo_vc': 'Cargo VC (Voyage Charter)',
            'cargo_tc': 'Cargo TC (Time Charter)',
        }[category],
        'records': records,
        'record_count': len(records),
        'confidence': confidence,
        'scores': scores,
        'sender': sender,
        'subject': subject,
        'timestamp': datetime.now().isoformat(),
        'snippet': text[:200].replace('\n', ' '),
    }
