"""Fuzzy matching, abbreviation expansion, and informal speech normalization."""

import re
import difflib
from typing import Dict, List, Optional, Tuple

# Comprehensive abbreviation mapping
LOCATION_ABBREVIATIONS: Dict[str, str] = {
    "hyd": "Hyderabad",
    "hydrabad": "Hyderabad",
    "hyderbad": "Hyderabad",
    "hyderabad": "Hyderabad",
    "blr": "Bengaluru",
    "bangalore": "Bengaluru",
    # South & West
    "bangalore": "Bengaluru",
    "banglore": "Bengaluru",
    "banglroe": "Bengaluru",
    "bangluru": "Bengaluru",
    "bengalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "blr": "Bengaluru",
    "bng": "Bengaluru",
    "mum": "Mumbai",
    "mumbaii": "Mumbai",
    "bombay": "Mumbai",
    "bom": "Mumbai",
    "del": "Delhi",
    "dilli": "Delhi",
    "ndls": "Delhi",
    "new delhi": "Delhi",
    "kol": "Kolkata",
    "cal": "Kolkata",
    "calcutta": "Kolkata",
    "ccu": "Kolkata",
    "kolkatta": "Kolkata",
    "kalkatta": "Kolkata",
    "che": "Chennai",
    "maa": "Chennai",
    "madras": "Chennai",
    "pune": "Pune",
    "puna": "Pune",
    "poona": "Pune",
    "pnq": "Pune",
    "hyd": "Hyderabad",
    "cyberabad": "Hyderabad",
    "secunderabad": "Secunderabad",
    "amd": "Ahmedabad",
    "jai": "Jaipur",
    "lko": "Lucknow",
    "bbsr": "Bhubaneswar",
    "bbi": "Bhubaneswar",
    "cok": "Kochi",
    "cochin": "Kochi",
    "trv": "Thiruvananthapuram",
    "trivandrum": "Thiruvananthapuram",
    "goi": "Goa",
    "pat": "Patna",
    "rpr": "Raipur",
    "rnc": "Ranchi",
    "gau": "Guwahati",
    "gauhati": "Guwahati",
    "ixc": "Chandigarh",
    "idr": "Indore",
    "bpl": "Bhopal",
    "ddn": "Dehradun",
    "gnr": "Gandhinagar",
    "shl": "Shillong",
    "imf": "Imphal",
    "ixa": "Agartala",
    # All 28 States & Shortcuts
    "ap": "Andhra Pradesh",
    "a.p.": "Andhra Pradesh",
    "a.p": "Andhra Pradesh",
    "andra": "Andhra Pradesh",
    "andra pradesh": "Andhra Pradesh",
    "ar": "Arunachal Pradesh",
    "a.r.": "Arunachal Pradesh",
    "br": "Bihar",
    "cg": "Chhattisgarh",
    "ct": "Chhattisgarh",
    "ga": "Goa",
    "gj": "Gujarat",
    "guj": "Gujarat",
    "gujrat": "Gujarat",
    "hr": "Haryana",
    "hp": "Himachal Pradesh",
    "h.p.": "Himachal Pradesh",
    "jh": "Jharkhand",
    "ka": "Karnataka",
    "kar": "Karnataka",
    "karnatka": "Karnataka",
    "kl": "Kerala",
    "ker": "Kerala",
    "mp": "Madhya Pradesh",
    "m.p.": "Madhya Pradesh",
    "m.p": "Madhya Pradesh",
    "mh": "Maharashtra",
    "maha": "Maharashtra",
    "m.h.": "Maharashtra",
    "mn": "Manipur",
    "ml": "Meghalaya",
    "meg": "Meghalaya",
    "mz": "Mizoram",
    "miz": "Mizoram",
    "nl": "Nagaland",
    "nag": "Nagaland",
    "od": "Odisha",
    "orissa": "Odisha",
    "pb": "Punjab",
    "pun": "Punjab",
    "rj": "Rajasthan",
    "raj": "Rajasthan",
    "sk": "Sikkim",
    "sik": "Sikkim",
    "tn": "Tamil Nadu",
    "t.n.": "Tamil Nadu",
    "ts": "Telangana",
    "t.s.": "Telangana",
    "t.s": "Telangana",
    "tg": "Telangana",
    "tel": "Telangana",
    "telengana": "Telangana",
    "tr": "Tripura",
    "tri": "Tripura",
    "up": "Uttar Pradesh",
    "u.p.": "Uttar Pradesh",
    "uk": "Uttarakhand",
    "ut": "Uttarakhand",
    "ua": "Uttarakhand",
    "wb": "West Bengal",
    "w.b.": "West Bengal",
    # Union Territories
    "an": "Andaman and Nicobar Islands",
    "ch": "Chandigarh",
    "chd": "Chandigarh",
    "dh": "Dadra and Nagar Haveli and Daman and Diu",
    "dd": "Dadra and Nagar Haveli and Daman and Diu",
    "dnh": "Dadra and Nagar Haveli and Daman and Diu",
    "dl": "Delhi",
    "nct": "Delhi",
    "jk": "Jammu and Kashmir",
    "jnk": "Jammu and Kashmir",
    "la": "Ladakh",
    "ld": "Lakshadweep",
    "py": "Puducherry",
    "pondy": "Puducherry",
    "pondicherry": "Puducherry",
    "chandigarh": "Chandigarh",
}

# Domain & Business term shortcuts
BUSINESS_ABBREVIATIONS: Dict[str, str] = {
    "sal": "salary",
    "amt": "amount",
    "qty": "quantity",
    "stock": "stock quantity",
    "dept": "department",
    "desig": "designation",
    "emp": "employee",
    "cust": "customer",
    "veh": "vehicle",
    "svc": "service",
    "prod": "product",
    "cat": "category",
    "mgr": "manager",
    "dev": "developer",
    "eng": "engineering",
    "hr": "Human Resources",
    "perf": "performance score",
    "rat": "rating",
    "stat": "status",
    "exp": "experience",
    "warr": "warranty",
    "avrg": "average",
    "totl": "total",
    "higest": "highest",
    "hihgest": "highest",
    "lowst": "lowest",
    "erans": "earns",
    "scnd": "second",
    "thrd": "third",
    "slary": "salary",
    "salry": "salary",
    "dpartmnt": "department",
    "depatment": "department",
    "preformer": "performer",
    "designtion": "designation",
    "whre": "where",
    "mximum": "maximum",
}

# Common informal / slang phrases
SLANG_PATTERNS: List[Tuple[str, str]] = [
    (r"\bgimme\b", "give me"),
    (r"\blemme\b", "let me"),
    (r"\bwanna\b", "want to"),
    (r"\bu got\b", "do you have"),
    (r"\bwho makes the most\b", "highest salary"),
    (r"\bwho makes the least\b", "lowest salary"),
    (r"\bwho gets paid least\b", "lowest salary"),
    (r"\bbiggest paycheck\b", "highest salary"),
    (r"\brockstar\b", "best performer"),
    (r"\bcostliest\b", "highest price"),
    (r"\bcheapest\b", "lowest price"),
    (r"\bbiggest bucks\b", "highest amount"),
    (r"\bmost dough\b", "highest salary"),
    (r"\bdups\b", "duplicates"),
    (r"\bany dupes\b", "duplicates"),
    (r"\brajdhani\b", "capital"),
    (r"\bwhats\b", "what is"),
]


def expand_abbreviations(text: str) -> str:
    """Expand known abbreviations and normalize slang in user input while protecting IDs."""
    normalized = text.lower()

    # Expand slang patterns
    for pat, rep in SLANG_PATTERNS:
        normalized = re.sub(pat, rep, normalized, flags=re.IGNORECASE)

    # Clean up conversational fillers at boundaries
    normalized = re.sub(r"\s+(?:bro|dude|please|pls|now|fast|yo|bhai|yaar|zara)\s*\??$", "?", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^(?:yo|hey|gimme|give\s+me)\s+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^(?:tell\s+me|tell|whats)\s+([a-zA-Z\s]+?)\s+(?:capital|rajdhani)\s*\??$", r"capital of \1?", normalized, flags=re.IGNORECASE)

    # 1. Hinglish possessive particles: "X ka/ki/ke Y" -> "Y of X"
    # Handles multi-word states e.g. "Andhra Pradesh ka capital", "West Bengal ki rajdhani"
    normalized = re.sub(
        r"^(?:mujhe\s+)?([a-zA-Z\s]+?)\s+(?:ka|ki|ke)\s+(?:capital|rajdhani)(?:.*)$",
        r"capital of \1",
        normalized,
        flags=re.IGNORECASE
    )
    normalized = re.sub(
        r"\b([a-zA-Z0-9\s]+?)\s+(?:ka|ki|ke)\s+(capital|rajdhani)\b",
        r"\2 of \1",
        normalized,
        flags=re.IGNORECASE
    )

    # 2. Informal location phrasing normalizations
    # e.g. "hyd which state comes?", "blr which state?" -> "which state has hyd?"
    normalized = re.sub(
        r"\b([a-zA-Z0-9]+)\s+(?:which|whch)\s+([a-zA-Z]+)(?:\s+(?:comes|is|belongs))?\s*\??$",
        r"which \2 has \1?",
        normalized,
        flags=re.IGNORECASE
    )

    # e.g. "arunachal pradesh capital?", "bihar capital?" -> "capital of arunachal pradesh"
    if not any(w in normalized for w in ["which", "what", "who", "how", "has", "have", "where"]):
        m = re.match(r"^([a-zA-Z\s]{2,30}?)\s+(?:capital|rajdhani)\s*\??$", normalized)
        if m and m.group(1).strip() not in ["the", "a", "an"]:
            normalized = f"capital of {m.group(1).strip()}?"

    # e.g. "capital ts?", "capital telangana?" -> "capital of ts?"
    if not any(w in normalized for w in ["which", "how", "has", "have", "where"]):
        normalized = re.sub(
            r"^(?:what\s+is\s+|tell\s+me\s+)?capital\s+(?:of\s+)?([a-zA-Z0-9\s]+?)\s*\??$",
            r"capital of \1?",
            normalized,
            flags=re.IGNORECASE
        )

    # Disambiguation: "HR" in location/capital contexts means Haryana, not Human Resources
    if re.search(r"\b(?:capital|state|city|district|region)\b", normalized, re.IGNORECASE):
        normalized = re.sub(r"\bhr\b", "Haryana", normalized, flags=re.IGNORECASE)

    # English word collision protection for state codes AS (Assam) and OR (Odisha):
    # Only expand when in state/capital pattern e.g. "capital of as", "state of as", or starting with "as capital"
    normalized = re.sub(r"\b(capital\s+of|state\s+of)\s+as\b", r"\1 Assam", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^(?:what\s+is\s+|tell\s+me\s+)?as\s+(capital|state)\b", r"Assam \1", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(capital\s+of|state\s+of)\s+or\b", r"\1 Odisha", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^(?:what\s+is\s+|tell\s+me\s+)?or\s+(capital|state)\b", r"Odisha \1", normalized, flags=re.IGNORECASE)
    # AI / ML collision protection for state code ML (Meghalaya) vs Machine Learning
    normalized = re.sub(r"\b(ai\s*(?:and|vs|versus|/)\s*)ml\b", r"\1machine learning", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bml\s*(?:and|vs|versus|/)\s*ai\b", r"machine learning and ai", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bml\s+(model|algorithm|engineer|system)\b", r"machine learning \1", normalized, flags=re.IGNORECASE)

    # Business term replacement with negative lookahead protecting IDs like EMP-1001 or CUST-101
    for abbr, full in BUSINESS_ABBREVIATIONS.items():
        pat = r"\b" + re.escape(abbr) + r"\b(?!\s*[-_]?\d)"
        normalized = re.sub(pat, full, normalized, flags=re.IGNORECASE)

    # Location replacement
    for abbr, full in LOCATION_ABBREVIATIONS.items():
        pat = r"\b" + re.escape(abbr) + r"\b"
        normalized = re.sub(pat, full, normalized, flags=re.IGNORECASE)

    return normalized


def fuzzy_match_column(candidate: str, available_columns: List[str], threshold: float = 0.60) -> Optional[str]:
    """Match a token or phrase to the closest column name using abbreviations, substring, and difflib."""
    if not candidate or not available_columns:
        return None

    cand_clean = candidate.strip().lower()
    cand_expanded = expand_abbreviations(cand_clean).lower()

    # 1. Exact or substring match with candidate or expanded form
    for col in available_columns:
        col_lower = col.lower()
        if cand_clean == col_lower or cand_expanded == col_lower:
            return col
        if cand_clean in col_lower or col_lower in cand_clean:
            return col
        if cand_expanded in col_lower or col_lower in cand_expanded:
            return col

    # 2. Check difflib close matches on normalized names
    cols_map = {c.lower(): c for c in available_columns}
    matches = difflib.get_close_matches(cand_clean, list(cols_map.keys()), n=1, cutoff=threshold)
    if matches:
        return cols_map[matches[0]]

    # 3. Try matching against expanded candidate
    if cand_expanded != cand_clean:
        matches = difflib.get_close_matches(cand_expanded, list(cols_map.keys()), n=1, cutoff=threshold)
        if matches:
            return cols_map[matches[0]]

    return None


def fuzzy_match_value(candidate: str, possible_values: List[str], threshold: float = 0.70) -> Optional[str]:
    """Match a misspelled value against known distinct column values."""
    if not candidate or not possible_values:
        return None

    cand_clean = candidate.strip().lower()

    # Check location abbreviations first
    if cand_clean in LOCATION_ABBREVIATIONS:
        expanded = LOCATION_ABBREVIATIONS[cand_clean].lower()
        for val in possible_values:
            if str(val).lower() == expanded:
                return str(val)

    # Exact case-insensitive match
    val_map = {str(v).lower(): str(v) for v in possible_values}
    if cand_clean in val_map:
        return val_map[cand_clean]

    # Substring match
    for k, v in val_map.items():
        if cand_clean in k or k in cand_clean:
            return v

    # Difflib close matches
    matches = difflib.get_close_matches(cand_clean, list(val_map.keys()), n=1, cutoff=threshold)
    if matches:
        return val_map[matches[0]]

    # Phonetic sound-alike matching
    from app.utils.phonetic import phonetic_match_score
    best_p_val = None
    best_p_score = 0.0
    for k, v in val_map.items():
        ps = phonetic_match_score(cand_clean, k)
        if ps > best_p_score and ps >= threshold:
            best_p_score = ps
            best_p_val = v
    if best_p_val:
        return best_p_val

    return None


def extract_best_match_from_text(text: str, candidates: List[str], threshold: float = 0.80) -> Optional[str]:
    """Scan words and n-grams in text to find the best match against a candidate list."""
    if not text or not candidates:
        return None

    cand_map = {str(c).lower(): str(c) for c in candidates}
    words = re.findall(r"\b\w+\b", text)

    STOPWORDS = {
        "top", "bottom", "most", "least", "best", "worst", "all", "any", "the", "and",
        "for", "with", "from", "show", "give", "find", "get", "what", "which", "how",
        "many", "who", "are", "is", "was", "were", "of", "in", "on", "at", "to", "a",
        "an", "by", "per", "records", "data", "products", "employees", "customers",
        "items", "details", "list", "each", "every", "more", "less", "than",
        "making", "make", "makes", "over", "dude", "guy", "money", "cash", "paid", "pay", "earning", "earns",
        "having", "bro", "now", "fast", "yo", "hey", "please", "pls", "tell", "me", "whats", "zara", "bhai", "yaar"
    }

    # 1. Direct word check with abbreviation expansion
    for w in words:
        wl = w.lower()
        if wl in STOPWORDS:
            continue
        if wl in LOCATION_ABBREVIATIONS:
            exp = LOCATION_ABBREVIATIONS[wl].lower()
            if exp in cand_map:
                return cand_map[exp]
        if wl in cand_map:
            return cand_map[wl]

    # 2. 2-gram check
    for i in range(len(words) - 1):
        bigram = f"{words[i].lower()} {words[i+1].lower()}"
        if bigram in cand_map:
            return cand_map[bigram]

    # 3. 3-gram check
    for i in range(len(words) - 2):
        trigram = f"{words[i].lower()} {words[i+1].lower()} {words[i+2].lower()}"
        if trigram in cand_map:
            return cand_map[trigram]

    # 4. Fuzzy match single words (only against single-word candidates to prevent partial collision)
    single_word_cands = [k for k in cand_map.keys() if " " not in k]
    for w in words:
        wl = w.lower()
        if wl in STOPWORDS or len(wl) < 4:
            continue
        matches = difflib.get_close_matches(wl, single_word_cands, n=1, cutoff=threshold)
        if matches:
            return cand_map[matches[0]]

    # 5. Phonetic sound-alike scan
    from app.utils.phonetic import phonetic_match_score
    for w in words:
        wl = w.lower()
        if wl in STOPWORDS or len(wl) < 4:
            continue
        for cand_k in single_word_cands:
            ps = phonetic_match_score(wl, cand_k)
            if ps >= 0.82:
                return cand_map[cand_k]

    return None

