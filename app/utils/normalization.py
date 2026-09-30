"""Question normalization, slang expansion, strict numeric parsing, and date parsing."""

import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

CONTRACTIONS: Dict[str, str] = {
    r"\bwho's\b": "who is",
    r"\bwhoz\b": "who is",
    r"\bgettin\b": "getting",
    r"\bwhat's\b": "what is",
    r"\bwhere's\b": "where is",
    r"\bthere's\b": "there is",
    r"\bhow's\b": "how is",
    r"\bwho r\b": "who are",
    r"\bu r\b": "you are",
    r"\bu\b": "you",
    r"\br\b": "are",
    r"\bppl\b": "people",
    r"\bguys\b": "people",
    r"\bwhich dude\b": "which person",
    r"\bwhich guy\b": "which person",
    r"\bthat guy\b": "that person",
    r"\bthat dude\b": "that person",
    r"\bwhat team\b": "what department",
    r"\bgimme\b": "give me",
    r"\blemme\b": "let me",
    r"\bwanna\b": "want to",
    r"\bu got\b": "do you have",
    r"\bdups\b": "duplicates",
    r"\bdupes\b": "duplicates",
}

SLANG_PHRASES: List[Tuple[str, str]] = [
    (r"\bwho makes the most\b", "who has highest salary"),
    (r"\bwho makes the least\b", "who has lowest salary"),
    (r"\bwho gets paid the most\b", "who has highest salary"),
    (r"\bwho gettin paid most\b", "who has highest salary"),
    (r"\bwho gets paid most\b", "who has highest salary"),
    (r"\bwho gets paid the least\b", "who has lowest salary"),
    (r"\bwho earns the most\b", "who has highest salary"),
    (r"\bwho earns most\b", "who has highest salary"),
    (r"\bwho earns the least\b", "who has lowest salary"),
    (r"\bmaking bank\b", "highest salary"),
    (r"\bwho joined first\b", "who joined earliest"),
    (r"\bwho joined last\b", "who joined latest"),
    (r"\btop earner\b", "highest salary"),
    (r"\blowest earner\b", "lowest salary"),
    (r"\bcostliest\b", "highest price"),
    (r"\bcheapest\b", "lowest price"),
    (r"\bbiggest bucks\b", "highest amount"),
    (r"\bmost dough\b", "highest salary"),
]


def parse_numeric_value(raw_val: str) -> Optional[float]:
    """Parse numeric strings including k, l, lakh, lac, cr, crore, million, and currency symbols."""
    clean = re.sub(r"[₹$,]", "", raw_val).strip().lower()
    if not clean:
        return None

    try:
        # Check crores (cr, crore, crores)
        if "crore" in clean or clean.endswith("cr") or " cr" in clean:
            num = float(re.sub(r"[^\d.]", "", clean))
            return num * 10_000_000

        # Check millions (m, mn, million)
        if "million" in clean or "mn" in clean or (clean.endswith("m") and not clean.endswith("km")):
            num = float(re.sub(r"[^\d.]", "", clean))
            return num * 1_000_000

        # Check lakhs (l, lac, lacs, lakh, lakhs)
        if "lakh" in clean or "lac" in clean or clean.endswith("l") or " l" in clean:
            num = float(re.sub(r"[^\d.]", "", clean))
            return num * 100_000

        # Check thousands (k or thousand)
        if clean.endswith("k") or "thousand" in clean:
            num = float(re.sub(r"[^\d.]", "", clean))
            return num * 1_000

        return float(re.sub(r"[^\d.]", "", clean))
    except Exception:
        return None


NUM_UNIT_PAT = r"(?:k|thousand|l|lac|lacs|lakh|lakhs|cr|crore|crores|m|mn|million)?"


def extract_numeric_filter(text: str) -> Optional[Dict[str, Any]]:
    """Extract strict numeric conditions recognizing >= vs > and <= vs < with typo resilience."""
    t = text.lower()
    # Normalize common typo "more then" -> "more than", "less then" -> "less than"
    t = re.sub(r"\bmore then\b", "more than", t)
    t = re.sub(r"\bless then\b", "less than", t)

    # Range: between X and Y
    between_match = re.search(
        rf"\bbetween\s+(?:₹|rs\.?|\$)?\s*([\d.]+\s*{NUM_UNIT_PAT})\s+and\s+(?:₹|rs\.?|\$)?\s*([\d.]+\s*{NUM_UNIT_PAT})\b",
        t
    )
    if between_match:
        val1 = parse_numeric_value(between_match.group(1))
        val2 = parse_numeric_value(between_match.group(2))
        if val1 is not None and val2 is not None:
            return {
                "type": "range",
                "min_operator": ">=",
                "min_value": min(val1, val2),
                "max_operator": "<=",
                "max_value": max(val1, val2)
            }

    # At least / Minimum / >=
    gte_match = re.search(
        rf"(?:\b(?:at least|minimum of|not less than)|>=)\s*(?:₹|rs\.?|\$)?\s*([\d.]+\s*{NUM_UNIT_PAT})\b",
        t
    )
    if gte_match:
        val = parse_numeric_value(gte_match.group(1))
        if val is not None:
            return {"operator": ">=", "value": val}

    # At most / Maximum / <=
    lte_match = re.search(
        rf"(?:\b(?:at most|maximum of|not more than|not exceeding)|<=)\s*(?:₹|rs\.?|\$)?\s*([\d.]+\s*{NUM_UNIT_PAT})\b",
        t
    )
    if lte_match:
        val = parse_numeric_value(lte_match.group(1))
        if val is not None:
            return {"operator": "<=", "value": val}

    # Strictly greater than / More than / Over / Above / >
    gt_match = re.search(
        rf"(?:\b(?:more than|greater than|above|over|higher than|paid over|earns over|priced over|getting more than|making more than)|>)\s*(?:₹|rs\.?|\$)?\s*([\d.]+\s*{NUM_UNIT_PAT})\b",
        t
    )
    if gt_match:
        val = parse_numeric_value(gt_match.group(1))
        if val is not None:
            return {"operator": ">", "value": val}

    # Strictly less than / Below / Under / Cheaper than / <
    lt_match = re.search(
        rf"(?:\b(?:less than|below|under|cheaper than|lower than)|<)\s*(?:₹|rs\.?|\$)?\s*([\d.]+\s*{NUM_UNIT_PAT})\b",
        t
    )
    if lt_match:
        val = parse_numeric_value(lt_match.group(1))
        if val is not None:
            return {"operator": "<", "value": val}

    # Exactly equals
    eq_match = re.search(
        rf"(?:\b(?:exactly|equal to)|=)\s*(?:₹|rs\.?|\$)?\s*([\d.]+\s*{NUM_UNIT_PAT})\b",
        t
    )
    if eq_match:
        val = parse_numeric_value(eq_match.group(1))
        if val is not None:
            return {"operator": "=", "value": val}

    return None


def extract_date_filter(text: str) -> Optional[Dict[str, Any]]:
    """Extract natural language date bounds (this year, last year, before 2024, etc.)."""
    t = text.lower()
    current_year = date.today().year

    if "this year" in t:
        return {"operator": ">=", "value": f"{current_year}-01-01", "explanation": f"Current year ({current_year})"}
    if "last year" in t:
        return {
            "type": "range",
            "min_operator": ">=",
            "min_value": f"{current_year - 1}-01-01",
            "max_operator": "<=",
            "max_value": f"{current_year - 1}-12-31",
            "explanation": f"Previous year ({current_year - 1})"
        }

    # Before YYYY
    before_m = re.search(r"\bbefore\s+(\d{4})\b", t)
    if before_m:
        yr = before_m.group(1)
        return {"operator": "<", "value": f"{yr}-01-01", "explanation": f"Before year {yr}"}

    # After YYYY
    after_m = re.search(r"\bafter\s+(\d{4})\b", t)
    if after_m:
        yr = after_m.group(1)
        return {"operator": ">", "value": f"{yr}-12-31", "explanation": f"After year {yr}"}

    # Since / From YYYY
    since_m = re.search(r"\b(?:since|from)\s+(\d{4})\b", t)
    if since_m:
        yr = since_m.group(1)
        return {"operator": ">=", "value": f"{yr}-01-01", "explanation": f"Since year {yr}"}

    # In YYYY
    in_m = re.search(r"\bin\s+(\d{4})\b", t)
    if in_m:
        yr = in_m.group(1)
        return {
            "type": "range",
            "min_operator": ">=",
            "min_value": f"{yr}-01-01",
            "max_operator": "<=",
            "max_value": f"{yr}-12-31",
            "explanation": f"During year {yr}"
        }

    return None


def normalize_question(question: str) -> str:
    """Standardize user question into clean, consistent representation."""
    if not question:
        return ""

    q = question.strip()

    # 1. Multilingual and code-switching normalization
    try:
        from app.utils.multilingual import multilingual_normalizer
        q = multilingual_normalizer.normalize(q)
    except Exception:
        pass

    # 2. Expand contractions
    for pat, rep in CONTRACTIONS.items():
        q = re.sub(pat, rep, q, flags=re.IGNORECASE)

    # 3. Expand slang phrases
    for pat, rep in SLANG_PHRASES:
        q = re.sub(pat, rep, q, flags=re.IGNORECASE)

    # 4. Clean up multiple spaces and trailing punctuation
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[?!.,;]+$", "", q).strip()

    return q


def universal_text_normalizer(text: str) -> Dict[str, Any]:
    """Produce all required normalized representations of input text per Section 8."""
    import unicodedata

    if not text:
        return {
            "raw": "",
            "trimmed": "",
            "lowercase": "",
            "casefolded": "",
            "unicode_normalized": "",
            "whitespace_normalized": "",
            "punctuation_normalized": "",
            "alphanumeric_normalized": "",
            "compact_form": "",
            "tokens": [],
        }

    raw = text
    trimmed = text.strip()
    lowercase = trimmed.lower()
    casefolded = trimmed.casefold()
    unicode_norm = unicodedata.normalize("NFKD", trimmed)
    ws_norm = re.sub(r"\s+", " ", unicode_norm).strip()
    punct_norm = re.sub(r"[^\w\s]", " ", ws_norm)
    punct_norm = re.sub(r"\s+", " ", punct_norm).strip()
    alphanumeric_norm = re.sub(r"[^a-zA-Z0-9\s]", "", ws_norm)
    alphanumeric_norm = re.sub(r"\s+", " ", alphanumeric_norm).strip()
    compact_form = re.sub(r"\s+", "", lowercase)
    tokens = [t for t in alphanumeric_norm.lower().split() if t]

    return {
        "raw": raw,
        "trimmed": trimmed,
        "lowercase": lowercase,
        "casefolded": casefolded,
        "unicode_normalized": unicode_norm,
        "whitespace_normalized": ws_norm,
        "punctuation_normalized": punct_norm,
        "alphanumeric_normalized": alphanumeric_norm,
        "compact_form": compact_form,
        "tokens": tokens,
    }


def parse_date_string(date_str: str) -> Optional[date]:
    """Parse date from arbitrary format including ISO, US, EU, textual, and relative."""
    if not date_str or not str(date_str).strip():
        return None
    s = str(date_str).strip()
    try:
        import dateparser
        dt = dateparser.parse(s)
        if dt:
            return dt.date()
    except Exception:
        pass

    try:
        from datetime import datetime
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                continue
    except Exception:
        pass
    return None


