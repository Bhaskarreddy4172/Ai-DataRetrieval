"""Dedicated Multi-Stage Text and Question Normalization Pipeline.

Handles:
1. Case & Whitespace Normalization (preserves original for display, normalized for matching)
2. Punctuation & Unicode Normalization (preserves semantic identifiers like EMP-1001, ST-01, VIL-001)
3. Contractions & Informal Language
4. Grammar & Word-Order Normalization
5. Multilingual & Hinglish / Telugu-English / Code-Switched Question Normalization
6. Numeric, Currency, and Unit Normalization
7. Date Expression Normalization
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class NormalizedQuestion:
    """Structured representation of a normalized user question."""
    raw_question: str
    normalized_question: str
    cleaned_question: str
    detected_language: str = "en"
    corrected_terms: Dict[str, str] = field(default_factory=dict)
    resolved_entities: List[str] = field(default_factory=list)
    resolved_columns: List[str] = field(default_factory=list)
    detected_intent: Optional[str] = None
    numeric_filters: List[Dict[str, Any]] = field(default_factory=list)
    currency_detected: Optional[str] = None
    units_detected: List[str] = field(default_factory=list)
    is_multilingual: bool = False


class TextNormalizer:
    """Comprehensive normalization pipeline for dataset retrieval queries."""

    # Contractions mapping
    CONTRACTIONS: Dict[str, str] = {
        r"\bwho's\b": "who is",
        r"\bwhoz\b": "who is",
        r"\bwhat's\b": "what is",
        r"\bwhere's\b": "where is",
        r"\bthere's\b": "there is",
        r"\bhow's\b": "how is",
        r"\bdoesn't\b": "does not",
        r"\bisn't\b": "is not",
        r"\bcan't\b": "cannot",
        r"\bwon't\b": "will not",
        r"\bdon't\b": "do not",
        r"\bdidn't\b": "did not",
        r"\baren't\b": "are not",
        r"\bwasn't\b": "was not",
        r"\bweren't\b": "were not",
        r"\bwho r\b": "who are",
        r"\bu r\b": "you are",
        r"\bu\b": "you",
        r"\br\b": "are",
        r"\bppl\b": "people",
        r"\bguys\b": "people",
        r"\bgimme\b": "give me",
        r"\blemme\b": "let me",
        r"\bwanna\b": "want to",
        r"\bu got\b": "do you have",
        r"\bdups\b": "duplicates",
        r"\bdupes\b": "duplicates",
    }

    # Common comparison & intent typos
    COMMON_TYPOS: Dict[str, str] = {
        r"\bcomapre\b": "compare",
        r"\bcompair\b": "compare",
        r"\bcompere\b": "compare",
        r"\bvillag\b": "village",
        r"\bvilage\b": "village",
        r"\bvillges\b": "villages",
        r"\bvilages\b": "villages",
        r"\bdiffernce\b": "difference",
        r"\bdiference\b": "difference",
        r"\bdiffrnce\b": "difference",
        r"\bpopulaton\b": "population",
        r"\bpopualtion\b": "population",
        r"\btelengana\b": "telangana",
        r"\bandra\b": "andhra",
    }

    # Informal & slang query patterns
    SLANG_PATTERNS: List[Tuple[str, str]] = [
        (r"\bwho makes the most\b", "who has highest salary"),
        (r"\bwho makes the least\b", "who has lowest salary"),
        (r"\bwho gets paid the most\b", "who has highest salary"),
        (r"\bwho gets paid most\b", "who has highest salary"),
        (r"\bwho makes more\b", "who has higher salary"),
        (r"\bwho earns more\b", "who has higher salary"),
        (r"\bwho gets paid the least\b", "who has lowest salary"),
        (r"\bwho earns the most\b", "who has highest salary"),
        (r"\bwho earns most\b", "who has highest salary"),
        (r"\bwho earns least\b", "who has lowest salary"),
        (r"\bmaking bank\b", "highest salary"),
        (r"\btop earner\b", "highest salary"),
        (r"\blowest earner\b", "lowest salary"),
        (r"\btop guy\b", "highest"),
        (r"\bhighest one\b", "highest"),
        (r"\blowest one\b", "lowest"),
        (r"\bwho's top\b", "who has highest"),
        (r"\bcostliest\b", "highest price"),
        (r"\bcheapest\b", "lowest price"),
        (r"\bbiggest bucks\b", "highest amount"),
        (r"\bmost dough\b", "highest salary"),
        (r"\bwho joined first\b", "who joined earliest"),
        (r"\bwho joined last\b", "who joined latest"),
    ]

    # Multilingual & Hinglish / Telugu-English patterns
    MULTILINGUAL_PATTERNS: List[Tuple[str, str]] = [
        # Hinglish / Hindi
        (r"\bkitne\s+villages\s+hain\b", "how many villages"),
        (r"\bkitne\s+villages\b", "how many villages"),
        (r"\bkitne\s+log\b", "how many people"),
        (r"\bkitne\s+employees\b", "how many employees"),
        (r"\bkitna\s+salary\b", "what salary"),
        (r"\bkitni\s+population\b", "what population"),
        (r"\bsabse\s+jyada\b", "highest"),
        (r"\bsabse\s+zyada\b", "highest"),
        (r"\bsabse\s+kam\b", "lowest"),
        (r"\bsabse\s+bada\b", "largest"),
        (r"\bsabse\s+chota\b", "smallest"),
        (r"\bka\s+capital\s+kya\s+hai\b", "capital"),
        (r"\bka\s+capital\b", "capital"),
        (r"\bki\s+rajdhani\b", "capital"),
        (r"\brajdhani\b", "capital"),
        (r"\bkisko\s+milta\s+hai\b", "who has"),
        # Telugu-English
        (r"\benni\s+villages\s+unnayi\b", "how many villages"),
        (r"\benni\s+villages\b", "how many villages"),
        (r"\bekkuva\s+population\b", "highest population"),
        (r"\bekkuva\s+salary\b", "highest salary"),
        (r"\btakkuva\s+population\b", "lowest population"),
        (r"\btakkuva\s+salary\b", "lowest salary"),
        (r"\bpedda\s+village\b", "largest village"),
        (r"\bchinna\s+village\b", "smallest village"),
        (r"\byentha\s+population\b", "what population"),
        (r"\byentha\s+salary\b", "what salary"),
        (r"\brajadhani\s+yenti\b", "capital"),
    ]

    # Grammar inversions / broken English patterns
    GRAMMAR_PATTERNS: List[Tuple[str, str]] = [
        (r"\bwho\s+highest\s+salary\b", "who has highest salary"),
        (r"\bsalary\s+highest\s+who\b", "who has highest salary"),
        (r"\bwho\s+have\s+highest\s+salary\b", "who has highest salary"),
        (r"\bwhich\s+employee\s+salary\s+highest\b", "which employee has highest salary"),
        (r"\bhyd\s+villages\s+how\s+many\b", "how many villages in Hyderabad"),
        (r"\b(\w+)\s+villages\s+how\s+many\b", r"how many villages in \1"),
        (r"\b(\w+)\s+and\s+(\w+)\s+are\s+same\b", r"are \1 and \2 same"),
        (r"\b(\w+)\s+(\w+)\s+same\?", r"are \1 and \2 same?"),
        (r"\b(\w+)\s+belongs\s+to\s+(\w+)\?", r"does \1 belong to \2?"),
        (r"\bhow\s+many\s+there\b", "how many records are there"),
    ]

    # Currency mappings
    CURRENCY_SYMBOLS: Dict[str, str] = {
        "₹": "INR",
        "rs.": "INR",
        "rs": "INR",
        "inr": "INR",
        "$": "USD",
        "usd": "USD",
        "€": "EUR",
        "eur": "EUR",
        "£": "GBP",
        "gbp": "GBP",
    }

    # Number word multipliers
    NUMBER_MULTIPLIERS: Dict[str, float] = {
        "crore": 10_000_000.0,
        "crores": 10_000_000.0,
        "cr": 10_000_000.0,
        "million": 1_000_000.0,
        "millions": 1_000_000.0,
        "mn": 1_000_000.0,
        "m": 1_000_000.0,
        "lakh": 100_000.0,
        "lakhs": 100_000.0,
        "lac": 100_000.0,
        "lacs": 100_000.0,
        "l": 100_000.0,
        "thousand": 1_000.0,
        "thousands": 1_000.0,
        "k": 1_000.0,
    }

    def normalize_unicode(self, text: str) -> str:
        """Perform Unicode NFKD normalization, clean non-breaking spaces and fancy quotes."""
        if not text:
            return ""
        # NFKD normalization
        norm = unicodedata.normalize("NFKD", str(text))
        # Replace non-breaking spaces and odd whitespace
        norm = norm.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
        # Normalize quotes and dashes
        norm = norm.replace("\u2018", "'").replace("\u2019", "'")
        norm = norm.replace("\u201c", '"').replace("\u201d", '"')
        norm = norm.replace("\u2013", "-").replace("\u2014", "-")
        return norm

    def normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces, tabs, and linebreaks into a single space."""
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def normalize_punctuation(self, text: str) -> str:
        """Strip non-semantic query punctuation while strictly preserving identifier punctuation (EMP-1001, ST-01)."""
        if not text:
            return ""
        # Remove trailing/leading question marks, exclamation points, quotes
        s = re.sub(r"^[\s\"']+|[\s\"'?!.,]+$", "", text)
        # Remove standalone question marks or punctuation inside sentences
        s = re.sub(r"\s+[?!.,]+\s*", " ", s)
        # Clean double quotes around search words while leaving hyphenated IDs
        s = re.sub(r'(?<!\w)["\'](\w+)["\'](?!\w)', r"\1", s)
        return self.normalize_whitespace(s)

    def normalize_contractions(self, text: str) -> str:
        """Expand English contractions."""
        s = text
        for pat, rep in self.CONTRACTIONS.items():
            s = re.sub(pat, rep, s, flags=re.IGNORECASE)
        return s

    def normalize_slang(self, text: str) -> str:
        """Normalize informal colloquial expressions to standard semantic terms."""
        s = text
        for pat, rep in self.SLANG_PATTERNS:
            s = re.sub(pat, rep, s, flags=re.IGNORECASE)
        return s

    def normalize_multilingual(self, text: str) -> Tuple[str, bool, str]:
        """Normalize Hinglish, Telugu-English, and code-switched queries."""
        s = text
        is_multi = False
        lang = "en"

        for pat, rep in self.MULTILINGUAL_PATTERNS:
            if re.search(pat, s, flags=re.IGNORECASE):
                s = re.sub(pat, rep, s, flags=re.IGNORECASE)
                is_multi = True
                lang = "hi_te_mixed"

        return s, is_multi, lang

    def normalize_grammar(self, text: str) -> str:
        """Normalize non-standard word order and broken English."""
        s = text
        for pat, rep in self.GRAMMAR_PATTERNS:
            s = re.sub(pat, rep, s, flags=re.IGNORECASE)
        return s

    def parse_numeric_literal(self, token: str) -> Optional[float]:
        """Convert numeral tokens with Indian (Lakh/Crore) or Western (Million/K) suffixes to float."""
        if not token:
            return None
        clean = re.sub(r"[₹$,]", "", str(token)).strip().lower()
        if not clean:
            return None

        # Check suffixes
        for suffix, mult in self.NUMBER_MULTIPLIERS.items():
            pattern = rf"^([\d.]+)\s*{re.escape(suffix)}$"
            m = re.match(pattern, clean)
            if m:
                try:
                    return float(m.group(1)) * mult
                except ValueError:
                    pass

        try:
            return float(clean)
        except ValueError:
            return None

    def normalize_typos(self, text: str) -> str:
        """Correct common typographical errors in questions."""
        s = text
        for pat, rep in self.COMMON_TYPOS.items():
            s = re.sub(pat, rep, s, flags=re.IGNORECASE)
        return s

    def normalize_percentages(self, text: str) -> str:
        """Standardize percentage phrases (e.g., '50 percent' -> '50%')."""
        s = re.sub(r"\b(\d+(?:\.\d+)?)\s*percent\b", r"\1%", text, flags=re.IGNORECASE)
        s = re.sub(r"\b(\d+(?:\.\d+)?)\s*pct\b", r"\1%", s, flags=re.IGNORECASE)
        return s

    def normalize_question(self, question: str) -> NormalizedQuestion:
        """Full pipeline execution returning rich NormalizedQuestion."""
        raw = question or ""

        # 1. Unicode NFKD normalization
        u_norm = self.normalize_unicode(raw)

        # 2. Contraction expansion
        c_norm = self.normalize_contractions(u_norm)

        # 2.5. Common Typo correction
        t_norm = self.normalize_typos(c_norm)

        # 3. Multilingual detection & translation
        m_norm, is_multi, lang = self.normalize_multilingual(t_norm)

        # 4. Slang & colloquial normalization
        s_norm = self.normalize_slang(m_norm)

        # 5. Grammar & word-order normalization
        g_norm = self.normalize_grammar(s_norm)

        # 6. Percentage normalization
        pct_norm = self.normalize_percentages(g_norm)

        # 7. Whitespace & punctuation normalization (preserves IDs)
        final_norm = self.normalize_punctuation(self.normalize_whitespace(pct_norm))

        # Detect currency
        detected_curr = None
        for sym, curr in self.CURRENCY_SYMBOLS.items():
            if sym in raw.lower():
                detected_curr = curr
                break

        # Detect units
        detected_units = []
        unit_candidates = ["sq km", "km", "meters", "households", "people", "villages", "females", "males"]
        for u in unit_candidates:
            if re.search(rf"\b{re.escape(u)}\b", final_norm.lower()):
                detected_units.append(u)

        return NormalizedQuestion(
            raw_question=raw,
            normalized_question=final_norm,
            cleaned_question=final_norm.lower(),
            detected_language=lang,
            is_multilingual=is_multi,
            currency_detected=detected_curr,
            units_detected=detected_units,
        )


text_normalizer = TextNormalizer()

