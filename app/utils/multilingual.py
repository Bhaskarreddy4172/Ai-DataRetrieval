"""Multilingual and Code-Switching Normalization Engine.

Translates and normalizes code-switched phrases, regional Indian language markers
(Hinglish, Telugu-English, Tamil-English, Kannada-English, Malayalam-English),
broken grammar, and inverted word-order patterns into standardized semantic representations.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from app.utils.logger import logger


class MultilingualNormalizer:
    """Normalizes multilingual code-switching, broken grammar, and non-standard syntax."""

    # 1. Indian language question particles and copulas
    REGIONAL_QUESTION_PATTERNS: List[Tuple[str, str]] = [
        # Telugu-English
        (r"\b([a-zA-Z0-9\s]+?)\s+capital\s+enti\b", r"what is the capital of \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+enti\b", r"what is \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+ekkada\b", r"where is \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+lo\s+entha\s+mandhi\b", r"how many in \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+lo\s+unnaru\b", r"who are in \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+lo\b", r"in \1"),
        (r"\bcheppu\b", r"tell me"),
        (r"\bchudu\b", r"show me"),

        # Hindi / Hinglish
        (r"\b([a-zA-Z0-9\s]+?)\s+ka\s+capital\s+kya\s+hai\b", r"what is the capital of \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+ka\s+capital\s+what\b", r"what is the capital of \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+ka\s+capital\b", r"what is the capital of \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+ki\s+capital\b", r"what is the capital of \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+kis\s+state\s+mein\s+hai\b", r"which state is \1 in"),
        (r"\b([a-zA-Z0-9\s]+?)\s+kis\s+state\s+mein\s+belongs\b", r"which state does \1 belong to"),
        (r"\b([a-zA-Z0-9\s]+?)\s+kis\s+state\s+me\b", r"which state is \1 in"),
        (r"\b([a-zA-Z0-9\s]+?)\s+mein\s+kitne\s+([a-zA-Z\s]+?)\s+work\s+karte\s+hain\b", r"how many \2 in \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+mein\s+kitne\s+([a-zA-Z\s]+?)\s+hain\b", r"how many \2 in \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+me\s+kitne\b", r"how many in \1"),
        (r"\bkitne\s+([a-zA-Z\s]+?)\s+hain\b", r"how many \1"),
        (r"\bkitne\s+hain\b", r"how many"),
        (r"\bkitna\s+hai\b", r"how much"),
        (r"\b([a-zA-Z0-9\s]+?)\s+kahan\s+hai\b", r"where is \1"),
        (r"\bkaun\s+hai\b", r"who is"),
        (r"\bkon\s+hai\b", r"who is"),
        (r"\bbatao\b", r"tell me"),
        (r"\bdikhao\b", r"show me"),
        (r"\bsabse\s+jyada\b", r"highest"),
        (r"\bsabse\s+kam\b", r"lowest"),
        (r"\b([a-zA-Z0-9\s]+?)\s+wala\s+employee\b", r"employee with \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+wali\b", r"\1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+wala\b", r"\1"),

        # Tamil-English
        (r"\b([a-zA-Z0-9\s]+?)\s+enna\b", r"what is \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+enga\b", r"where is \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+la\b", r"in \1"),
        (r"\bethana\b", r"how many"),

        # Kannada-English
        (r"\b([a-zA-Z0-9\s]+?)\s+yenu\b", r"what is \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+enu\b", r"what is \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+elli\b", r"where is \1"),
        (r"\beshtu\b", r"how many"),

        # Malayalam-English
        (r"\b([a-zA-Z0-9\s]+?)\s+entha\b", r"what is \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+evide\b", r"where is \1"),
        (r"\bethra\b", r"how many"),
    ]

    # 2. Broken grammar and word-order inversion patterns
    INVERTED_SYNTAX_PATTERNS: List[Tuple[str, str]] = [
        # Word-order inversion: "salary max who?", "salary highest who?", "highest salary who?"
        (r"\b([a-zA-Z]+)\s+(?:max|highest|maximum|top)\s+who\b", r"who has highest \1"),
        (r"\b(?:max|highest|maximum|top)\s+([a-zA-Z]+)\s+who\b", r"who has highest \1"),
        (r"\bwho\s+(?!(?:has|is|are|had)\b)([a-zA-Z]+)\s+(?:max|highest|maximum|top)\b", r"who has highest \1"),
        (r"\bwho\s+(?:max|highest|maximum|top)\s+([a-zA-Z]+)\b", r"who has highest \1"),
        (r"\b([a-zA-Z]+)\s+(?:min|lowest|minimum|bottom)\s+who\b", r"who has lowest \1"),
        (r"\b(?:min|lowest|minimum|bottom)\s+([a-zA-Z]+)\s+who\b", r"who has lowest \1"),
        (r"\bwho\s+(?!(?:has|is|are|had)\b)([a-zA-Z]+)\s+(?:min|lowest|minimum|bottom)\b", r"who has lowest \1"),

        # Broken possessives / questions: "what salary ananya have?", "where ananya working?"
        (r"\bwhat\s+([a-zA-Z]+)\s+([a-zA-Z\s]+?)\s+have\b", r"what is \2's \1"),
        (r"\bwhat\s+([a-zA-Z]+)\s+([a-zA-Z\s]+?)\s+has\b", r"what is \2's \1"),
        (r"\bwhere\s+([a-zA-Z\s]+?)\s+working\b", r"where does \1 work"),
        (r"\bwhere\s+([a-zA-Z\s]+?)\s+lives\b", r"where does \1 live"),

        # Elliptical / short forms: "hyd which state?", "tell me hyd belongs which state"
        (r"\btell\s+me\s+([a-zA-Z0-9]+)\s+belongs?\s+(?:to\s+)?which\s+state\b", r"which state does \1 belong to"),
        (r"\b(?!(?:and|or|in|to|the|that|this|of|for|with|is|are|a|an|tell|me)\b)([a-zA-Z0-9]+)\s+belongs?\s+(?:to\s+)?which\s+state\b", r"which state does \1 belong to"),
        (r"\b(?!(?:and|or|in|to|the|that|this|of|for|with|is|are|a|an|tell|me)\b)([a-zA-Z0-9]+)\s+belongs?\s+(?:to\s+)?which\s+([a-zA-Z]+)\b", r"which \2 does \1 belong to"),
        (r"\b(?!(?:and|or|in|to|the|that|this|of|for|with|is|are|a|an|tell|me)\b)([a-zA-Z0-9]+)\s+which\s+state\b", r"which state is \1 in"),

        # Informal phrases: "who haz highest", "highest paid guy", "who is getting more money"
        (r"\bwho\s+haz\s+highest\b", r"who has highest"),
        (r"\bwho\s+haz\s+", r"who has "),
        (r"\bhighest\s+paid\s+(?:guy|person|employee|dude)\b", r"who has highest salary"),
        (r"\blowest\s+paid\s+(?:guy|person|employee|dude)\b", r"who has lowest salary"),
        (r"\bwho\s+is\s+getting\s+more\s+money\b", r"who has highest salary"),
        (r"\btell\s+me\s+the\s+top\s+paid\s+person\b", r"who has highest salary"),
        (r"\bwhich\s+employee\s+makes\s+maximum\b", r"who has highest salary"),
        (r"\bwho\s+takes\s+biggest\s+salary\b", r"who has highest salary"),
        (r"\bwho\s+is\s+earning\s+most\s+amount\b", r"who has highest salary"),
        (r"\bwho\s+is\s+earning\s+maximum\b", r"who has highest salary"),
        (r"\bwho\s+got\s+max\s+money\b", r"who has highest salary"),
        (r"\bwho\s+got\s+more\s+salary\b", r"who has highest salary"),
        (r"\bwho\s+takes\s+home\s+the\s+biggest\s+(?:amount\s+of\s+)?money\b", r"who has highest salary"),
        (r"\bperson\s+getting\s+biggest\s+pay\b", r"highest salary"),
    ]

    # 3. Possessive Hindi markers: X ka Y, X ki Y, X ke Y -> Y of X
    POSSESSIVE_REGIONAL_PATTERNS: List[Tuple[str, str]] = [
        (r"\b([a-zA-Z0-9\s]+?)\s+ka\s+([a-zA-Z]+)\s+kya\s+hai\b", r"what is the \2 of \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+ki\s+([a-zA-Z]+)\s+kya\s+hai\b", r"what is the \2 of \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+ka\s+([a-zA-Z]+)\s+what\b", r"what is the \2 of \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+ki\s+([a-zA-Z]+)\s+what\b", r"what is the \2 of \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+ka\s+([a-zA-Z]+)\b", r"\2 of \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+ki\s+([a-zA-Z]+)\b", r"\2 of \1"),
        (r"\b([a-zA-Z0-9\s]+?)\s+ke\s+([a-zA-Z]+)\b", r"\2 of \1"),
    ]

    def normalize(self, question: str) -> str:
        """Apply multilingual and code-switching normalization."""
        if not question or not question.strip():
            return ""

        q = question.strip()

        # Step 1: Regional Indian language question patterns
        for pattern, replacement in self.REGIONAL_QUESTION_PATTERNS:
            if re.search(pattern, q, flags=re.IGNORECASE):
                q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)

        # Step 2: Regional possessives (ka / ki / ke)
        for pattern, replacement in self.POSSESSIVE_REGIONAL_PATTERNS:
            if re.search(pattern, q, flags=re.IGNORECASE):
                q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)

        # Step 3: Broken grammar and inverted syntax
        for pattern, replacement in self.INVERTED_SYNTAX_PATTERNS:
            if re.search(pattern, q, flags=re.IGNORECASE):
                q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)

        # Step 4: Cleanup whitespace and punctuation
        q = re.sub(r"\s+", " ", q).strip()

        return q


multilingual_normalizer = MultilingualNormalizer()
