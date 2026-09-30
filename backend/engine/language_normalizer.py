"""Multilingual Language Normalizer (Deliverable 7).

Normalizes code-switching, regional Indian English (Hinglish, Telugu-English,
Hindi-English, Tamil-English, Kannada-English), and Butler/broken English
into canonical English semantic representations.
"""

from typing import Any, Dict, List, Optional
from app.utils.multilingual import multilingual_normalizer
from app.utils.normalization import normalize_question


class LanguageNormalizer:
    """Universal Multilingual and Broken English Normalizer."""

    def normalize(self, text: str) -> Dict[str, Any]:
        """Normalize input query text into canonical English form."""
        step1 = multilingual_normalizer.normalize_code_switching(text)
        step2 = normalize_question(step1)

        detected_lang = "english"
        lower = text.lower()
        if any(w in lower for w in ["mein", "kya", "hai", "kitne", "batao", "ka", "ki"]):
            detected_lang = "hinglish"
        elif any(w in lower for w in ["enti", "ekkada", "unnaru", "cheppu"]):
            detected_lang = "telugu-english"
        elif any(w in lower for w in ["enna", "enga", "ethana", "soller"]):
            detected_lang = "tamil-english"
        elif any(w in lower for w in ["yenu", "elli", "alli", "eshtu"]):
            detected_lang = "kannada-english"

        return {
            "raw_input": text,
            "normalized": step2,
            "detected_language": detected_lang,
            "was_modified": text.strip() != step2.strip()
        }


language_normalizer = LanguageNormalizer()

