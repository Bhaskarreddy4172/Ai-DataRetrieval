"""Spell Correction Engine (Deliverable 5).

Combines RapidFuzz, Edit Distance, Levenshtein, Double Metaphone, and Soundex
against active dataset vocabulary, entities, and location dictionaries.
"""

from typing import Any, Dict, List, Optional
from app.dataset.spell_checker import dataset_spell_checker, DatasetSpellChecker
from app.dataset.loader import dataset_loader
from app.utils.fuzzy_match import LOCATION_ABBREVIATIONS


class SpellCorrectionEngine:
    """Universal Dataset-Aware Spell Correction Engine."""

    def __init__(self):
        self._checker = dataset_spell_checker

    def correct(self, word: str, column: Optional[str] = None) -> Dict[str, Any]:
        """Correct spelling mistakes and typos with confidence score."""
        clean_word = word.strip()
        lower_word = clean_word.lower()

        # 1. Check direct abbreviation / shortcut expansion
        if lower_word in LOCATION_ABBREVIATIONS:
            return {
                "original": word,
                "corrected": LOCATION_ABBREVIATIONS[lower_word],
                "confidence": 0.99,
                "method": "abbreviation_lookup"
            }

        # 2. Check dataset-aware vocabulary spell checker
        ds_match = self._checker.match_token(clean_word, column=column)
        if ds_match and ds_match.get("matched"):
            return {
                "original": word,
                "corrected": ds_match.get("canonical_value", clean_word),
                "confidence": round(float(ds_match.get("confidence", 0.95)), 2),
                "method": ds_match.get("match_type", "dataset_aware")
            }

        # 3. Check query word corrections
        if lower_word in self._checker.QUERY_WORD_CORRECTIONS:
            return {
                "original": word,
                "corrected": self._checker.QUERY_WORD_CORRECTIONS[lower_word],
                "confidence": 0.98,
                "method": "query_word_dictionary"
            }

        return {
            "original": word,
            "corrected": word,
            "confidence": 1.0 if not clean_word else 0.5,
            "method": "identity"
        }


spell_correction_engine = SpellCorrectionEngine()
