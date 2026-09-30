"""Dataset-aware fuzzy search using RapidFuzz with confidence scoring and ambiguity detection."""

import difflib
from typing import Any, Dict, List, Optional, Tuple
from app.dataset.indexer import dataset_indexer
from app.utils.logger import logger

try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


class FuzzySearchEngine:
    """Performs dataset-aware fuzzy matching against columns and unique values."""

    def search_column(self, candidate: str, available_columns: List[str], threshold: float = 0.70) -> List[Dict[str, Any]]:
        """Fuzzy match a term against available column names."""
        c_lower = candidate.strip().lower()
        results = []

        for col in available_columns:
            col_l = col.lower()
            if HAS_RAPIDFUZZ:
                score = fuzz.ratio(c_lower, col_l) / 100.0
                token_score = fuzz.token_sort_ratio(c_lower, col_l) / 100.0
                best_score = max(score, token_score)
            else:
                best_score = difflib.SequenceMatcher(None, c_lower, col_l).ratio()

            if best_score >= threshold:
                results.append({
                    "column": col,
                    "confidence": round(best_score, 3),
                    "match_type": "fuzzy_column"
                })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def search_value_in_column(
        self,
        candidate: str,
        column: str,
        threshold: float = 0.75
    ) -> Optional[Dict[str, Any]]:
        """Fuzzy match candidate string against unique values of a specific column."""
        c_clean = candidate.strip().lower()
        uniques = dataset_indexer.get_column_values(column)
        if not uniques:
            return None

        val_map = {str(v).strip().lower(): v for v in uniques if v is not None}
        str_keys = list(val_map.keys())

        if HAS_RAPIDFUZZ:
            match = process.extractOne(c_clean, str_keys, scorer=fuzz.ratio)
            if match and (match[1] / 100.0) >= threshold:
                best_key, score = match[0], match[1] / 100.0
                return {
                    "column": column,
                    "matched_value": val_map[best_key],
                    "raw_key": best_key,
                    "confidence": round(score, 3),
                    "match_type": "fuzzy_value"
                }
        else:
            scored = []
            for k in str_keys:
                r = difflib.SequenceMatcher(None, c_clean, k).ratio()
                if r >= threshold:
                    scored.append((r, k))
            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                top_score, top_key = scored[0]
                return {
                    "column": column,
                    "matched_value": val_map[top_key],
                    "raw_key": top_key,
                    "confidence": round(top_score, 3),
                    "match_type": "fuzzy_value"
                }

        return None


fuzzy_search = FuzzySearchEngine()
