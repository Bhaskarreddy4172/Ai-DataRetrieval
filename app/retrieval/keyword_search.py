"""Keyword search engine for lexical pattern matching, token extraction, and exact index lookups."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from app.dataset.indexer import dataset_indexer

KEYWORD_OPERATORS: Dict[str, str] = {
    "highest": "MAX",
    "maximum": "MAX",
    "max": "MAX",
    "top": "MAX",
    "best": "MAX",
    "lowest": "MIN",
    "minimum": "MIN",
    "min": "MIN",
    "bottom": "MIN",
    "worst": "MIN",
    "least": "MIN",
    "average": "AVERAGE",
    "avg": "AVERAGE",
    "mean": "AVERAGE",
    "median": "MEDIAN",
    "total": "SUM",
    "sum": "SUM",
    "count": "COUNT",
    "how many": "COUNT",
    "number of": "COUNT",
    "outlier": "OUTLIER",
    "unusual": "OUTLIER",
    "anomalous": "OUTLIER"
}


class KeywordSearchEngine:
    """Extracts lexical keywords, operational tokens, and exact index matches."""

    def extract_operators(self, text: str) -> List[Tuple[str, str]]:
        """Identify operational keywords in the query text."""
        text_lower = text.lower()
        found = []
        for kw, op in KEYWORD_OPERATORS.items():
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, text_lower):
                found.append((kw, op))
        return found

    def exact_value_match(self, token: str, columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Look up exact string matches across indexed columns."""
        token_clean = token.strip().lower()
        matches = []
        target_cols = columns or list(dataset_indexer.column_uniques.keys())

        for col in target_cols:
            val_map = dataset_indexer.column_uniques_lower.get(col, {})
            if token_clean in val_map:
                matches.append({
                    "column": col,
                    "value": val_map[token_clean],
                    "match_type": "exact",
                    "confidence": 1.0
                })
        return matches


keyword_search = KeywordSearchEngine()
