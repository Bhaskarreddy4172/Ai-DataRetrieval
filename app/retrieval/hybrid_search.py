"""Hybrid search engine implementing Section 8 6-tier search priority."""

from typing import Any, Dict, List, Optional, Tuple
from app.retrieval.keyword_search import keyword_search
from app.retrieval.fuzzy_search import fuzzy_search
from app.retrieval.semantic_search import semantic_search
from app.dataset.semantic_mapper import semantic_column_mapper
from app.dataset.indexer import dataset_indexer
from app.utils.fuzzy_match import expand_abbreviations
from app.utils.logger import logger


class HybridSearchEngine:
    """Orchestrates 6-tier prioritized candidate retrieval across dataset schema and values."""

    def resolve_column(
        self,
        query_token: str,
        available_columns: List[str],
        schema_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Resolve a column reference via strict 6-tier priority."""
        token_clean = query_token.strip().lower()

        # Priority 1: Exact Match
        for col in available_columns:
            if col.lower() == token_clean:
                return {"column": col, "confidence": 1.0, "tier": 1, "tier_name": "exact_match"}

        # Priority 2: Normalized Exact Match (abbreviations expanded, punctuation removed)
        expanded = expand_abbreviations(token_clean)
        for col in available_columns:
            if col.lower() == expanded or col.lower().replace("_", " ") == expanded.replace("_", " "):
                return {"column": col, "confidence": 0.98, "tier": 2, "tier_name": "normalized_exact"}

        # Priority 3: Dataset Aliases & Semantic Ontology
        mapped = semantic_column_mapper.map_column(token_clean, available_columns, schema_dict)
        if mapped.get("column") and mapped.get("confidence", 0.0) >= 0.90:
            return {
                "column": mapped["column"],
                "confidence": mapped["confidence"],
                "tier": 3,
                "tier_name": "dataset_alias_ontology",
                "is_ambiguous": mapped.get("is_ambiguous", False),
                "candidates": mapped.get("candidates", [])
            }

        # Priority 4: High-Confidence Fuzzy Match (RapidFuzz > 0.85)
        fuzzy_matches = fuzzy_search.search_column(token_clean, available_columns, threshold=0.80)
        if fuzzy_matches and fuzzy_matches[0]["confidence"] >= 0.85:
            return {
                "column": fuzzy_matches[0]["column"],
                "confidence": fuzzy_matches[0]["confidence"],
                "tier": 4,
                "tier_name": "high_confidence_fuzzy"
            }

        # Priority 5: Semantic Match / Vector Cosine Similarity
        sem_matches = semantic_search.search_semantic_candidate(token_clean, available_columns, threshold=0.65)
        if sem_matches:
            return {
                "column": sem_matches[0]["candidate"],
                "confidence": sem_matches[0]["similarity"],
                "tier": 5,
                "tier_name": "vector_semantic"
            }

        # Priority 6: Moderate Fuzzy Match or Low-Confidence
        if fuzzy_matches:
            return {
                "column": fuzzy_matches[0]["column"],
                "confidence": fuzzy_matches[0]["confidence"],
                "tier": 6,
                "tier_name": "moderate_fuzzy",
                "is_ambiguous": len(fuzzy_matches) > 1
            }

        return {"column": None, "confidence": 0.0, "tier": None, "tier_name": "unmatched"}


hybrid_search = HybridSearchEngine()
