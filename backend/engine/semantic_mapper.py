"""Semantic Column Mapping Engine (Deliverable 8).

Dynamically maps natural language user terminology (synonyms, slang, domain terms)
to actual schema column headers across arbitrary datasets without domain hardcoding.
"""

from typing import Any, Dict, List, Optional
from app.dataset.semantic_mapper import semantic_column_mapper
from app.dataset.loader import dataset_loader


class SemanticMappingEngine:
    """Universal Semantic Column Mapper."""

    def map_term_to_column(
        self,
        term: str,
        available_columns: Optional[List[str]] = None
    ) -> Optional[str]:
        """Map a user keyword (e.g. 'pay', 'income', 'cost', 'where') to an actual column name."""
        cols = available_columns or dataset_loader.get_columns()
        return semantic_column_mapper.map_column(term, cols)

    def resolve_all_columns(
        self,
        query: str,
        available_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Extract and map all potential column references from user question."""
        cols = available_columns or dataset_loader.get_columns()
        mapped: Dict[str, str] = {}
        words = query.lower().split()
        for w in words:
            matched = semantic_column_mapper.map_column(w, cols)
            if matched:
                mapped[w] = matched

        return {
            "query": query,
            "mapped_columns": mapped,
            "available_columns": cols
        }


semantic_mapping_engine = SemanticMappingEngine()

