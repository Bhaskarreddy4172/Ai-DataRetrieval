"""Entity Resolution Engine (Deliverable 6).

Resolves entities, aliases, acronyms, abbreviations, phonetic variants,
and plural/singular differences dynamically for any uploaded dataset.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from app.dataset.alias_resolver import entity_alias_resolver
from app.dataset.loader import dataset_loader
from app.dataset.vectorstore import dataset_vector_store
from backend.engine.spell_checker import spell_correction_engine


class EntityResolutionEngine:
    """Universal entity and alias resolution engine."""

    def resolve(
        self,
        query_term: str,
        df: Optional[pd.DataFrame] = None,
        available_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Resolve raw user query entity into canonical dataset entity."""
        term_clean = query_term.strip()
        active_df = df if df is not None else dataset_loader.dataframe
        cols = available_columns or (list(active_df.columns) if not active_df.empty else [])

        # 1. Check entity alias and abbreviation resolver
        alias_match = entity_alias_resolver.resolve_alias(term_clean, available_columns=cols)
        if alias_match:
            return {
                "input": query_term,
                "canonical_entity": alias_match["canonical_name"],
                "column": alias_match.get("column"),
                "confidence": alias_match.get("confidence", 0.95),
                "resolution_type": alias_match.get("match_type", "alias_lookup")
            }

        # 2. Check spell and typo resolution
        spell_res = spell_correction_engine.correct(term_clean)
        if spell_res and spell_res["confidence"] >= 0.85 and spell_res["corrected"] != term_clean:
            return {
                "input": query_term,
                "canonical_entity": spell_res["corrected"],
                "column": None,
                "confidence": spell_res["confidence"],
                "resolution_type": spell_res.get("method", "spell_correction")
            }

        # 3. Check hybrid vector store for candidate entities
        vector_cands = dataset_vector_store.hybrid_search(term_clean, top_k=1)
        if vector_cands and vector_cands[0]["score"] >= 0.35:
            top_cand = vector_cands[0]
            return {
                "input": query_term,
                "canonical_entity": top_cand["text"],
                "column": top_cand["metadata"].get("column"),
                "confidence": top_cand["score"],
                "resolution_type": "hybrid_vector_search"
            }

        return {
            "input": query_term,
            "canonical_entity": term_clean,
            "column": None,
            "confidence": 0.5,
            "resolution_type": "exact_literal"
        }


entity_resolution_engine = EntityResolutionEngine()

