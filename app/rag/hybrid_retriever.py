"""Hybrid Retriever: Merges vector semantic search, exact/alias matching, and lexical scoring."""

import re
from typing import Any, Dict, List, Optional, Set
from app.database.connection import db_manager
from app.database.models import RAGDocumentModel
from app.database.repositories import entity_repo
from app.rag.embeddings import ollama_embeddings
from app.rag.vector_store import database_vector_store
from app.utils.logger import logger


class HybridRetriever:
    """Multi-stage retriever combining dense embeddings, alias resolution, and keyword token matching."""

    def __init__(self):
        self.vector_store = database_vector_store
        self.entity_repo = entity_repo
        self.embeddings = ollama_embeddings

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return set(re.findall(r"\w+", text.lower()))

    def _lexical_score(self, query_tokens: Set[str], content: str) -> float:
        if not query_tokens:
            return 0.0
        doc_tokens = self._tokenize(content)
        if not doc_tokens:
            return 0.0
        overlap = query_tokens.intersection(doc_tokens)
        return len(overlap) / len(query_tokens)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        document_type: Optional[str] = None,
        dataset_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval over RAG catalog."""
        clean_query = query.strip()
        if not clean_query:
            return []

        # 1. Exact entity & alias check
        resolved_entity = self.entity_repo.resolve_alias(clean_query)
        enhanced_query = f"{clean_query} {resolved_entity}" if resolved_entity else clean_query

        # 2. Dense vector search
        vector_results = self.vector_store.similarity_search(
            query=enhanced_query,
            k=max(k * 2, 10),
            document_type=document_type,
            dataset_id=dataset_id
        )

        # 3. Lexical matching on all candidate documents
        query_tokens = self._tokenize(clean_query)
        session = db_manager.get_session()
        try:
            q = session.query(RAGDocumentModel)
            if document_type:
                q = q.filter_by(document_type=document_type)
            if dataset_id:
                q = q.filter_by(dataset_id=dataset_id)
            candidates = q.all()
        finally:
            session.close()

        lexical_scores: Dict[str, float] = {}
        for c in candidates:
            score = self._lexical_score(query_tokens, c.content)
            if score > 0.0:
                lexical_scores[c.document_id] = score

        # 4. Hybrid Reciprocal Rank Fusion / Weighted combination
        doc_map: Dict[str, Dict[str, Any]] = {}

        # Incorporate vector candidates
        for vr in vector_results:
            did = vr["document_id"]
            v_score = vr.get("score", 0.0)
            l_score = lexical_scores.get(did, 0.0)
            # 60% semantic + 40% lexical
            final_score = 0.6 * v_score + 0.4 * l_score
            doc_map[did] = {
                **vr,
                "score": round(final_score, 4),
                "semantic_score": v_score,
                "lexical_score": l_score,
            }

        # Incorporate strong lexical candidates that might have low vector score
        for c in candidates:
            did = c.document_id
            if did not in doc_map and lexical_scores.get(did, 0.0) > 0.4:
                l_score = lexical_scores[did]
                doc_map[did] = {
                    "document_id": c.document_id,
                    "dataset_id": c.dataset_id,
                    "document_type": c.document_type,
                    "content": c.content,
                    "metadata": c.doc_metadata,
                    "score": round(0.4 * l_score, 4),
                    "semantic_score": 0.0,
                    "lexical_score": l_score,
                }

        # Sort combined results descending by composite score
        combined = list(doc_map.values())
        combined.sort(key=lambda x: x["score"], reverse=True)

        return combined[:k]


hybrid_retriever = HybridRetriever()
