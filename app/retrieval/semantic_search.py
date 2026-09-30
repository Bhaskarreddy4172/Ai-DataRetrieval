"""Semantic search engine using TF-IDF / Scikit-Learn vector similarity for candidate retrieval."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from app.dataset.semantic_mapper import UNIVERSAL_ONTOLOGY
from app.utils.logger import logger

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class SemanticSearchEngine:
    """Computes vector similarity between query phrases, column concepts, and ontology terms."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1) if HAS_SKLEARN else None

    def search_semantic_candidate(
        self,
        query_term: str,
        candidates: List[str],
        threshold: float = 0.60
    ) -> List[Dict[str, Any]]:
        """Compute cosine similarity between query term and candidate strings."""
        if not HAS_SKLEARN or not candidates or not query_term:
            return []

        corpus = [query_term] + candidates
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            sim_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            results = []
            for idx, score in enumerate(sim_scores):
                if score >= threshold:
                    results.append({
                        "candidate": candidates[idx],
                        "similarity": round(float(score), 3),
                        "match_type": "vector_semantic"
                    })

            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []


semantic_search = SemanticSearchEngine()
