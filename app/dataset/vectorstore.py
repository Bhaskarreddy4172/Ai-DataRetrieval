"""Hybrid Retrieval and Vector Store Engine for Dataset Entities and Rows.

Combines exact lexical matching, RapidFuzz token matching, Double Metaphone phonetic matching,
and dense semantic vector embeddings with Reciprocal Rank Fusion (RRF).
Supports FAISS if available, with graceful high-performance NumPy vectorized cosine fallback.
"""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from app.utils.logger import logger
from app.utils.phonetic import phonetic_match_score


class DatasetVectorStore:
    """Hybrid Retrieval Vector Store supporting dense semantic embeddings and fuzzy/phonetic indexing."""

    def __init__(self):
        self._dataset_name: str = ""
        self._dataset_hash: str = ""
        self._entity_texts: List[str] = []
        self._entity_metadata: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None
        self._faiss_index: Any = None
        self._vocab_weights: Dict[str, float] = {}
        self._dim: int = 64
        self._encoder_model: Any = None
        self._init_encoder()

    def _init_encoder(self) -> None:
        """Initialize sentence-transformers model if available, otherwise use fast hashed projection."""
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._dim = 384
            logger.info("DatasetVectorStore: Loaded SentenceTransformer ('all-MiniLM-L6-v2') for 384-dim dense embeddings.")
        except Exception:
            self._encoder_model = None
            self._dim = 64
            logger.info("DatasetVectorStore: SentenceTransformer not available; using 64-dim hashed n-gram projection.")

    def clear(self) -> None:
        """Reset all in-memory vector indices and metadata."""
        self._dataset_name = ""
        self._dataset_hash = ""
        self._entity_texts = []
        self._entity_metadata = []
        self._embeddings = None
        self._faiss_index = None
        self._vocab_weights.clear()

    def build_index(self, df: pd.DataFrame, dataset_name: str = "", dataset_hash: str = "") -> None:
        """Build embedding and inverted lexical index across all categorical/string columns."""
        if df.empty:
            self.clear()
            return

        # Avoid repeated indexing if hash matches
        if dataset_hash and dataset_hash == self._dataset_hash and self._embeddings is not None:
            logger.info(f"DatasetVectorStore: Reusing cached vector index for '{dataset_name}'.")
            return

        self.clear()
        self._dataset_name = dataset_name
        self._dataset_hash = dataset_hash

        entities: List[str] = []
        metadata: List[Dict[str, Any]] = []

        # 1. Collect candidate entities and row representations
        for col in df.columns:
            if col.startswith("_"):
                continue
            series = df[col].dropna()
            # String / categorical columns
            if series.dtype == "object" or series.dtype.name == "category" or str(series.dtype) == "string":
                unique_vals = series.unique()
                for val in unique_vals:
                    val_str = str(val).strip()
                    if len(val_str) >= 2 and not val_str.replace(".", "", 1).isdigit():
                        entities.append(val_str)
                        metadata.append({
                            "type": "cell_value",
                            "column": col,
                            "value": val,
                            "text": val_str,
                        })

        # Also add row summary sentences for multi-column semantic lookup (capped at 500 rows for memory safety)
        for row_idx, row in df.head(500).iterrows():
            parts = [f"{c}: {row[c]}" for c in df.columns if not str(c).startswith("_") and pd.notna(row[c])]
            summary_text = ", ".join(parts)
            entities.append(summary_text)
            metadata.append({
                "type": "row_summary",
                "row_id": int(row.get("_internal_row_id", row_idx)),
                "text": summary_text,
            })

        self._entity_texts = entities
        self._entity_metadata = metadata

        if not entities:
            return

        # 2. Generate dense normalized embeddings
        self._embeddings = self._generate_embeddings(entities)

        # 3. Optional FAISS Index initialization
        try:
            import faiss
            dim = self._embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(self._embeddings.astype("float32"))
            logger.info(f"DatasetVectorStore: Built FAISS vector index with {len(entities)} vectors (dim={dim}).")
        except Exception:
            self._faiss_index = None
            logger.info("DatasetVectorStore: FAISS not available; using high-performance NumPy cosine similarity.")

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate normalized dense embeddings matching self._dim."""
        if self._encoder_model is not None:
            try:
                embs = self._encoder_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return embs.astype("float32")
            except Exception as e:
                logger.warning(f"SentenceTransformer encoding failed: {e}; falling back to hashed projection.")

        # High-performance, zero-external-dependency semantic hashed character & word n-gram projection
        num_texts = len(texts)
        embs = np.zeros((num_texts, self._dim), dtype=np.float32)

        for i, txt in enumerate(texts):
            clean = txt.lower().strip()
            tokens = re.findall(r"\w+", clean)
            vec = np.zeros(self._dim, dtype=np.float32)

            for t in tokens:
                # Word hash
                h_w = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % self._dim
                vec[h_w] += 1.0
                # Character 3-grams
                if len(t) >= 3:
                    for g in range(len(t) - 2):
                        gram = t[g:g+3]
                        h_g = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % self._dim
                        vec[h_g] += 0.5

            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
            embs[i] = vec

        return embs

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query, guaranteeing identical dimensionality to self._embeddings."""
        if self._encoder_model is not None:
            try:
                emb = self._encoder_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
                return emb.astype("float32")
            except Exception:
                pass

        clean = query.lower().strip()
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = re.findall(r"\w+", clean)
        for t in tokens:
            h_w = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % self._dim
            vec[h_w] += 1.0
            if len(t) >= 3:
                for g in range(len(t) - 2):
                    gram = t[g:g+3]
                    h_g = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % self._dim
                    vec[h_g] += 0.5
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec.reshape(1, -1)

    def vector_search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Perform dense semantic vector search."""
        if not query or not query.strip() or self._embeddings is None or len(self._entity_texts) == 0:
            return []

        q_vec = self._embed_query(query)

        if self._faiss_index is not None:
            try:
                k_search = min(top_k, len(self._entity_texts))
                distances, indices = self._faiss_index.search(q_vec.astype("float32"), k_search)
                results = []
                for score, idx in zip(distances[0], indices[0]):
                    if 0 <= idx < len(self._entity_metadata):
                        results.append((self._entity_metadata[idx], float(score)))
                return results
            except Exception as e:
                logger.warning(f"FAISS search failed ({e}); falling back to NumPy cosine similarity.")

        # Vectorized NumPy cosine similarity
        similarities = np.dot(self._embeddings, q_vec.T).squeeze()
        if similarities.ndim == 0:
            similarities = np.array([float(similarities)])

        k_eff = min(top_k, len(similarities))
        top_indices = np.argsort(-similarities)[:k_eff]
        results = []
        for idx in top_indices:
            results.append((self._entity_metadata[idx], float(similarities[idx])))
        return results

    def hybrid_search(
        self,
        query: str,
        column: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rank candidates using Exact Lexical + RapidFuzz + Phonetic + Dense Vector Similarity."""
        if not query or not query.strip() or not self._entity_texts:
            return []

        query_clean = query.strip().lower()
        scored_candidates: Dict[str, Dict[str, Any]] = {}

        # 1. Vector candidates
        v_results = self.vector_search(query, top_k=top_k * 2)
        for meta, score in v_results:
            if column and meta.get("column") != column:
                continue
            txt = meta.get("text", "")
            if txt not in scored_candidates:
                scored_candidates[txt] = {
                    "meta": meta,
                    "vec_score": max(0.0, float(score)),
                    "fuzzy_score": 0.0,
                    "phonetic_score": 0.0,
                    "exact": 0.0
                }
            else:
                scored_candidates[txt]["vec_score"] = max(scored_candidates[txt]["vec_score"], float(score))

        # 2. Exact & Fuzzy lexical matching
        for meta in self._entity_metadata:
            if column and meta.get("column") != column:
                continue
            txt = meta.get("text", "")
            txt_lower = txt.lower()

            # Exact match
            exact_score = 1.0 if txt_lower == query_clean else (0.8 if query_clean in txt_lower else 0.0)

            # RapidFuzz / string distance score
            try:
                from rapidfuzz.distance import Levenshtein
                fuzzy_score = float(Levenshtein.normalized_similarity(query_clean, txt_lower))
            except Exception:
                import difflib
                fuzzy_score = float(difflib.SequenceMatcher(None, query_clean, txt_lower).ratio())

            # Phonetic score
            phon_score = float(phonetic_match_score(query_clean, txt_lower))

            if exact_score > 0 or fuzzy_score > 0.65 or phon_score > 0.7:
                if txt not in scored_candidates:
                    scored_candidates[txt] = {
                        "meta": meta,
                        "vec_score": 0.0,
                        "fuzzy_score": fuzzy_score,
                        "phonetic_score": phon_score,
                        "exact": exact_score
                    }
                else:
                    scored_candidates[txt]["exact"] = max(scored_candidates[txt]["exact"], exact_score)
                    scored_candidates[txt]["fuzzy_score"] = max(scored_candidates[txt]["fuzzy_score"], fuzzy_score)
                    scored_candidates[txt]["phonetic_score"] = max(scored_candidates[txt]["phonetic_score"], phon_score)

        # 3. Composite score calculation
        ranked: List[Dict[str, Any]] = []
        for txt, data in scored_candidates.items():
            composite = (
                0.40 * data["exact"] +
                0.25 * data["fuzzy_score"] +
                0.20 * data["vec_score"] +
                0.15 * data["phonetic_score"]
            )
            ranked.append({
                "text": txt,
                "score": round(float(composite), 4),
                "metadata": data["meta"],
                "breakdown": {
                    "exact": data["exact"],
                    "fuzzy": round(data["fuzzy_score"], 2),
                    "vector": round(data["vec_score"], 2),
                    "phonetic": round(data["phonetic_score"], 2)
                }
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]


dataset_vector_store = DatasetVectorStore()
