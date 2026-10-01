"""Database-backed Vector Store: Persists document embeddings and performs similarity retrieval."""

import json
import math
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session

from app.database.connection import db_manager
from app.database.models import RAGDocumentModel
from app.database.repositories import rag_doc_repo
from app.rag.embeddings import ollama_embeddings
from app.utils.logger import logger


class DatabaseVectorStore:
    """Manages persistent embeddings and semantic retrieval across database tables."""

    def __init__(self):
        self.repo = rag_doc_repo
        self.embeddings = ollama_embeddings

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two float vectors."""
        if not vec1 or not vec2:
            return 0.0
        min_len = min(len(vec1), len(vec2))
        dot = sum(vec1[i] * vec2[i] for i in range(min_len))
        norm1 = math.sqrt(sum(x * x for x in vec1[:min_len]))
        norm2 = math.sqrt(sum(x * x for x in vec2[:min_len]))
        if norm1 <= 0.0 or norm2 <= 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        compute_embeddings: bool = True
    ) -> int:
        """Persist documents with optional vector embedding generation."""
        count = 0
        session = db_manager.get_session()
        try:
            for doc in documents:
                doc_id = doc["document_id"]
                content = doc["content"]
                doc_type = doc.get("document_type", "GENERAL")
                meta = doc.get("metadata", {})
                dataset_id = doc.get("dataset_id")

                emb = None
                if compute_embeddings:
                    emb = self.embeddings.embed_text(content)

                emb_str = json.dumps(emb) if emb else None
                existing = session.query(RAGDocumentModel).filter_by(document_id=doc_id).first()
                if not existing:
                    new_doc = RAGDocumentModel(
                        document_id=doc_id,
                        dataset_id=dataset_id,
                        document_type=doc_type,
                        content=content,
                        doc_metadata=meta,
                        embedding=emb_str
                    )
                    session.add(new_doc)
                else:
                    existing.content = content
                    existing.document_type = doc_type
                    existing.doc_metadata = meta
                    if emb_str:
                        existing.embedding = emb_str
                count += 1
            session.commit()
            logger.info(f"Vector Store: Saved {count} documents to database.")
            return count
        except Exception as ex:
            session.rollback()
            logger.error(f"Vector Store add_documents failed: {ex}")
            raise
        finally:
            session.close()

    def similarity_search_by_vector(
        self,
        query_vector: List[float],
        k: int = 5,
        document_type: Optional[str] = None,
        dataset_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k most similar documents using cosine similarity."""
        session = db_manager.get_session()
        try:
            q = session.query(RAGDocumentModel)
            if document_type:
                q = q.filter_by(document_type=document_type)
            if dataset_id:
                q = q.filter_by(dataset_id=dataset_id)

            docs = q.all()
            scored: List[Tuple[float, RAGDocumentModel]] = []

            for doc in docs:
                if not doc.embedding:
                    continue
                emb = json.loads(doc.embedding)
                sim = self._cosine_similarity(query_vector, emb)
                scored.append((sim, doc))

            # Rank descending by similarity
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:k]

            return [
                {
                    "document_id": doc.document_id,
                    "dataset_id": doc.dataset_id,
                    "document_type": doc.document_type,
                    "content": doc.content,
                    "metadata": doc.doc_metadata,
                    "score": round(sim, 4),
                }
                for sim, doc in top
            ]
        finally:
            session.close()

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        document_type: Optional[str] = None,
        dataset_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Compute query embedding and retrieve top-k similar documents."""
        q_vec = self.embeddings.embed_query(query)
        return self.similarity_search_by_vector(
            query_vector=q_vec,
            k=k,
            document_type=document_type,
            dataset_id=dataset_id
        )

    def count_documents(self) -> int:
        session = db_manager.get_session()
        try:
            return session.query(RAGDocumentModel).count()
        finally:
            session.close()


database_vector_store = DatabaseVectorStore()
