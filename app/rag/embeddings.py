"""RAG Embeddings: Ollama embedding client with deterministic fallback."""

import hashlib
import json
import math
from typing import List, Optional
import requests

from app.config import settings
from app.utils.logger import logger


class OllamaEmbeddings:
    """Computes dense vector representations via Ollama API with graceful deterministic fallback."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        dim: int = 768
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or getattr(settings, "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.dim = getattr(settings, "EMBEDDING_DIM", dim)
        self._ollama_online: Optional[bool] = None
        self._last_check: float = 0.0

    def _is_ollama_available(self) -> bool:
        import time
        now = time.time()
        if self._ollama_online is False and (now - self._last_check) < 15.0:
            return False
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=0.8)
            self._ollama_online = (resp.status_code == 200)
        except Exception:
            self._ollama_online = False
        self._last_check = now
        return bool(self._ollama_online)

    def _fallback_vector(self, text: str) -> List[float]:
        """Deterministic hashing-based normalized pseudo-embedding when Ollama is unavailable."""
        norm_text = text.lower().strip()
        tokens = norm_text.split()
        vec = [0.0] * self.dim

        for i, token in enumerate(tokens):
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            val = ((h >> 8) % 1000) / 1000.0 - 0.5
            vec[idx] += val * (1.0 / (1.0 + 0.1 * i))

        # Also encode character trigrams for subword robustness
        for i in range(len(norm_text) - 2):
            trigram = norm_text[i:i+3]
            h = int(hashlib.md5(trigram.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 0.2

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            return [round(x / norm, 6) for x in vec]
        return [0.0] * self.dim

    def embed_text(self, text: str) -> List[float]:
        """Embed a single piece of text."""
        if not text.strip():
            return [0.0] * self.dim

        if not self._is_ollama_available():
            return self._fallback_vector(text)

        try:
            url = f"{self.base_url}/api/embeddings"
            resp = requests.post(
                url,
                json={"model": self.model, "prompt": text},
                timeout=2.0
            )
            if resp.status_code == 200:
                data = resp.json()
                emb = data.get("embedding")
                if emb and isinstance(emb, list) and len(emb) > 0:
                    return emb
        except Exception:
            pass

        # Try alternative Ollama /api/embed endpoint (newer Ollama versions)
        try:
            url = f"{self.base_url}/api/embed"
            resp = requests.post(
                url,
                json={"model": self.model, "input": text},
                timeout=5.0
            )
            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("embeddings")
                if embeddings and isinstance(embeddings, list) and len(embeddings) > 0:
                    self._ollama_online = True
                    return embeddings[0]
        except Exception:
            pass

        # Use deterministic fallback
        return self._fallback_vector(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of documents."""
        return [self.embed_text(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Compute embedding for user search query."""
        return self.embed_text(query)


ollama_embeddings = OllamaEmbeddings()
