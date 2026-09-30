"""Thread-safe Query Cache for deterministic query execution results.

Stores cached results keyed by (dataset_hash, normalized_query, query_plan_hash).
Provides automatic thread safety, bounded capacity (LRU), and instant invalidation.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class QueryCache:
    """Thread-safe LRU cache for query results."""

    def __init__(self, maxsize: int = 1000, default_ttl_seconds: int = 3600):
        self._maxsize = maxsize
        self._default_ttl = default_ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def _make_key(
        self,
        dataset_hash: str,
        normalized_query: str,
        query_plan: Optional[Any] = None,
        dataset_id: Optional[str] = None,
        dataset_version: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Construct a deterministic cache key from dataset hash, query, plan, version, and parameters."""
        plan_str = ""
        if query_plan is not None:
            if isinstance(query_plan, (dict, list)):
                try:
                    plan_str = json.dumps(query_plan, sort_keys=True, default=str)
                except Exception:
                    plan_str = str(query_plan)
            else:
                plan_str = str(query_plan)

        params_str = ""
        if parameters:
            try:
                params_str = json.dumps(parameters, sort_keys=True, default=str)
            except Exception:
                params_str = str(parameters)

        raw_key = f"{dataset_id or ''}::{dataset_version or ''}::{dataset_hash}::{normalized_query.strip().lower()}::{plan_str}::{params_str}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(
        self,
        dataset_hash: str,
        normalized_query: str,
        query_plan: Optional[Any] = None,
        dataset_id: Optional[str] = None,
        dataset_version: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Retrieve a cached query result if valid and not expired."""
        key = self._make_key(dataset_hash, normalized_query, query_plan, dataset_id, dataset_version, parameters)
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            # Check TTL
            if entry["expires_at"] is not None and time.time() > entry["expires_at"]:
                del self._cache[key]
                return None
            
            # Move to end for LRU
            self._cache.move_to_end(key)
            return entry["result"]

    def set(
        self,
        dataset_hash: str,
        normalized_query: str,
        result: Any,
        query_plan: Optional[Any] = None,
        ttl_seconds: Optional[int] = None,
        dataset_id: Optional[str] = None,
        dataset_version: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a query result in cache."""
        key = self._make_key(dataset_hash, normalized_query, query_plan, dataset_id, dataset_version, parameters)
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expires_at = time.time() + ttl if ttl > 0 else None

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = {
                "dataset_hash": dataset_hash,
                "result": result,
                "expires_at": expires_at,
            }
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def invalidate(self, dataset_hash: Optional[str] = None) -> int:
        """Invalidate cache entries. If dataset_hash is provided, invalidate only that dataset's entries."""
        with self._lock:
            if dataset_hash is None:
                count = len(self._cache)
                self._cache.clear()
                return count
            
            keys_to_del = [
                k for k, v in self._cache.items() if v.get("dataset_hash") == dataset_hash
            ]
            for k in keys_to_del:
                del self._cache[k]
            return len(keys_to_del)

    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# Global singleton cache
query_cache = QueryCache()

