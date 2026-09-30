"""Authoritative Python retrieval engine operating directly on the dataset."""

from typing import Any, Dict, List, Optional
import pandas as pd
from app.data.data_loader import data_loader, DataLoader
from app.utils.logger import logger
from app.utils.validators import sanitize_text


class QueryEngine:
    """Executes deterministic retrieval queries directly on the pandas DataFrame."""

    def __init__(self, loader: DataLoader = data_loader):
        self.loader = loader

    @property
    def df(self) -> pd.DataFrame:
        return self.loader.dataframe

    def get_capital(self, state_name: str) -> List[Dict[str, str]]:
        """Retrieve capital for a specific state (case-insensitive & whitespace-tolerant)."""
        if self.df.empty or not state_name:
            return []
        target = sanitize_text(state_name).lower()
        matched = self.df[self.df["state"].str.lower() == target]
        return matched[["state", "capital"]].to_dict(orient="records")

    def get_state(self, capital_name: str) -> List[Dict[str, str]]:
        """Retrieve state for a specific capital city (case-insensitive & whitespace-tolerant)."""
        if self.df.empty or not capital_name:
            return []
        target = sanitize_text(capital_name).lower()
        matched = self.df[self.df["capital"].str.lower() == target]
        return matched[["state", "capital"]].to_dict(orient="records")

    def list_all(self, limit: int = 100) -> List[Dict[str, str]]:
        """Return all state-capital pairs up to limit."""
        if self.df.empty:
            return []
        safe_limit = max(1, min(limit, 200))
        return self.df.head(safe_limit)[["state", "capital"]].to_dict(orient="records")

    def count(self) -> int:
        """Return total number of states in dataset."""
        return len(self.df)

    def search_state(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """Search state names containing substring."""
        if self.df.empty or not query:
            return []
        q = sanitize_text(query).lower()
        matched = self.df[self.df["state"].str.lower().str.contains(q, na=False, regex=False)]
        return matched.head(limit)[["state", "capital"]].to_dict(orient="records")

    def search_capital(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """Search capital names containing substring."""
        if self.df.empty or not query:
            return []
        q = sanitize_text(query).lower()
        matched = self.df[self.df["capital"].str.lower().str.contains(q, na=False, regex=False)]
        return matched.head(limit)[["state", "capital"]].to_dict(orient="records")

    def filter_by_prefix(self, prefix: str, column: str = "state", limit: int = 50) -> List[Dict[str, str]]:
        """Filter states or capitals starting with prefix."""
        if self.df.empty or not prefix:
            return []
        col = "capital" if column.lower() == "capital" else "state"
        p = sanitize_text(prefix).lower()
        matched = self.df[self.df[col].str.lower().str.startswith(p)]
        return matched.head(limit)[["state", "capital"]].to_dict(orient="records")

    def execute_structured_query(self, structured: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a validated structured query and return results."""
        intent = (structured.get("intent") or "UNKNOWN").upper()
        state = structured.get("state")
        capital = structured.get("capital")
        prefix = structured.get("prefix")
        limit = int(structured.get("limit") or 10)

        results: List[Dict[str, str]] = []
        count_val: Optional[int] = None

        if intent == "GET_CAPITAL":
            if state:
                results = self.get_capital(state)
                # If exact match failed, try substring
                if not results:
                    results = self.search_state(state, limit=limit)
        elif intent == "GET_STATE":
            if capital:
                results = self.get_state(capital)
                # If exact match failed, try substring
                if not results:
                    results = self.search_capital(capital, limit=limit)
        elif intent == "LIST_ALL":
            results = self.list_all(limit=limit or 100)
        elif intent == "COUNT":
            count_val = self.count()
        elif intent == "SEARCH_STATE":
            if state:
                results = self.search_state(state, limit=limit)
        elif intent == "SEARCH_CAPITAL":
            if capital:
                results = self.search_capital(capital, limit=limit)
        elif intent == "FILTER_BY_PREFIX":
            col = "capital" if capital and not state else "state"
            p = prefix or state or capital or ""
            results = self.filter_by_prefix(p, column=col, limit=limit)
        elif intent == "UNKNOWN":
            results = []
        else:
            # Fallback search across both columns if state/capital provided
            term = state or capital
            if term:
                results = self.search_state(term, limit=limit) or self.search_capital(term, limit=limit)

        logger.info(f"QueryEngine executed intent '{intent}': {len(results)} records found.")
        return {
            "intent": intent,
            "results": results,
            "count": count_val,
            "result_count": len(results) if count_val is None else count_val,
        }


query_engine = QueryEngine()
