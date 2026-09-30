"""Schema metadata generator and semantic column alias resolution."""

import re
from typing import Any, Dict, List, Optional


class DatasetMetadata:
    """Generates compact LLM-friendly schemas and semantic aliases for any dataset."""

    SYNONYM_MAP = {
        "amount": ["cost", "price", "spending", "spent", "revenue", "bill", "charge", "paid", "fee", "money", "service cost"],
        "customer": ["client", "user", "buyer", "person", "name", "who", "account", "customer name"],
        "city": ["location", "place", "town", "branch", "region", "where"],
        "vehicle": ["car", "automobile", "model", "motor", "bike"],
        "date": ["time", "when", "day", "timestamp", "period", "service date"],
        "status": ["condition", "state", "progress", "stage", "service status"],
        "state": ["province", "region", "territory"],
        "capital": ["capital city", "headquarter", "main city"],
        "warranty": ["guarantee", "warranty expiry", "coverage"],
    }

    @staticmethod
    def get_compact_schema(columns_profile: List[Dict[str, Any]], dataset_name: str, row_count: int) -> Dict[str, Any]:
        cols = []
        for c in columns_profile:
            col_info = {
                "name": c["name"],
                "type": c["type"],
                "samples": c["sample_values"][:4],
            }
            if c.get("stats"):
                if "min" in c["stats"]:
                    col_info["min"] = c["stats"]["min"]
                    col_info["max"] = c["stats"]["max"]
                if "earliest" in c["stats"]:
                    col_info["range"] = f"{c['stats']['earliest']} to {c['stats']['latest']}"
            cols.append(col_info)

        return {
            "dataset_name": dataset_name,
            "row_count": row_count,
            "columns": cols,
        }

    @classmethod
    def match_column(cls, candidate: str, available_columns: List[str]) -> Optional[str]:
        """Match candidate column name to actual dataset column using exact, token, and semantic matching."""
        if not candidate:
            return None

        clean_candidate = candidate.strip().lower()

        # 1. Exact match (case-insensitive)
        for col in available_columns:
            if col.strip().lower() == clean_candidate:
                return col

        # 2. Normalized whitespace & underscore match
        norm_candidate = re.sub(r"[\s_-]+", "", clean_candidate)
        for col in available_columns:
            if re.sub(r"[\s_-]+", "", col.lower()) == norm_candidate:
                return col

        # 3. Substring match
        for col in available_columns:
            if clean_candidate in col.lower() or col.lower() in clean_candidate:
                return col

        # 4. Token & Synonym match
        candidate_words = set(re.findall(r"\b[a-zA-Z]+\b", clean_candidate))
        for col in available_columns:
            col_lower = col.lower()
            col_words = set(re.findall(r"\b[a-zA-Z]+\b", col_lower))

            # Check overlap
            if candidate_words & col_words:
                return col

            # Check semantic map
            for syn_key, synonyms in cls.SYNONYM_MAP.items():
                all_syns = set([syn_key] + synonyms)
                if (candidate_words & all_syns or clean_candidate in all_syns) and (col_words & all_syns or any(s in col_lower for s in all_syns)):
                    return col

        return None
