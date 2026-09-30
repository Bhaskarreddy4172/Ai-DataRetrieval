"""Deep schema intelligence, semantic type classification, and dynamic column alias generation."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from app.utils.logger import logger

# Semantic category rules
ALIAS_DICTIONARY: Dict[str, List[str]] = {
    "salary": ["salary", "salaries", "pay", "income", "earnings", "compensation", "wage", "package", "money", "stipend", "making", "makes", "earns", "earning", "paid", "highest paid", "lowest paid", "earner", "earners"],
    "amount": ["amount", "cost", "price", "spending", "expense", "bill", "charge", "fee", "total cost"],
    "price": ["price", "cost", "rate", "mrp", "retail price", "tag"],
    "stock": ["stock", "quantity", "inventory", "available units", "stock count", "in stock", "qty"],
    "city": ["city", "cities", "location", "locations", "place", "places", "branch", "where", "office", "based in", "works at", "working from", "lives in", "living in", "residing in", "hometown", "town"],
    "state": ["state", "states", "province", "region", "territory"],
    "capital": ["capital", "capitals", "capital city", "headquarters"],
    "category": ["category", "categories", "type", "types", "genre", "class", "segment"],
    "department": ["department", "departments", "dept", "depts", "team", "teams", "division", "section", "group", "unit", "function"],
    "designation": ["designation", "role", "title", "job title", "position", "post"],
    "employee": ["employee", "employees", "staff", "worker", "workers", "person", "people", "team member", "who", "name", "individual"],
    "customer": ["customer", "customers", "client", "clients", "buyer", "consumer", "patron", "who", "name"],
    "product": ["product", "products", "item", "items", "gadget", "gadgets", "device", "devices", "unit", "model", "name"],
    "vehicle": ["vehicle", "vehicles", "car", "cars", "bike", "scooter", "automobile", "model"],
    "rating": ["rating", "ratings", "score", "scores", "stars", "review", "satisfaction", "performance", "performer", "best performer", "performance score", "perf"],
    "status": ["status", "state", "condition", "stage"],
    "date": ["date", "dates", "time", "day", "when", "timestamp", "joined", "joining date", "date joined", "start date", "started", "when joined", "recently", "earliest", "service date", "expiry"]
}


class SchemaIntelligence:
    """Extracts semantic types, distributions, and dynamic alias maps from active DataFrame."""

    def analyze_column(self, col: str, series: pd.Series) -> Dict[str, Any]:
        col_lower = col.lower()
        total_len = len(series)
        null_count = int(series.isnull().sum())
        nunique = series.nunique()
        samples = [str(x) for x in series.dropna().unique()[:6]]

        # 1. Detect Numeric
        num_clean = pd.to_numeric(series.astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce").dropna()
        is_numeric = len(num_clean) >= total_len * 0.65

        # 2. Determine semantic type
        semantic_type = "text_general"
        if is_numeric:
            if any(term in col_lower for term in ["salary", "amount", "price", "cost", "revenue", "spending", "fee"]):
                semantic_type = "currency/measure"
            elif any(term in col_lower for term in ["stock", "quantity", "count", "qty"]):
                semantic_type = "count/quantity"
            elif any(term in col_lower for term in ["rating", "score", "perf"]):
                semantic_type = "numeric_score"
            else:
                semantic_type = "numeric_measure"
        elif any(term in col_lower for term in ["date", "time", "day", "expiry", "joining"]):
            semantic_type = "temporal/date"
        elif null_count == 0 and nunique == total_len and any(term in col_lower for term in ["id", "code", "no"]):
            semantic_type = "identifier"
        elif any(term in col_lower for term in ["city", "location", "branch", "state", "region", "address"]):
            semantic_type = "geographic"
        elif any(term in col_lower for term in ["employee", "customer", "person", "patient", "name", "client"]):
            semantic_type = "entity/person"
        elif nunique <= min(40, max(2, int(total_len * 0.7))):
            semantic_type = "categorical/dimension"

        # 3. Dynamic Alias Generation
        aliases = set()
        aliases.add(col_lower)
        for w in col_lower.replace("_", " ").replace("-", " ").split():
            if len(w) >= 3:
                aliases.add(w)

        # Match against predefined alias dictionary
        for key, syns in ALIAS_DICTIONARY.items():
            if key in col_lower or any(s in col_lower for s in syns):
                for syn in syns:
                    aliases.add(syn)

        stats: Dict[str, Any] = {}
        if is_numeric and not num_clean.empty:
            stats = {
                "min": round(float(num_clean.min()), 2),
                "max": round(float(num_clean.max()), 2),
                "mean": round(float(num_clean.mean()), 2),
                "median": round(float(num_clean.median()), 2),
                "std": round(float(num_clean.std()), 2) if len(num_clean) > 1 else 0.0
            }

        return {
            "name": col,
            "semantic_type": semantic_type,
            "is_numeric": is_numeric,
            "cardinality": nunique,
            "null_count": null_count,
            "sample_values": samples,
            "aliases": sorted(list(aliases)),
            "stats": stats
        }

    def build_schema(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Build full schema intelligence dictionary for all columns."""
        if df.empty:
            return {"dataset_name": dataset_name, "columns": {}, "column_names": []}

        cols_info = {}
        for col in df.columns:
            if col == "_internal_row_id":
                continue
            cols_info[col] = self.analyze_column(col, df[col])

        user_cols = [c for c in df.columns if c != "_internal_row_id"]
        return {
            "dataset_name": dataset_name,
            "row_count": len(df),
            "column_count": len(user_cols),
            "columns": cols_info,
            "column_names": user_cols
        }


schema_intelligence = SchemaIntelligence()

