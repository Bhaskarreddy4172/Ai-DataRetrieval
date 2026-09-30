"""Automated dataset inspection, statistical profiling, semantic tagging, and version fingerprinting."""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from app.utils.logger import logger


class DatasetProfiler:
    """Extracts column types, statistics, null distributions, value histograms, and version fingerprints."""

    def __init__(self):
        self._version_registry: Dict[str, int] = {}

    @staticmethod
    def format_size(size_bytes: int) -> str:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes / (1024 * 1024):.2f} MB"

    @staticmethod
    def compute_dataset_id(df: pd.DataFrame, dataset_name: str) -> str:
        """Calculate deterministic SHA-256 fingerprint for dataset content and structure."""
        cols = sorted([str(c) for c in df.columns if c != "_internal_row_id"])
        data_sample = f"{dataset_name}:{len(df)}:{'-'.join(cols)}"
        if not df.empty:
            data_sample += f":{str(df.iloc[0].values)}:{str(df.iloc[-1].values)}"
        return hashlib.sha256(data_sample.encode("utf-8", errors="ignore")).hexdigest()[:16]

    @staticmethod
    def compute_schema_hash(columns_profile: List[Dict[str, Any]]) -> str:
        """Calculate deterministic SHA-256 hash of schema structure and inferred types."""
        schema_repr = "|".join(f"{c['name']}:{c['type']}" for c in columns_profile)
        return hashlib.sha256(schema_repr.encode("utf-8")).hexdigest()[:16]

    def profile(self, df: pd.DataFrame, dataset_name: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive statistical profile of a DataFrame with versioning."""
        file_size_bytes = 0
        file_type = "DataFrame"
        if file_path and file_path.exists():
            file_size_bytes = os.path.getsize(file_path)
            file_type = file_path.suffix.lower().replace(".", "").upper()

        cols = [c for c in df.columns if c != "_internal_row_id"]
        row_count = len(df)
        col_count = len(cols)

        # Version tracking
        self._version_registry[dataset_name] = self._version_registry.get(dataset_name, 0) + 1
        dataset_version = self._version_registry[dataset_name]

        dataset_id = self.compute_dataset_id(df, dataset_name)

        columns_profile: List[Dict[str, Any]] = []
        type_counts = {"text": 0, "numeric": 0, "date": 0, "boolean": 0}
        total_missing = 0

        for col in cols:
            series = df[col]
            null_count = int(series.isnull().sum())
            total_missing += null_count
            null_pct = round((null_count / row_count * 100), 1) if row_count > 0 else 0.0

            # Detect data type & semantic tag
            inferred_type = "text"
            semantic_tag = "text"
            stats: Dict[str, Any] = {}
            col_l = col.lower()

            if pd.api.types.is_bool_dtype(series):
                inferred_type = "boolean"
                semantic_tag = "boolean"
                type_counts["boolean"] += 1
            elif pd.api.types.is_numeric_dtype(series):
                inferred_type = "numeric"
                semantic_tag = "metric"
                if any(id_w in col_l for id_w in ["id", "code", "zip", "pin"]):
                    semantic_tag = "identifier"
                type_counts["numeric"] += 1
                non_null = series.dropna()
                if not non_null.empty:
                    stats = {
                        "min": round(float(non_null.min()), 2),
                        "max": round(float(non_null.max()), 2),
                        "average": round(float(non_null.mean()), 2),
                        "mean": round(float(non_null.mean()), 2),
                        "median": round(float(non_null.median()), 2),
                        "sum": round(float(non_null.sum()), 2),
                    }
            elif pd.api.types.is_datetime64_any_dtype(series):
                inferred_type = "date"
                semantic_tag = "temporal"
                type_counts["date"] += 1
                non_null = series.dropna()
                if not non_null.empty:
                    stats = {
                        "earliest": str(non_null.min())[:10],
                        "latest": str(non_null.max())[:10],
                    }
            else:
                # Check if text column might be dates
                sample_non_null = series.dropna().astype(str).head(20)
                is_date_col = False
                if not sample_non_null.empty and any(k in col_l for k in ["date", "expiry", "time", "joined", "created"]):
                    try:
                        parsed = pd.to_datetime(sample_non_null, errors="coerce")
                        if parsed.notnull().sum() > len(sample_non_null) * 0.7:
                            inferred_type = "date"
                            semantic_tag = "temporal"
                            type_counts["date"] += 1
                            is_date_col = True
                            non_null_dt = parsed.dropna()
                            if not non_null_dt.empty:
                                stats = {
                                    "earliest": str(non_null_dt.min())[:10],
                                    "latest": str(non_null_dt.max())[:10],
                                }
                    except Exception:
                        pass

                if not is_date_col:
                    inferred_type = "text"
                    type_counts["text"] += 1
                    # Text statistics
                    str_lens = series.dropna().astype(str).str.len()
                    if not str_lens.empty:
                        stats = {
                            "min_length": int(str_lens.min()),
                            "max_length": int(str_lens.max()),
                        }
                    # Semantic tag heuristics
                    if any(loc_w in col_l for loc_w in ["city", "state", "country", "region", "location", "address"]):
                        semantic_tag = "location"
                    elif any(name_w in col_l for name_w in ["name", "person", "employee", "customer", "manager"]):
                        semantic_tag = "entity_name"
                    elif any(id_w in col_l for id_w in ["id", "code", "key", "number"]):
                        semantic_tag = "identifier"
                    else:
                        semantic_tag = "categorical"

            # Unique values and sample values
            unique_vals = series.dropna().unique()
            unique_count = len(unique_vals)
            sample_vals = [str(v) for v in unique_vals[:5]]

            # Determine cardinality classification
            if unique_count == row_count and row_count > 0:
                cardinality = "UNIQUE"
            elif row_count > 0 and unique_count > row_count * 0.5:
                cardinality = "HIGH"
            elif unique_count <= 10:
                cardinality = "LOW"
            else:
                cardinality = "MEDIUM"

            # Possible unit detection
            possible_unit = None
            if any(u in col_l for u in ["salary", "price", "cost", "revenue", "profit", "amount", "budget", "wage", "ctc"]):
                possible_unit = "currency"
            elif any(u in col_l for u in ["percent", "pct", "rate", "%"]):
                possible_unit = "percentage"
            elif any(u in col_l for u in ["age", "tenure", "experience", "yoe"]):
                possible_unit = "years"
            elif any(u in col_l for u in ["count", "quantity", "qty", "units", "volume"]):
                possible_unit = "count"

            # Possible entity type detection
            possible_entity_type = "attribute"
            if any(k in col_l for k in ["name", "emp", "employee", "person", "candidate", "customer", "manager"]):
                possible_entity_type = "person"
            elif any(k in col_l for k in ["company", "org", "department", "dept", "vendor"]):
                possible_entity_type = "organization"
            elif any(k in col_l for k in ["city", "state", "country", "location", "address"]):
                possible_entity_type = "location"
            elif any(k in col_l for k in ["product", "item", "sku"]):
                possible_entity_type = "product"
            elif inferred_type == "date":
                possible_entity_type = "date"
            elif any(k in col_l for k in ["id", "code", "key"]):
                possible_entity_type = "code"

            # Top value histogram (frequency distribution)
            val_counts = series.value_counts().head(10)
            histogram = [
                {
                    "value": str(val),
                    "count": int(count),
                    "percentage": round(float(count) / row_count * 100, 1) if row_count > 0 else 0.0
                }
                for val, count in val_counts.items()
            ]

            norm_name = "".join(c if c.isalnum() else "_" for c in col_l).strip("_")

            columns_profile.append({
                "name": str(col),
                "column_name": str(col),
                "original_name": str(col),
                "normalized_name": norm_name,
                "type": inferred_type,
                "data_type": inferred_type,
                "semantic_tag": semantic_tag,
                "semantic_type": semantic_tag,
                "null_count": null_count,
                "null_percentage": null_pct,
                "unique_count": unique_count,
                "unique_values": unique_count,
                "cardinality": cardinality,
                "sample_values": sample_vals,
                "stats": stats,
                "min": stats.get("min"),
                "max": stats.get("max"),
                "mean": stats.get("mean"),
                "median": stats.get("median"),
                "min_length": stats.get("min_length"),
                "max_length": stats.get("max_length"),
                "earliest": stats.get("earliest"),
                "latest": stats.get("latest"),
                "possible_unit": possible_unit,
                "possible_entity_type": possible_entity_type,
                "possible_relationships": [],
                "histogram": histogram
            })

        schema_hash = self.compute_schema_hash(columns_profile)

        return {
            "basic_info": {
                "dataset_name": dataset_name,
                "dataset_id": dataset_id,
                "dataset_version": dataset_version,
                "schema_hash": schema_hash,
                "row_count": row_count,
                "col_count": col_count,
                "file_type": file_type,
                "file_size": self.format_size(file_size_bytes),
                "total_cells": row_count * col_count,
                "total_missing_values": total_missing,
            },
            "fingerprint": {
                "dataset_id": dataset_id,
                "dataset_version": dataset_version,
                "schema_hash": schema_hash,
            },
            "type_breakdown": type_counts,
            "columns": columns_profile,
        }


dataset_profiler = DatasetProfiler()
