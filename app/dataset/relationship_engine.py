"""Dataset relationship inference engine: candidate keys, hierarchies, and entity comparisons."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from app.utils.logger import logger


class RelationshipEngine:
    """Discovers structural relationships, primary keys, categorical dimensions, and entity correlations."""

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive relationship and structural inference across all columns."""
        if df.empty:
            return {
                "candidate_keys": [],
                "dimensions": [],
                "metrics": [],
                "date_columns": [],
                "numeric_correlations": [],
                "summary": "Dataset is empty.",
            }

        total_rows = len(df)
        candidate_keys = []
        dimensions = []
        metrics = []
        date_columns = []

        for col in df.columns:
            series = df[col]
            null_count = int(series.isnull().sum())
            distinct_count = series.nunique()

            # 1. Primary key detection (100% unique, zero nulls)
            if null_count == 0 and distinct_count == total_rows:
                candidate_keys.append(col)

            # 2. Check if column is numeric metric
            num_test = pd.to_numeric(series.astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce").dropna()
            if len(num_test) >= total_rows * 0.7:
                metrics.append({
                    "column": col,
                    "min": float(num_test.min()),
                    "max": float(num_test.max()),
                    "mean": round(float(num_test.mean()), 2),
                    "median": round(float(num_test.median()), 2),
                })
                continue

            # 3. Check if column is date
            if any(term in col.lower() for term in ["date", "time", "day", "expiry", "joining"]):
                date_columns.append(col)
                continue

            # 4. Check if column is categorical dimension (low cardinality)
            if 1 < distinct_count <= min(50, total_rows * 0.8):
                top_vals = series.value_counts().head(5).to_dict()
                dimensions.append({
                    "column": col,
                    "cardinality": distinct_count,
                    "top_values": {str(k): int(v) for k, v in top_vals.items()}
                })

        # 5. Compute numeric correlations if 2 or more metrics exist
        correlations = []
        metric_cols = [m["column"] for m in metrics]
        if len(metric_cols) >= 2:
            num_sub_df = pd.DataFrame()
            for mc in metric_cols:
                num_sub_df[mc] = pd.to_numeric(df[mc].astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce")
            corr_matrix = num_sub_df.corr()
            for i, col1 in enumerate(metric_cols):
                for j, col2 in enumerate(metric_cols):
                    if i < j:
                        val = corr_matrix.loc[col1, col2]
                        if not np.isnan(val) and abs(val) >= 0.2:
                            correlations.append({
                                "col1": col1,
                                "col2": col2,
                                "correlation": round(float(val), 2),
                                "strength": "Strong" if abs(val) >= 0.7 else "Moderate"
                            })

        return {
            "candidate_keys": candidate_keys,
            "dimensions": dimensions,
            "metrics": metrics,
            "date_columns": date_columns,
            "numeric_correlations": correlations,
            "row_count": total_rows,
            "column_count": len(df.columns)
        }

    def compare_entities(self, df: pd.DataFrame, entity_col: str, val1: str, val2: str) -> Dict[str, Any]:
        """Compare two rows/entities side-by-side across all common columns."""
        if entity_col not in df.columns:
            return {"status": "error", "message": f"Column '{entity_col}' not found."}

        s1 = df[df[entity_col].astype(str).str.strip().str.lower() == val1.strip().lower()]
        s2 = df[df[entity_col].astype(str).str.strip().str.lower() == val2.strip().lower()]

        if s1.empty or s2.empty:
            return {"status": "not_found", "message": f"Could not find both '{val1}' and '{val2}'."}

        row1 = s1.iloc[0].to_dict()
        row2 = s2.iloc[0].to_dict()

        comparison = []
        for col in df.columns:
            v1 = row1[col]
            v2 = row2[col]
            diff = (v1 != v2)
            comparison.append({
                "attribute": col,
                f"entity_1 ({val1})": v1,
                f"entity_2 ({val2})": v2,
                "different": diff
            })

        return {
            "status": "success",
            "entity_1": val1,
            "entity_2": val2,
            "comparison": comparison
        }

    def find_peers(self, df: pd.DataFrame, entity_col: str, entity_val: str, group_col: str) -> List[Dict[str, Any]]:
        """Find peers that share the same dimension value as the specified entity."""
        target_row = df[df[entity_col].astype(str).str.strip().str.lower() == entity_val.strip().lower()]
        if target_row.empty or group_col not in df.columns:
            return []

        shared_val = target_row.iloc[0][group_col]
        peers = df[
            (df[group_col] == shared_val) &
            (df[entity_col].astype(str).str.strip().str.lower() != entity_val.strip().lower())
        ]
        return peers.head(10).to_dict(orient="records")


relationship_engine = RelationshipEngine()

