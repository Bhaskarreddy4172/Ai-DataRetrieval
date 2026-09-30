"""Statistical operations and deterministic outlier detection (Sections 24 & 25)."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class DatasetAnalytics:
    """Computes rigorous deterministic statistical measures and outlier distributions."""

    @staticmethod
    def calculate_statistics(series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Compute full statistical summary on numeric series."""
        clean_s = pd.to_numeric(
            series.astype(str).str.replace(r"[\u20b9$,]", "", regex=True),
            errors="coerce"
        ).dropna()

        if clean_s.empty:
            return {"column": column_name, "error": "No valid numeric data"}

        arr = clean_s.values
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        iqr = float(q75 - q25)

        return {
            "column": column_name,
            "count": int(len(arr)),
            "mean": round(float(np.mean(arr)), 2),
            "median": round(float(q50), 2),
            "mode": round(float(clean_s.mode().iloc[0]), 2) if not clean_s.mode().empty else None,
            "std": round(float(np.std(arr, ddof=1)), 2) if len(arr) > 1 else 0.0,
            "variance": round(float(np.var(arr, ddof=1)), 2) if len(arr) > 1 else 0.0,
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
            "range": round(float(np.max(arr) - np.min(arr)), 2),
            "q25": round(float(q25), 2),
            "q50": round(float(q50), 2),
            "q75": round(float(q75), 2),
            "iqr": round(iqr, 2)
        }

    @staticmethod
    def detect_outliers_iqr(
        df: pd.DataFrame,
        column_name: str,
        k: float = 1.5
    ) -> Dict[str, Any]:
        """Identify outliers using Tukey's Interquartile Range (IQR) method (Section 25)."""
        if column_name not in df.columns:
            return {"error": f"Column {column_name} not found"}

        clean_s = pd.to_numeric(
            df[column_name].astype(str).str.replace(r"[\u20b9$,]", "", regex=True),
            errors="coerce"
        )
        valid = clean_s.dropna()
        if len(valid) < 4:
            return {"outliers": [], "count": 0, "method": "IQR"}

        q25 = float(np.percentile(valid.values, 25))
        q75 = float(np.percentile(valid.values, 75))
        iqr = q75 - q25
        lower_bound = q25 - (k * iqr)
        upper_bound = q75 + (k * iqr)

        mask = (clean_s < lower_bound) | (clean_s > upper_bound)
        outlier_df = df[mask]

        records = outlier_df.to_dict(orient="records")
        return {
            "method": "IQR",
            "k": k,
            "column": column_name,
            "q25": round(q25, 2),
            "q75": round(q75, 2),
            "iqr": round(iqr, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "outlier_count": len(records),
            "records": records
        }

    @staticmethod
    def detect_outliers_zscore(
        df: pd.DataFrame,
        column_name: str,
        threshold: float = 2.5
    ) -> Dict[str, Any]:
        """Identify outliers using standard z-score threshold (Section 25)."""
        if column_name not in df.columns:
            return {"error": f"Column {column_name} not found"}

        clean_s = pd.to_numeric(
            df[column_name].astype(str).str.replace(r"[\u20b9$,]", "", regex=True),
            errors="coerce"
        )
        valid = clean_s.dropna()
        if len(valid) < 3:
            return {"outliers": [], "count": 0, "method": "z-score"}

        mean = float(np.mean(valid.values))
        std = float(np.std(valid.values, ddof=1))
        if std == 0:
            return {"outliers": [], "count": 0, "method": "z-score"}

        z_scores = (clean_s - mean).abs() / std
        mask = z_scores > threshold
        outlier_df = df[mask]

        records = outlier_df.to_dict(orient="records")
        return {
            "method": "z-score",
            "threshold": threshold,
            "column": column_name,
            "mean": round(mean, 2),
            "std": round(std, 2),
            "outlier_count": len(records),
            "records": records
        }

    @staticmethod
    def generate_chart_spec(
        operation: str,
        results: List[Dict[str, Any]],
        aggregation: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Declarative chart recommendation: KPI card, Bar chart, Line chart, or Table."""
        if aggregation and not results:
            return {
                "type": "kpi",
                "title": f"{aggregation.get('metric', 'Metric')} {aggregation.get('column', '')}".strip(),
                "metric_label": aggregation.get("metric"),
                "value": aggregation.get("value"),
                "data": [aggregation],
            }

        if not results:
            return None

        if len(results) == 1 and aggregation:
            return {
                "type": "kpi",
                "title": f"{aggregation.get('metric', 'Result')}",
                "metric_label": aggregation.get("metric"),
                "value": aggregation.get("value"),
                "data": results,
            }

        sample = results[0]
        keys = [k for k in sample.keys() if k != "_internal_row_id"]

        date_col = next((k for k in keys if any(d in k.lower() for d in ["date", "year", "month", "day", "time"])), None)
        num_cols = [k for k in keys if isinstance(sample.get(k), (int, float))]
        cat_cols = [k for k in keys if isinstance(sample.get(k), str)]

        if date_col and num_cols:
            return {
                "type": "line",
                "title": f"{num_cols[0]} over {date_col}",
                "x_axis": date_col,
                "y_axis": num_cols[0],
                "data": results[:30],
            }

        if (operation in {"GROUP", "GROUP_EXTREME", "TOP_N", "BOTTOM_N"} or len(results) <= 15) and num_cols and cat_cols:
            return {
                "type": "bar",
                "title": f"{num_cols[0]} by {cat_cols[0]}",
                "x_axis": cat_cols[0],
                "y_axis": num_cols[0],
                "data": results[:20],
            }

        if len(results) > 1 or len(keys) >= 2:
            return {
                "type": "table",
                "title": "Dataset Records",
                "columns": keys,
                "data": results[:50],
            }

        return None

    @staticmethod
    def build_evidence_payload(
        operation: str,
        results: List[Dict[str, Any]],
        columns_referenced: List[str],
        aggregation: Optional[Dict[str, Any]] = None,
        dataset_name: str = "",
    ) -> Dict[str, Any]:
        """Construct verified evidence audit payload."""
        sample_rows = results[:10] if results else []
        row_ids = [r.get("_internal_row_id") for r in sample_rows if "_internal_row_id" in r]

        formula = None
        if aggregation:
            metric = aggregation.get("metric")
            col = aggregation.get("column", "")
            val = aggregation.get("value")
            formula = f"{metric}({col}) = {val}"

        return {
            "dataset_name": dataset_name,
            "operation": operation,
            "columns_referenced": columns_referenced,
            "total_matches": len(results),
            "matched_row_ids": row_ids,
            "formula_verified": formula,
            "evidence_rows": sample_rows,
        }


dataset_analytics = DatasetAnalytics()
