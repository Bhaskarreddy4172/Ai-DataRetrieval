"""Deterministic Python query execution engine operating directly on pandas DataFrame."""

import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from app.query.operations import (
    execute_aggregation,
    execute_distinct,
    execute_duplicates,
    execute_group,
    execute_cross_column_search,
)
from app.query.schema import StructuredQuery
from app.utils.logger import logger


class QueryEngine:
    """Executes safe StructuredQuery operations against a DataFrame with zero raw code execution."""

    def execute(self, query: StructuredQuery, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute structured query on dataframe using unified query_executor."""
        from app.query.executor import query_executor
        return query_executor.execute(query, df)

        if query.operation == "UNKNOWN":
            return {
                "operation": "UNKNOWN",
                "results": [],
                "aggregation": None,
                "result_count": 0,
                "columns": [],
                "status": "unknown_query",
                "missing_column": query.missing_column,
            }

        # Handle DUPLICATES
        if query.operation == "DUPLICATES":
            dup_res = execute_duplicates(df, query.target_column)
            clean_records = self._clean_records(dup_res["records"])
            return {
                "operation": "DUPLICATES",
                "results": clean_records,
                "aggregation": {"metric": "DUPLICATES", "value": dup_res["duplicate_count"]},
                "result_count": dup_res["duplicate_count"],
                "columns": list(df.columns),
                "status": "success"
            }

        # Handle DISTINCT
        if query.operation == "DISTINCT" and query.target_column:
            dist_res = execute_distinct(df, query.target_column)
            records = [{query.target_column: val} for val in dist_res["values"]]
            return {
                "operation": "DISTINCT",
                "results": records,
                "aggregation": {"metric": "DISTINCT", "column": query.target_column, "value": dist_res["total_distinct"]},
                "result_count": dist_res["total_distinct"],
                "columns": [query.target_column],
                "status": "success"
            }

        filtered_df = df.copy()

        # Apply filtering conditions if present
        if query.conditions:
            masks = []
            for cond in query.conditions:
                mask = self._build_condition_mask(filtered_df, cond.column, cond.operator, cond.value)
                masks.append(mask)

            if masks:
                if query.logical_operator.upper() == "OR":
                    combined = masks[0]
                    for m in masks[1:]:
                        combined = combined | m
                else:
                    combined = masks[0]
                    for m in masks[1:]:
                        combined = combined & m
                filtered_df = filtered_df[combined]

        # Handle EXISTS
        if query.operation == "EXISTS":
            exists = len(filtered_df) > 0
            records = self._clean_records(filtered_df.head(query.limit).to_dict(orient="records"))
            return {
                "operation": "EXISTS",
                "results": records,
                "aggregation": {"metric": "EXISTS", "value": exists, "count": len(filtered_df)},
                "result_count": len(filtered_df),
                "columns": list(filtered_df.columns),
                "status": "success"
            }

        # Handle Aggregation operations (COUNT, SUM, AVERAGE, MIN, MAX, MEDIAN)
        aggregation_result: Optional[Dict[str, Any]] = None
        if query.operation in {"COUNT", "SUM", "AVERAGE", "AVG", "MIN", "MAX", "MEDIAN"}:
            aggregation_result = execute_aggregation(filtered_df, query.operation, query.target_column)

        # Handle GROUP
        elif query.operation == "GROUP" and query.group_by_column:
            filtered_df = execute_group(filtered_df, query.group_by_column, "COUNT", query.target_column)

        # Apply sorting if specified
        if query.sort_column and query.sort_column in filtered_df.columns:
            ascending = (query.sort_order.upper() == "ASC")
            numeric_test = pd.to_numeric(filtered_df[query.sort_column].astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce")
            if numeric_test.notnull().sum() > len(filtered_df) * 0.7:
                filtered_df["_sort_tmp"] = numeric_test
                filtered_df = filtered_df.sort_values(by="_sort_tmp", ascending=ascending).drop(columns=["_sort_tmp"])
            else:
                filtered_df = filtered_df.sort_values(by=query.sort_column, ascending=ascending)

        # Apply column selection if specified
        result_cols = list(filtered_df.columns)
        if query.select_columns:
            valid_select = [c for c in query.select_columns if c in filtered_df.columns]
            if valid_select:
                filtered_df = filtered_df[valid_select]
                result_cols = valid_select

        total_matches = len(filtered_df)
        bounded_df = filtered_df.head(query.limit)
        records = bounded_df.to_dict(orient="records")
        clean_records = self._clean_records(records)

        return {
            "operation": query.operation,
            "results": clean_records,
            "aggregation": aggregation_result,
            "result_count": total_matches,
            "columns": result_cols,
            "status": "success",
        }

    def _clean_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure all fields in records are clean JSON-serializable primitives."""
        clean_records = []
        for r in records:
            clean_r = {}
            for k, v in r.items():
                if pd.isna(v):
                    clean_r[k] = None
                elif isinstance(v, (np.integer, int)):
                    clean_r[k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    clean_r[k] = round(float(v), 2)
                else:
                    clean_r[k] = str(v)
            clean_records.append(clean_r)
        return clean_records

    def _build_condition_mask(self, df: pd.DataFrame, column: str, operator: str, value: Any) -> pd.Series:
        """Construct safe boolean filtering mask for a condition."""
        if column not in df.columns:
            return pd.Series(False, index=df.index)

        series = df[column]

        # Check if column is numeric or value is numeric
        is_num = False
        num_val: Optional[float] = None
        if isinstance(value, (int, float)):
            is_num = True
            num_val = float(value)
        elif isinstance(value, str):
            clean_val_str = re.sub(r"[₹$,]", "", value).strip()
            if clean_val_str.lower().endswith("k"):
                try:
                    num_val = float(clean_val_str[:-1]) * 1000
                    is_num = True
                except Exception:
                    pass
            else:
                try:
                    num_val = float(clean_val_str)
                    is_num = True
                except Exception:
                    pass

        # Numeric comparison
        if is_num and num_val is not None and operator in {">", "<", ">=", "<=", "=", "!="}:
            num_series = pd.to_numeric(series.astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce")
            if operator == ">":
                return num_series > num_val
            elif operator == "<":
                return num_series < num_val
            elif operator == ">=":
                return num_series >= num_val
            elif operator == "<=":
                return num_series <= num_val
            elif operator == "=":
                return num_series == num_val
            elif operator == "!=":
                return num_series != num_val

        # Text and string comparisons
        str_series = series.astype(str).str.strip().str.lower()
        str_val = str(value).strip().lower()

        if operator in {"=", "=="}:
            return str_series == str_val
        elif operator in {"!=", "<>"}:
            return str_series != str_val
        elif operator == "contains":
            return str_series.str.contains(re.escape(str_val), na=False, regex=True)
        elif operator == "not_contains":
            return ~str_series.str.contains(re.escape(str_val), na=False, regex=True)
        elif operator == "starts_with":
            return str_series.str.startswith(str_val)
        elif operator == "ends_with":
            return str_series.str.endswith(str_val)
        elif operator == "in" and isinstance(value, list):
            val_list = [str(v).lower().strip() for v in value]
            return str_series.isin(val_list)

        return str_series == str_val


query_engine = QueryEngine()
