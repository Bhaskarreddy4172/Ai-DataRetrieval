"""Deterministic dataset query executor with tie handling, comparisons, and precision."""

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


class QueryExecutor:
    """Executes validated StructuredQuery operations against a Pandas DataFrame with exact precision."""

    def execute(self, query: StructuredQuery, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute structured query on dataframe and return verified results."""
        if df.empty:
            return {
                "operation": query.operation,
                "results": [],
                "aggregation": None,
                "result_count": 0,
                "columns": [],
                "status": "empty_dataset"
            }

        if query.operation in {"UNSUPPORTED_QUERY", "UNKNOWN"}:
            return {
                "operation": "UNSUPPORTED_QUERY",
                "results": [],
                "aggregation": None,
                "result_count": 0,
                "columns": [],
                "status": "unsupported",
                "missing_column": query.missing_column or "requested information",
                "explanation": query.explanation
            }

        if query.operation == "NO_MATCH":
            return {
                "operation": "NO_MATCH",
                "results": [],
                "aggregation": None,
                "result_count": 0,
                "columns": [],
                "status": "no_match",
                "missing_entity": query.missing_entity or "requested entity",
                "explanation": query.explanation
            }

        if query.operation == "NULL_VALUE":
            return {
                "operation": "NULL_VALUE",
                "results": [],
                "aggregation": None,
                "result_count": 0,
                "columns": [],
                "status": "null_value",
                "missing_entity": query.missing_entity or "requested entity",
                "missing_column": query.target_column,
                "explanation": query.explanation
            }

        if query.operation == "CLARIFICATION":
            return {
                "operation": "CLARIFICATION",
                "results": [],
                "aggregation": None,
                "result_count": 0,
                "columns": [],
                "status": "clarification",
                "explanation": query.explanation
            }

        # Handle DUPLICATES
        if query.operation == "DUPLICATES":
            dup_res = execute_duplicates(df, query.target_column)
            return {
                "operation": "DUPLICATES",
                "results": self._clean_records(dup_res["records"]),
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

        # Handle OUTLIER
        if query.operation == "OUTLIER":
            from app.query.analytics import dataset_analytics
            target = query.target_column
            if not target:
                num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                num_cols = [c for c in num_cols if not c.startswith("_")]
                target = num_cols[0] if num_cols else None
            if target and target in df.columns:
                out_res = dataset_analytics.detect_outliers_iqr(df, target)
                records = out_res.get("records", [])
                return {
                    "operation": "OUTLIER",
                    "results": self._clean_records(records),
                    "aggregation": {"metric": "OUTLIERS", "column": target, "value": len(records), "iqr": out_res.get("iqr")},
                    "result_count": len(records),
                    "columns": list(df.columns),
                    "status": "success"
                }

        # Handle MISSING_DATA
        if query.operation == "MISSING_DATA":
            target_col = query.target_column
            if target_col and target_col in df.columns:
                missing_df = df[df[target_col].isnull() | (df[target_col].astype(str).str.strip() == "")]
            else:
                missing_df = df[df.isnull().any(axis=1)]
            return {
                "operation": "MISSING_DATA",
                "results": self._clean_records(missing_df.head(query.limit).to_dict(orient="records")),
                "aggregation": {"metric": "MISSING_COUNT", "value": len(missing_df)},
                "result_count": len(missing_df),
                "columns": list(df.columns),
                "status": "success"
            }

        if query.operation in {"UNKNOWN", "UNSUPPORTED_QUERY"}:
            return {
                "operation": query.operation,
                "results": [],
                "aggregation": None,
                "result_count": 0,
                "columns": [],
                "status": "out_of_scope"
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
            return {
                "operation": "EXISTS",
                "results": self._clean_records(filtered_df.head(query.limit).to_dict(orient="records")),
                "aggregation": {"metric": "EXISTS", "value": exists, "count": len(filtered_df)},
                "result_count": len(filtered_df),
                "columns": list(filtered_df.columns),
                "status": "success"
            }

        aggregation_result: Optional[Dict[str, Any]] = None

        # Handle GROUP_EXTREME (e.g. "which department has highest average salary", "which city has most employees")
        if query.operation == "GROUP_EXTREME" and query.group_by_column in filtered_df.columns:
            grp_col = query.group_by_column
            target_col = query.target_column
            if target_col and target_col in filtered_df.columns:
                num_s = pd.to_numeric(filtered_df[target_col].astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce")
                tmp_df = filtered_df.copy()
                tmp_df["_grp_metric"] = num_s
                grp_res = tmp_df.groupby(grp_col)["_grp_metric"].mean().dropna()
                if not grp_res.empty:
                    best_grp = grp_res.idxmax() if query.sort_order.upper() == "DESC" else grp_res.idxmin()
                    best_val = round(float(grp_res[best_grp]), 2)
                    records = [{grp_col: best_grp, f"Average {target_col}": best_val}]
                    return {
                        "operation": "GROUP_EXTREME",
                        "results": records,
                        "aggregation": {"metric": f"{'HIGHEST' if query.sort_order.upper() == 'DESC' else 'LOWEST'}_AVERAGE", "group": best_grp, "value": best_val, "column": target_col},
                        "result_count": 1,
                        "columns": [grp_col, f"Average {target_col}"],
                        "status": "success"
                    }
            else:
                grp_counts = filtered_df.groupby(grp_col).size()
                if not grp_counts.empty:
                    best_grp = grp_counts.idxmax() if query.sort_order.upper() == "DESC" else grp_counts.idxmin()
                    best_cnt = int(grp_counts[best_grp])
                    records = [{grp_col: best_grp, "Employee Count": best_cnt}]
                    return {
                        "operation": "GROUP_EXTREME",
                        "results": records,
                        "aggregation": {"metric": f"{'MOST' if query.sort_order.upper() == 'DESC' else 'LEAST'}_COUNT", "group": best_grp, "value": best_cnt},
                        "result_count": 1,
                        "columns": [grp_col, "Employee Count"],
                        "status": "success"
                    }

        # Handle CROSS_ROW (e.g. "who earns the same salary", "who works in the same city")
        if query.operation == "CROSS_ROW" and query.target_column in filtered_df.columns:
            dup_mask = filtered_df.duplicated(subset=[query.target_column], keep=False)
            filtered_df = filtered_df[dup_mask].sort_values(by=query.target_column)

        # Handle DATE_EXTREME (e.g. "who joined most recently", "who joined earliest")
        if query.operation == "DATE_EXTREME" or (query.sort_column and any(t in str(query.sort_column).lower() for t in ["date", "time", "joining", "joined"])):
            date_col = query.target_column or query.sort_column
            if date_col and date_col in filtered_df.columns:
                parsed_dates = pd.to_datetime(filtered_df[date_col], errors="coerce")
                if parsed_dates.notnull().any():
                    if query.sort_order.upper() == "DESC":
                        target_dt = parsed_dates.max()
                        dt_metric = "LATEST_DATE"
                    else:
                        target_dt = parsed_dates.min()
                        dt_metric = "EARLIEST_DATE"
                    filtered_df = filtered_df[parsed_dates == target_dt]
                    aggregation_result = {"metric": dt_metric, "column": date_col, "value": str(target_dt.date())}

        # Handle Ordinal Ranking with rank_offset (e.g. "second highest", "third highest")
        elif query.rank_offset and query.sort_column and query.sort_column in filtered_df.columns:
            ascending = (query.sort_order.upper() == "ASC")
            num_s = pd.to_numeric(filtered_df[query.sort_column].astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce")
            if num_s.notnull().any():
                unique_sorted = sorted(num_s.dropna().unique(), reverse=not ascending)
                if len(unique_sorted) >= query.rank_offset:
                    target_val = unique_sorted[query.rank_offset - 1]
                    filtered_df = filtered_df[num_s == target_val]
                    aggregation_result = {"metric": f"RANK_{query.rank_offset}", "column": query.sort_column, "value": round(float(target_val), 2)}
                else:
                    filtered_df = filtered_df.iloc[0:0]

        # Handle Row-Retrieving MAX and MIN (with exact tie handling)
        elif (query.operation == "MAX" or (query.operation == "TOP_N" and query.limit == 1)) and not filtered_df.empty:
            target_col = query.target_column or query.sort_column
            if target_col and target_col in filtered_df.columns:
                num_s = pd.to_numeric(filtered_df[target_col].astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce")
                if num_s.notnull().any():
                    max_val = num_s.max()
                    filtered_df = filtered_df[num_s == max_val]
                    aggregation_result = {"metric": "MAX", "column": target_col, "value": round(float(max_val), 2)}

        elif (query.operation == "MIN" or (query.operation == "BOTTOM_N" and query.limit == 1)) and not filtered_df.empty:
            target_col = query.target_column or query.sort_column
            if target_col and target_col in filtered_df.columns:
                num_s = pd.to_numeric(filtered_df[target_col].astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce")
                if num_s.notnull().any():
                    min_val = num_s.min()
                    filtered_df = filtered_df[num_s == min_val]
                    aggregation_result = {"metric": "MIN", "column": target_col, "value": round(float(min_val), 2)}

        # Aggregation operations (COUNT, SUM, AVERAGE, AVG, MEAN, MEDIAN, VARIANCE, STD, RANGE)
        elif query.operation in {"COUNT", "SUM", "AVERAGE", "AVG", "MEAN", "MEDIAN", "VARIANCE", "VAR", "STD", "STDDEV", "STANDARD_DEVIATION", "RANGE"}:
            aggregation_result = execute_aggregation(filtered_df, query.operation, query.target_column)

        # Handle GROUP
        elif query.operation == "GROUP" and query.group_by_column:
            filtered_df = execute_group(filtered_df, query.group_by_column, "COUNT", query.target_column)

        # Apply sorting with explicit tie handling
        if query.sort_column and query.sort_column in filtered_df.columns and not query.rank_offset:
            ascending = (query.sort_order.upper() == "ASC")
            numeric_test = pd.to_numeric(filtered_df[query.sort_column].astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce")
            if numeric_test.notnull().sum() > len(filtered_df) * 0.7:
                filtered_df["_sort_tmp"] = numeric_test
                filtered_df = filtered_df.sort_values(by="_sort_tmp", ascending=ascending).drop(columns=["_sort_tmp"])
            else:
                filtered_df = filtered_df.sort_values(by=query.sort_column, ascending=ascending)

        # Column selection (always preserve _internal_row_id if present)
        result_cols = [c for c in filtered_df.columns if c != "_internal_row_id"]
        if query.select_columns:
            valid_select = [c for c in query.select_columns if c in filtered_df.columns]
            if valid_select:
                select_with_id = valid_select + (["_internal_row_id"] if "_internal_row_id" in filtered_df.columns else [])
                filtered_df = filtered_df[select_with_id]
                result_cols = valid_select

        if query.operation in {"TOP_N", "BOTTOM_N"} and query.limit:
            filtered_df = filtered_df.head(query.limit)

        total_matches = len(filtered_df)
        bounded_df = filtered_df.head(query.limit)
        records = self._clean_records(bounded_df.to_dict(orient="records"))

        # Check for NULL_VALUE in retrieved single row lookup
        if query.operation == "LOOKUP" and records and query.target_column:
            target_val = records[0].get(query.target_column)
            if target_val is None or pd.isna(target_val) or str(target_val).strip() == "" or str(target_val).lower() in {"nan", "none", "null", "n/a", "-"}:
                ent_name = records[0].get("state") or records[0].get("Employee Name") or records[0].get("Customer Name") or "The entity"
                null_msg = f"{ent_name} was found in the dataset, but its {query.target_column} value is not available."
                return {
                    "operation": "NULL_VALUE",
                    "results": records,
                    "aggregation": None,
                    "result_count": 0,
                    "columns": result_cols,
                    "status": "null_value",
                    "missing_entity": ent_name,
                    "missing_column": query.target_column,
                    "explanation": null_msg
                }

        # Check for 0 records in filtered query (NO_MATCHING_ROWS)
        if total_matches == 0 and query.conditions:
            first_c = query.conditions[0]
            val_str = str(first_c.value)
            if first_c.operator == "starts_with":
                no_match_msg = f'No matching records were found in the uploaded dataset for {first_c.column} beginning with "{val_str}".'
            else:
                no_match_msg = f'No matching records were found in the uploaded dataset for "{val_str}".'
            return {
                "operation": "NO_MATCH",
                "results": [],
                "aggregation": None,
                "result_count": 0,
                "columns": result_cols,
                "status": "no_matching_rows",
                "missing_entity": val_str,
                "explanation": no_match_msg
            }

        return {
            "operation": query.operation,
            "results": records,
            "aggregation": aggregation_result,
            "result_count": total_matches,
            "columns": result_cols,
            "status": "success",
        }

    def execute_comparison(
        self,
        df: pd.DataFrame,
        entity_col: str,
        val1: str,
        val2: str,
        metric_col: str
    ) -> Dict[str, Any]:
        """Execute precise side-by-side comparison between two entities or groups."""
        if df.empty or metric_col not in df.columns:
            return {"status": "error", "message": "Metric column not found."}

        s1 = df[df[entity_col].astype(str).str.strip().str.lower() == val1.strip().lower()]
        s2 = df[df[entity_col].astype(str).str.strip().str.lower() == val2.strip().lower()]

        if s1.empty or s2.empty:
            return {"status": "not_found", "message": f"Could not find both '{val1}' and '{val2}' in column '{entity_col}'."}

        # Calculate metric values
        num1 = pd.to_numeric(s1[metric_col].astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce").mean()
        num2 = pd.to_numeric(s2[metric_col].astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce").mean()

        diff = round(float(abs(num1 - num2)), 2)
        higher_entity = val1 if num1 > num2 else (val2 if num2 > num1 else "Tied")

        return {
            "status": "success",
            "entity_1": val1,
            "value_1": round(float(num1), 2),
            "entity_2": val2,
            "value_2": round(float(num2), 2),
            "difference": diff,
            "higher": higher_entity,
            "metric": metric_col
        }

    def _clean_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        clean = []
        for r in records:
            cr = {}
            for k, v in r.items():
                if pd.isna(v):
                    cr[k] = None
                elif isinstance(v, (np.integer, int)):
                    cr[k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    cr[k] = round(float(v), 2)
                else:
                    cr[k] = str(v)
            clean.append(cr)
        return clean

    def _build_condition_mask(self, df: pd.DataFrame, column: str, operator: str, value: Any) -> pd.Series:
        if column not in df.columns:
            return pd.Series(False, index=df.index)

        series = df[column]

        # Numeric check
        is_num = False
        num_val: Optional[float] = None
        if isinstance(value, (int, float)):
            is_num = True
            num_val = float(value)
        elif isinstance(value, str):
            clean_val_str = re.sub(r"[₹$,]", "", value).strip().lower()
            if clean_val_str.endswith("k"):
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

        # Strictly respect >= vs > and <= vs <
        if is_num and num_val is not None and operator in {">=", "<=", ">", "<", "=", "!="}:
            num_series = pd.to_numeric(series.astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce")
            if operator == ">=":
                return num_series >= num_val
            elif operator == "<=":
                return num_series <= num_val
            elif operator == ">":
                return num_series > num_val
            elif operator == "<":
                return num_series < num_val
            elif operator == "=":
                return num_series == num_val
            elif operator == "!=":
                return num_series != num_val

        # Text conditions
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


query_executor = QueryExecutor()

