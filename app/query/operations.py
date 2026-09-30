"""Execution handlers for individual dataset query operations."""

import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from app.query.schema import FilterCondition, StructuredQuery


def execute_aggregation(df: pd.DataFrame, metric: str, target_column: Optional[str]) -> Optional[Dict[str, Any]]:
    """Execute statistical aggregation: SUM, AVERAGE, MIN, MAX, MEDIAN, COUNT."""
    if df.empty:
        return {"metric": metric, "value": 0}

    if metric == "COUNT":
        return {"metric": "COUNT", "value": len(df)}

    # Determine numeric column
    col = target_column
    if not col or col not in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns
        col = num_cols[0] if len(num_cols) > 0 else None

    if not col or col not in df.columns:
        return None

    series = pd.to_numeric(df[col].astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce").dropna()
    if series.empty:
        return {"metric": metric, "column": col, "value": 0}

    np_arr = np.asarray(series.values, dtype=float)
    if len(np_arr) == 0:
        return {"metric": metric, "column": col, "value": 0}

    val: float = 0.0
    if metric == "SUM":
        val = float(np.sum(np_arr))
    elif metric in {"AVERAGE", "AVG", "MEAN"}:
        val = float(np.mean(np_arr))
    elif metric == "MIN":
        val = float(np.min(np_arr))
    elif metric == "MAX":
        val = float(np.max(np_arr))
    elif metric == "MEDIAN":
        val = float(np.median(np_arr))
    elif metric in {"VARIANCE", "VAR"}:
        val = float(np.var(np_arr, ddof=1 if len(np_arr) > 1 else 0))
    elif metric in {"STD", "STDDEV", "STANDARD_DEVIATION"}:
        val = float(np.std(np_arr, ddof=1 if len(np_arr) > 1 else 0))
    elif metric == "RANGE":
        val = float(np.ptp(np_arr))

    return {
        "metric": metric,
        "column": col,
        "value": round(val, 2)
    }


def execute_distinct(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Retrieve unique non-null values of a target column."""
    if column not in df.columns:
        return {"values": [], "count": 0}

    uniques = df[column].dropna().unique().tolist()
    return {
        "column": column,
        "values": uniques[:50],
        "total_distinct": len(uniques)
    }


def execute_duplicates(df: pd.DataFrame, column: Optional[str] = None) -> Dict[str, Any]:
    """Find duplicate records in DataFrame or duplicate values in a column."""
    if df.empty:
        return {"duplicate_count": 0, "records": []}

    if column and column in df.columns:
        dups = df[df.duplicated(subset=[column], keep=False)].sort_values(by=column)
    else:
        dups = df[df.duplicated(keep=False)]

    records = dups.head(30).to_dict(orient="records")
    return {
        "duplicate_count": len(dups),
        "column": column or "all_columns",
        "records": records
    }


def execute_group(df: pd.DataFrame, group_col: str, metric: str = "COUNT", target_col: Optional[str] = None) -> pd.DataFrame:
    """Group by categorical column with specified aggregation."""
    if group_col not in df.columns or df.empty:
        return df

    if metric == "COUNT" or not target_col or target_col not in df.columns:
        res = df.groupby(group_col).size().reset_index(name="Count")
        return res.sort_values(by="Count", ascending=False)

    num_series = pd.to_numeric(df[target_col].astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce")
    tmp_df = df.copy()
    tmp_df["_metric_col"] = num_series

    if metric == "SUM":
        res = tmp_df.groupby(group_col)["_metric_col"].sum().reset_index(name=f"Total_{target_col}")
    elif metric in {"AVERAGE", "AVG", "MEAN"}:
        res = tmp_df.groupby(group_col)["_metric_col"].mean().reset_index(name=f"Avg_{target_col}")
    elif metric == "MEDIAN":
        res = tmp_df.groupby(group_col)["_metric_col"].median().reset_index(name=f"Median_{target_col}")
    elif metric in {"VARIANCE", "VAR"}:
        res = tmp_df.groupby(group_col)["_metric_col"].var().reset_index(name=f"Var_{target_col}")
    elif metric in {"STD", "STDDEV", "STANDARD_DEVIATION"}:
        res = tmp_df.groupby(group_col)["_metric_col"].std().reset_index(name=f"Std_{target_col}")
    elif metric == "MAX":
        res = tmp_df.groupby(group_col)["_metric_col"].max().reset_index(name=f"Max_{target_col}")
    elif metric == "MIN":
        res = tmp_df.groupby(group_col)["_metric_col"].min().reset_index(name=f"Min_{target_col}")
    else:
        res = tmp_df.groupby(group_col)["_metric_col"].sum().reset_index(name=f"Total_{target_col}")

    # Round metric column
    m_col = res.columns[1]
    res[m_col] = res[m_col].round(2)
    return res.sort_values(by=m_col, ascending=False)


def execute_cross_column_search(df: pd.DataFrame, term: str) -> pd.DataFrame:
    """Search for keyword across all text columns simultaneously."""
    if df.empty or not term:
        return df

    term_clean = term.strip().lower()
    masks = []
    for col in df.columns:
        mask = df[col].astype(str).str.lower().str.contains(re.escape(term_clean), na=False)
        masks.append(mask)

    if not masks:
        return df

    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m

    return df[combined]

