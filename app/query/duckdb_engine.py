"""DuckDB In-Memory Query Engine for analytical SQL execution directly on Pandas DataFrames.

Features:
- In-memory execution with zero data copy overhead via duckdb.connect(":memory:")
- Exact SQL generation for aggregations, rankings, window functions, and percentiles
- Robust handling of numeric parsing, currency symbols, and text comparisons
- Sanitized column identifiers to prevent SQL injection
- Guaranteed resource cleanup via try/finally
- Graceful error handling and fallback to Pandas execution engine
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from app.query.schema import StructuredQuery
from app.utils.logger import logger
from app.utils.normalization import parse_numeric_value

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


class DuckDBQueryEngine:
    """Executes analytical SQL queries directly on DataFrames using DuckDB."""

    def __init__(self):
        self.is_available = HAS_DUCKDB

    def sanitize_ident(self, ident: str) -> str:
        """Escape and quote SQL column/table identifier safely."""
        cleaned = ident.replace('"', '""')
        return f'"{cleaned}"'

    def execute_sql(
        self,
        df: pd.DataFrame,
        sql: str,
        table_name: str = "df",
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        """Execute arbitrary SQL query against df in-memory with guaranteed cleanup."""
        if not self.is_available:
            return {"status": "error", "message": "DuckDB is not installed."}
        if df.empty:
            return {"status": "empty_dataset", "results": [], "result_count": 0, "columns": []}

        con = None
        try:
            con = duckdb.connect(database=":memory:")
            con.register(table_name, df)
            rel = con.execute(sql)
            res_df = rel.fetchdf()
            records = self._clean_records(res_df.to_dict(orient="records"))
            return {
                "status": "success",
                "results": records,
                "result_count": len(records),
                "columns": list(res_df.columns),
                "engine": "duckdb",
            }
        except Exception as e:
            logger.warning(f"DuckDB SQL execution error: {e}. SQL: {sql}")
            return {"status": "error", "message": str(e)}
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass

    def execute_structured_query(
        self,
        query: StructuredQuery,
        df: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        """Attempt to translate StructuredQuery into DuckDB SQL and execute it.
        
        Returns None if the query type is better handled by Pandas or if translation fails.
        """
        if not self.is_available or df.empty:
            return None

        # DuckDB handles aggregations, group operations, ranking, and extremes
        supported_ops = {
            "COUNT", "SUM", "AVERAGE", "AVG", "MEDIAN",
            "GROUP", "GROUP_EXTREME", "TOP_N", "BOTTOM_N",
            "MAX", "MIN"
        }
        if query.operation not in supported_ops:
            return None

        try:
            sql, agg_info = self._build_sql(query, df)
            if not sql:
                return None

            exec_res = self.execute_sql(df, sql)
            if exec_res.get("status") != "success":
                return None

            results = exec_res["results"]
            columns = exec_res["columns"]

            # Format aggregation result if present
            aggregation_result = None
            if agg_info and results:
                metric_name = agg_info.get("metric")
                col_name = agg_info.get("column")
                if metric_name in {"MAX", "MIN", "SUM", "AVERAGE", "MEDIAN", "COUNT"}:
                    first_val = list(results[0].values())[0] if results[0] else None
                    aggregation_result = {
                        "metric": metric_name,
                        "column": col_name,
                        "value": first_val,
                    }
                elif "GROUP" in str(metric_name) or "COUNT" in str(metric_name) or "AVERAGE" in str(metric_name):
                    aggregation_result = {
                        "metric": metric_name,
                        "group": results[0].get(query.group_by_column),
                        "value": list(results[0].values())[-1] if results[0] else None,
                        "column": col_name,
                    }

            return {
                "operation": query.operation,
                "results": results,
                "aggregation": aggregation_result,
                "result_count": len(results),
                "columns": columns,
                "status": "success",
                "engine": "duckdb",
            }
        except Exception as e:
            logger.warning(f"DuckDB translation failed: {e}. Falling back to Pandas engine.")
            return None

    def _build_sql(
        self,
        query: StructuredQuery,
        df: pd.DataFrame,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Convert a StructuredQuery to DuckDB SQL statement."""
        where_clauses = []
        if query.conditions:
            for cond in query.conditions:
                if cond.column in df.columns:
                    col_q = self.sanitize_ident(cond.column)
                    val = cond.value
                    op = cond.operator

                    # Handle numeric comparisons
                    if isinstance(val, (int, float)):
                        num_expr = f"TRY_CAST(REGEXP_REPLACE(CAST({col_q} AS VARCHAR), '[₹$,]', '', 'g') AS DOUBLE)"
                        where_clauses.append(f"{num_expr} {op} {val}")
                    elif op in {">", ">=", "<", "<="}:
                        parsed_num = parse_numeric_value(str(val))
                        num_expr = f"TRY_CAST(REGEXP_REPLACE(CAST({col_q} AS VARCHAR), '[₹$,]', '', 'g') AS DOUBLE)"
                        if parsed_num is not None:
                            where_clauses.append(f"{num_expr} {op} {parsed_num}")
                        else:
                            val_escaped = str(val).replace("'", "''")
                            where_clauses.append(f"{col_q} {op} '{val_escaped}'")
                    else:
                        val_escaped = str(val).replace("'", "''")
                        if op in {"=", "=="}:
                            where_clauses.append(f"LOWER(CAST({col_q} AS VARCHAR)) = LOWER('{val_escaped}')")
                        elif op in {"!=", "<>"}:
                            where_clauses.append(f"LOWER(CAST({col_q} AS VARCHAR)) != LOWER('{val_escaped}')")
                        elif op == "contains":
                            where_clauses.append(f"LOWER(CAST({col_q} AS VARCHAR)) LIKE '%{val_escaped.lower()}%'")
                        elif op == "starts_with":
                            where_clauses.append(f"LOWER(CAST({col_q} AS VARCHAR)) LIKE '{val_escaped.lower()}%'")
                        elif op == "ends_with":
                            where_clauses.append(f"LOWER(CAST({col_q} AS VARCHAR)) LIKE '%{val_escaped.lower()}'")

        where_str = ""
        if where_clauses:
            logical_op = (query.logical_operator or "AND").upper()
            joiner = f" {logical_op} "
            where_str = "WHERE " + joiner.join(where_clauses)

        agg_info = None

        # 1. Aggregations (COUNT, SUM, AVERAGE, AVG, MEDIAN)
        if query.operation in {"COUNT", "SUM", "AVERAGE", "AVG", "MEDIAN"}:
            target_col = query.target_column
            if not target_col or target_col not in df.columns:
                return None, None

            t_ident = self.sanitize_ident(target_col)
            num_expr = f"TRY_CAST(REGEXP_REPLACE(CAST({t_ident} AS VARCHAR), '[₹$,]', '', 'g') AS DOUBLE)"

            if query.operation == "COUNT":
                sql = f"SELECT COUNT({t_ident}) AS count_val FROM df {where_str}"
                agg_info = {"metric": "COUNT", "column": target_col}
            elif query.operation == "SUM":
                sql = f"SELECT ROUND(SUM({num_expr}), 2) AS sum_val FROM df {where_str}"
                agg_info = {"metric": "SUM", "column": target_col}
            elif query.operation in {"AVERAGE", "AVG"}:
                sql = f"SELECT ROUND(AVG({num_expr}), 2) AS avg_val FROM df {where_str}"
                agg_info = {"metric": "AVERAGE", "column": target_col}
            elif query.operation == "MEDIAN":
                sql = f"SELECT ROUND(MEDIAN({num_expr}), 2) AS median_val FROM df {where_str}"
                agg_info = {"metric": "MEDIAN", "column": target_col}
            return sql, agg_info

        # 2. GROUP_EXTREME
        elif query.operation == "GROUP_EXTREME" and query.group_by_column in df.columns:
            grp_ident = self.sanitize_ident(query.group_by_column)
            target_col = query.target_column
            sort_dir = "DESC" if (query.sort_order or "DESC").upper() == "DESC" else "ASC"

            if target_col and target_col in df.columns:
                t_ident = self.sanitize_ident(target_col)
                num_expr = f"TRY_CAST(REGEXP_REPLACE(CAST({t_ident} AS VARCHAR), '[₹$,]', '', 'g') AS DOUBLE)"
                sql = (
                    f"SELECT {grp_ident}, ROUND(AVG({num_expr}), 2) AS average_val "
                    f"FROM df {where_str} "
                    f"GROUP BY {grp_ident} "
                    f"ORDER BY average_val {sort_dir} NULLS LAST "
                    f"LIMIT 1"
                )
                agg_info = {
                    "metric": f"{'HIGHEST' if sort_dir == 'DESC' else 'LOWEST'}_AVERAGE",
                    "column": target_col,
                }
            else:
                sql = (
                    f"SELECT {grp_ident}, COUNT(*) AS count_val "
                    f"FROM df {where_str} "
                    f"GROUP BY {grp_ident} "
                    f"ORDER BY count_val {sort_dir} "
                    f"LIMIT 1"
                )
                agg_info = {
                    "metric": f"{'MOST' if sort_dir == 'DESC' else 'LEAST'}_COUNT",
                    "column": query.group_by_column,
                }
            return sql, agg_info

        # 3. GROUP
        elif query.operation == "GROUP" and query.group_by_column and query.group_by_column in df.columns:
            grp_ident = self.sanitize_ident(query.group_by_column)
            target_col = query.target_column
            if target_col and target_col in df.columns:
                t_ident = self.sanitize_ident(target_col)
                num_expr = f"TRY_CAST(REGEXP_REPLACE(CAST({t_ident} AS VARCHAR), '[₹$,]', '', 'g') AS DOUBLE)"
                clean_target = re.sub(r'[^a-zA-Z0-9_]', '', str(target_col)) or "metric"
                sql = (
                    f"SELECT {grp_ident}, ROUND(SUM({num_expr}), 2) AS total_{clean_target} "
                    f"FROM df {where_str} "
                    f"GROUP BY {grp_ident} "
                    f"ORDER BY total_{clean_target} DESC NULLS LAST"
                )
            else:
                sql = (
                    f"SELECT {grp_ident}, COUNT(*) AS count "
                    f"FROM df {where_str} "
                    f"GROUP BY {grp_ident} "
                    f"ORDER BY count DESC"
                )
            agg_info = {"metric": "GROUP", "column": query.group_by_column}
            return sql, agg_info

        # 4. MAX / MIN
        elif query.operation in {"MAX", "MIN"}:
            target_col = query.target_column or query.sort_column
            if not target_col or target_col not in df.columns:
                return None, None
            t_ident = self.sanitize_ident(target_col)
            num_expr = f"TRY_CAST(REGEXP_REPLACE(CAST({t_ident} AS VARCHAR), '[₹$,]', '', 'g') AS DOUBLE)"
            sort_dir = "DESC" if query.operation == "MAX" else "ASC"
            sql = f"SELECT * FROM df {where_str} ORDER BY {num_expr} {sort_dir} NULLS LAST LIMIT 1"
            agg_info = {"metric": query.operation, "column": target_col}
            return sql, agg_info

        # 5. TOP_N / BOTTOM_N
        elif query.operation in {"TOP_N", "BOTTOM_N"}:
            order_col = query.sort_column or query.target_column
            if not order_col or order_col not in df.columns:
                return None, None
            col_ident = self.sanitize_ident(order_col)
            num_expr = f"TRY_CAST(REGEXP_REPLACE(CAST({col_ident} AS VARCHAR), '[₹$,]', '', 'g') AS DOUBLE)"
            if query.operation == "TOP_N":
                sort_dir = "ASC" if (query.sort_column and query.sort_order.upper() == "ASC") else "DESC"
            else:
                sort_dir = "DESC" if (query.sort_column and query.sort_order.upper() == "DESC") else "ASC"
            limit = query.limit or 5
            sql = f"SELECT * FROM df {where_str} ORDER BY {num_expr} {sort_dir} NULLS LAST LIMIT {limit}"
            agg_info = {"metric": query.operation, "column": order_col}
            return sql, agg_info

        return None, None

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


duckdb_engine = DuckDBQueryEngine()
