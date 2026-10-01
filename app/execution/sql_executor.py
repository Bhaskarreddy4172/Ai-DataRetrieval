"""SQL Executor: Safe, parameterized execution engine for state and village data."""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database.connection import db_manager
from app.utils.logger import logger


class SQLExecutor:
    """Deterministic, parameterized SQL execution engine supporting PostgreSQL and SQLite."""

    def __init__(self):
        self.db = db_manager

    def execute_query(
        self,
        query_sql: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute parameterized SQL query safely and return records + timing."""
        params = params or {}
        start_time = time.perf_counter()
        session = self.db.get_session()
        try:
            stmt = text(query_sql)
            result = session.execute(stmt, params)
            duration = time.perf_counter() - start_time

            if result.returns_rows:
                keys = list(result.keys())
                rows = [dict(zip(keys, row)) for row in result.fetchall()]
                return {
                    "success": True,
                    "row_count": len(rows),
                    "rows": rows,
                    "columns": keys,
                    "execution_time_seconds": round(duration, 4),
                    "sql": query_sql,
                }
            else:
                session.commit()
                return {
                    "success": True,
                    "row_count": result.rowcount,
                    "rows": [],
                    "columns": [],
                    "execution_time_seconds": round(duration, 4),
                    "sql": query_sql,
                }
        except Exception as ex:
            session.rollback()
            duration = time.perf_counter() - start_time
            logger.error(f"SQLExecutor error on query [{query_sql}]: {ex}")
            return {
                "success": False,
                "error": str(ex),
                "row_count": 0,
                "rows": [],
                "columns": [],
                "execution_time_seconds": round(duration, 4),
                "sql": query_sql,
            }
        finally:
            session.close()

    def get_village_extreme(
        self,
        metric: str = "population",
        direction: str = "MAX",
        state_filter: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Find the extreme (MAX/MIN) village for a metric, optionally filtered by state."""
        metric_col = metric.lower().strip()
        valid_cols = ["population", "area_sq_km", "literacy_rate_percent", "households", "no_of_males", "no_of_females"]
        if metric_col not in valid_cols:
            metric_col = "population"

        order_dir = "DESC" if direction.upper() == "MAX" else "ASC"
        params: Dict[str, Any] = {}

        if state_filter:
            sql = f"""
                SELECT state, capital, village, {metric_col} as value,
                       population, area_sq_km, literacy_rate_percent, households
                FROM village_data
                WHERE LOWER(state) = :state_name
                ORDER BY {metric_col} {order_dir}
                LIMIT 1
            """
            params["state_name"] = state_filter.lower().strip()
        else:
            sql = f"""
                SELECT state, capital, village, {metric_col} as value,
                       population, area_sq_km, literacy_rate_percent, households
                FROM village_data
                ORDER BY {metric_col} {order_dir}
                LIMIT 1
            """

        res = self.execute_query(sql, params)
        if res["success"] and res["rows"]:
            return res["rows"][0]
        return None

    def get_state_aggregation(
        self,
        metric: str = "population",
        agg_func: str = "SUM",
        state_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Aggregate a metric across villages, optionally for a specific state or grouped by state."""
        metric_col = metric.lower().strip()
        valid_cols = ["population", "area_sq_km", "literacy_rate_percent", "households", "no_of_males", "no_of_females"]
        if metric_col not in valid_cols:
            metric_col = "population"

        func_name = agg_func.upper()
        if func_name not in ["SUM", "AVG", "MIN", "MAX", "COUNT"]:
            func_name = "SUM"

        params: Dict[str, Any] = {}
        if state_filter:
            sql = f"""
                SELECT :state_name as state,
                       {func_name}({metric_col}) as aggregate_value,
                       COUNT(id) as record_count
                FROM village_data
                WHERE LOWER(state) = :state_name
            """
            params["state_name"] = state_filter.lower().strip()
            res = self.execute_query(sql, params)
            if res["success"] and res["rows"]:
                return res["rows"][0]
            return {"state": state_filter, "aggregate_value": 0, "record_count": 0}
        else:
            # Grouped by state ranking
            sql = f"""
                SELECT state,
                       {func_name}({metric_col}) as aggregate_value,
                       COUNT(id) as record_count
                FROM village_data
                GROUP BY state
                ORDER BY aggregate_value DESC
            """
            res = self.execute_query(sql)
            return res

    def get_all_villages_df(self, state_filter: Optional[str] = None) -> pd.DataFrame:
        """Fetch villages as a standardized DataFrame."""
        params = {}
        if state_filter:
            sql = "SELECT * FROM village_data WHERE LOWER(state) = :state_name"
            params["state_name"] = state_filter.lower().strip()
        else:
            sql = "SELECT * FROM village_data"

        res = self.execute_query(sql, params)
        if res["success"] and res["rows"]:
            return pd.DataFrame(res["rows"])
        return pd.DataFrame()


sql_executor = SQLExecutor()
