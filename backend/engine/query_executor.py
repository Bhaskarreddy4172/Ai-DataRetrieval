"""Deterministic Data Execution Engine (Deliverable 12).

Executes structured query plans strictly via Pandas and DuckDB.
Never uses LLM/Ollama for arithmetic, filtering, counting, sorting, or aggregations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from app.query.executor import query_executor as core_executor
from app.query.schema import StructuredQuery
from app.dataset.loader import dataset_loader


class QueryExecutionEngine:
    """Strict Deterministic Execution Engine using Pandas & DuckDB."""

    def execute(
        self,
        query: Union[Dict[str, Any], StructuredQuery],
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Execute query plan and return verified records and aggregations."""
        active_df = df if df is not None else dataset_loader.dataframe

        if isinstance(query, dict):
            from app.query.schema import FilterCondition
            conditions = [
                FilterCondition(**c) if isinstance(c, dict) else c
                for c in query.get("filters", [])
            ]
            q_obj = StructuredQuery(
                operation=query.get("intent", query.get("operation", "FILTER")),
                target_column=query.get("target_column") or query.get("metric"),
                conditions=conditions,
                limit=query.get("limit", 10),
                aggregation=query.get("aggregation")
            )
        else:
            q_obj = query

        retrieved_df, agg_res = core_executor.execute_plan(q_obj, active_df)

        return {
            "result_count": len(retrieved_df),
            "records": retrieved_df.to_dict(orient="records"),
            "aggregation": agg_res,
            "columns": list(retrieved_df.columns),
            "execution_engine": "Pandas/DuckDB",
            "is_empty": retrieved_df.empty and agg_res is None
        }


query_execution_engine = QueryExecutionEngine()

