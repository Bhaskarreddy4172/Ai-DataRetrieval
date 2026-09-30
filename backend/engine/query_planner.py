"""Query Planner Engine (Deliverable 10).

Translates natural language questions into safe, deterministic JSON Query DSL.
Never generates or executes raw unchecked SQL strings.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from app.query.planner import query_planner as core_planner
from app.query.schema import StructuredQuery
from app.dataset.loader import dataset_loader


class QueryPlannerEngine:
    """Universal Query Planner for JSON Query DSL generation."""

    def plan(
        self,
        question: str,
        df: Optional[pd.DataFrame] = None,
        available_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate structured JSON Query Plan."""
        active_df = df if df is not None else dataset_loader.dataframe
        cols = available_columns or dataset_loader.get_columns()
        schema_info = dataset_loader.schema_intelligence

        structured_q = core_planner.create_plan(
            question=question,
            available_columns=cols,
            schema_intelligence=schema_info,
            df=active_df
        )

        return {
            "intent": structured_q.operation,
            "target_column": structured_q.target_column,
            "metric": structured_q.target_column,
            "sort": "DESC" if structured_q.operation in {"MAX", "TOP_N"} else ("ASC" if structured_q.operation in {"MIN", "BOTTOM_N"} else None),
            "filters": [
                {
                    "column": f.column,
                    "operator": f.operator,
                    "value": f.value,
                    "condition_type": f.condition_type
                }
                for f in structured_q.conditions
            ],
            "limit": structured_q.limit,
            "aggregation": structured_q.aggregation,
            "explanation": structured_q.explanation,
            "raw_dsl": structured_q.model_dump()
        }


query_planner_engine = QueryPlannerEngine()

