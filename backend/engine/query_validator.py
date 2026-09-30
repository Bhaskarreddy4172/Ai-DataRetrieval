"""Query Plan Validator (Deliverable 11).

Validates structured query plans against dataset schema, column existence,
data types, operators, and entity presence.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
from app.query.validator import query_validator as core_validator
from app.query.schema import StructuredQuery
from app.dataset.loader import dataset_loader


class QueryValidatorEngine:
    """Rigorous Query Plan Validator."""

    def validate(
        self,
        query: Union[Dict[str, Any], StructuredQuery],
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Validate query plan and produce a detailed safety and schema report."""
        active_df = df if df is not None else dataset_loader.dataframe
        if isinstance(query, dict):
            # Parse dict into StructuredQuery
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

        valid, errors = core_validator.validate_plan(q_obj, active_df)

        return {
            "is_valid": valid,
            "errors": errors,
            "columns_checked": [c.column for c in q_obj.conditions if c.column],
            "target_column": q_obj.target_column,
            "operation": q_obj.operation,
            "dataset_rows": len(active_df)
        }


query_validator_engine = QueryValidatorEngine()
