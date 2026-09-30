"""Tool: filter_dataset - Apply precise conditions (EQUALS, NOT_EQUALS, CONTAINS, GREATER_THAN, LESS_THAN, IN)."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.llm.schemas import ToolResult
from app.tools.base import BaseTool


class FilterCondition(BaseModel):
    column: str = Field(..., description="Target column to filter")
    operator: str = Field(..., description="Operator: EQUALS, NOT_EQUALS, CONTAINS, GREATER_THAN, LESS_THAN, GREATER_EQUAL, LESS_EQUAL, IN")
    value: Any = Field(..., description="Value or list of values to compare")


class FilterDatasetInput(BaseModel):
    conditions: List[FilterCondition] = Field(..., description="List of filter conditions to apply (AND logic)")
    limit: int = Field(default=50, ge=1, le=500, description="Maximum matching records to return")


class FilterDatasetTool(BaseTool):
    name = "filter_dataset"
    description = "Filter dataset records by applying explicit column conditions (EQUALS, GREATER_THAN, CONTAINS, etc.)."
    input_schema = FilterDatasetInput

    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> ToolResult:
        conditions = arguments.get("conditions", [])
        limit = arguments.get("limit", 50)

        if df.empty:
            return ToolResult(success=True, tool_name=self.name, operation="FILTER", results=[], row_count=0)

        filtered_df = df.copy()

        for cond in conditions:
            col = cond.get("column")
            op = cond.get("operator", "EQUALS").upper()
            val = cond.get("value")

            if col not in filtered_df.columns:
                continue

            series = filtered_df[col]

            if op == "EQUALS":
                filtered_df = filtered_df[series.astype(str).str.lower() == str(val).lower()]
            elif op == "NOT_EQUALS":
                filtered_df = filtered_df[series.astype(str).str.lower() != str(val).lower()]
            elif op == "CONTAINS":
                filtered_df = filtered_df[series.astype(str).str.lower().str.contains(str(val).lower(), regex=False, na=False)]
            elif op in ["GREATER_THAN", "GT"]:
                filtered_df = filtered_df[pd.to_numeric(series, errors="coerce") > float(val)]
            elif op in ["LESS_THAN", "LT"]:
                filtered_df = filtered_df[pd.to_numeric(series, errors="coerce") < float(val)]
            elif op in ["GREATER_EQUAL", "GTE"]:
                filtered_df = filtered_df[pd.to_numeric(series, errors="coerce") >= float(val)]
            elif op in ["LESS_EQUAL", "LTE"]:
                filtered_df = filtered_df[pd.to_numeric(series, errors="coerce") <= float(val)]
            elif op == "IN" and isinstance(val, list):
                val_strs = [str(v).lower() for v in val]
                filtered_df = filtered_df[series.astype(str).str.lower().isin(val_strs)]

        records = filtered_df.head(limit).replace({np.nan: None}).to_dict(orient="records")

        return ToolResult(
            success=True,
            tool_name=self.name,
            operation="FILTER",
            results=records,
            row_count=len(filtered_df),
            metadata={"conditions": conditions},
        )

