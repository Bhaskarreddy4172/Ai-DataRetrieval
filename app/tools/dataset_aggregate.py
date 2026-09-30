"""Tool: aggregate_dataset - Perform aggregation metrics (COUNT, SUM, AVG, MIN, MAX, MEDIAN)."""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.llm.schemas import ToolResult
from app.tools.base import BaseTool


class AggregateDatasetInput(BaseModel):
    operation: str = Field(..., description="Aggregation operation: COUNT, SUM, AVG, MIN, MAX, MEDIAN")
    column: str = Field(..., description="Target column to perform aggregation on")


class AggregateDatasetTool(BaseTool):
    name = "aggregate_dataset"
    description = "Compute statistical aggregation (COUNT, SUM, AVG, MIN, MAX, MEDIAN) on a numeric or categorical column."
    input_schema = AggregateDatasetInput

    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> ToolResult:
        op = arguments.get("operation", "COUNT").upper()
        col = arguments.get("column")

        if df.empty or col not in df.columns:
            return ToolResult(
                success=False,
                tool_name=self.name,
                operation=op,
                error=f"Column '{col}' not found in dataset.",
            )

        series = df[col]
        num_series = pd.to_numeric(series, errors="coerce")

        calc_val = None
        matched_records = []

        if op == "COUNT":
            calc_val = int(series.dropna().count())
        elif op == "SUM":
            calc_val = float(num_series.sum())
        elif op == "AVG" or op == "AVERAGE" or op == "MEAN":
            calc_val = float(num_series.mean())
        elif op == "MEDIAN":
            calc_val = float(num_series.median())
        elif op == "MIN":
            if num_series.notna().any():
                min_val = num_series.min()
                calc_val = float(min_val) if not np.isnan(min_val) else str(series.min())
                matched_records = df[num_series == min_val].head(5).replace({np.nan: None}).to_dict(orient="records")
            else:
                calc_val = str(series.min())
        elif op == "MAX":
            if num_series.notna().any():
                max_val = num_series.max()
                calc_val = float(max_val) if not np.isnan(max_val) else str(series.max())
                matched_records = df[num_series == max_val].head(5).replace({np.nan: None}).to_dict(orient="records")
            else:
                calc_val = str(series.max())

        return ToolResult(
            success=True,
            tool_name=self.name,
            operation=op,
            results=matched_records,
            aggregation={"column": col, "operation": op, "value": calc_val},
            row_count=len(matched_records) if matched_records else 1,
            metadata={"calculated_value": calc_val},
        )

