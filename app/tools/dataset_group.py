"""Tool: group_by - Group dataset records by categorical column and compute aggregate metrics."""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.llm.schemas import ToolResult
from app.tools.base import BaseTool


class GroupByInput(BaseModel):
    group_column: str = Field(..., description="Categorical column to group records by")
    target_column: Optional[str] = Field(None, description="Metric column to aggregate within groups")
    operation: str = Field(default="COUNT", description="Aggregation operation: COUNT, SUM, AVG, MIN, MAX")


class GroupByTool(BaseTool):
    name = "group_by"
    description = "Group records by a categorical column (e.g. Department, City) and aggregate metrics."
    input_schema = GroupByInput

    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> ToolResult:
        g_col = arguments.get("group_column")
        t_col = arguments.get("target_column")
        op = arguments.get("operation", "COUNT").upper()

        if df.empty or g_col not in df.columns:
            return ToolResult(success=False, tool_name=self.name, error=f"Group column '{g_col}' not found.")

        if op == "COUNT" or not t_col or t_col not in df.columns:
            grouped = df.groupby(g_col, as_index=False).size().rename(columns={"size": "count"})
            grouped = grouped.sort_values(by="count", ascending=False)
        else:
            df_work = df.copy()
            df_work[t_col] = pd.to_numeric(df_work[t_col], errors="coerce")
            
            if op == "SUM":
                grouped = df_work.groupby(g_col, as_index=False)[t_col].sum()
            elif op in ["AVG", "AVERAGE", "MEAN"]:
                grouped = df_work.groupby(g_col, as_index=False)[t_col].mean()
            elif op == "MAX":
                grouped = df_work.groupby(g_col, as_index=False)[t_col].max()
            elif op == "MIN":
                grouped = df_work.groupby(g_col, as_index=False)[t_col].min()
            else:
                grouped = df_work.groupby(g_col, as_index=False).size().rename(columns={"size": "count"})

            grouped = grouped.sort_values(by=t_col, ascending=False)

        records = grouped.replace({np.nan: None}).to_dict(orient="records")
        return ToolResult(
            success=True,
            tool_name=self.name,
            operation="GROUP_BY",
            results=records,
            row_count=len(records),
            metadata={"group_column": g_col, "target_column": t_col, "operation": op},
        )

