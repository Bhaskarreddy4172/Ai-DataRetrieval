"""Tools: sort_dataset, top_n, bottom_n - Order and rank dataset records."""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.llm.schemas import ToolResult
from app.tools.base import BaseTool


class SortDatasetInput(BaseModel):
    column: str = Field(..., description="Column to sort by")
    ascending: bool = Field(default=True, description="True for ASC, False for DESC")
    limit: int = Field(default=10, ge=1, le=100, description="Number of records to return")


class SortDatasetTool(BaseTool):
    name = "sort_dataset"
    description = "Sort dataset by a numeric or alphabetical column in ascending or descending order."
    input_schema = SortDatasetInput

    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> ToolResult:
        col = arguments.get("column")
        asc = arguments.get("ascending", True)
        limit = arguments.get("limit", 10)

        if df.empty or col not in df.columns:
            return ToolResult(success=False, tool_name=self.name, error=f"Column '{col}' not found.")

        # Attempt numeric sort first
        num_s = pd.to_numeric(df[col], errors="coerce")
        if num_s.notna().sum() > len(df) * 0.5:
            sorted_df = df.iloc[num_s.sort_values(ascending=asc).index]
        else:
            sorted_df = df.sort_values(by=col, ascending=asc)

        records = sorted_df.head(limit).replace({np.nan: None}).to_dict(orient="records")
        return ToolResult(
            success=True,
            tool_name=self.name,
            operation="SORT",
            results=records,
            row_count=len(records),
        )


class TopNInput(BaseModel):
    column: str = Field(..., description="Metric column to rank top records")
    n: int = Field(default=5, ge=1, le=50, description="Top N count")


class TopNTool(BaseTool):
    name = "top_n"
    description = "Retrieve the top N records with highest values in a specified column."
    input_schema = TopNInput

    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> ToolResult:
        col = arguments.get("column")
        n = arguments.get("n", 5)

        if df.empty or col not in df.columns:
            return ToolResult(success=False, tool_name=self.name, error=f"Column '{col}' not found.")

        num_s = pd.to_numeric(df[col], errors="coerce")
        if num_s.notna().sum() > 0:
            top_df = df.iloc[num_s.sort_values(ascending=False).index].head(n)
        else:
            top_df = df.sort_values(by=col, ascending=False).head(n)

        records = top_df.replace({np.nan: None}).to_dict(orient="records")
        return ToolResult(
            success=True,
            tool_name=self.name,
            operation="TOP_N",
            results=records,
            row_count=len(records),
        )


class BottomNInput(BaseModel):
    column: str = Field(..., description="Metric column to rank bottom records")
    n: int = Field(default=5, ge=1, le=50, description="Bottom N count")


class BottomNTool(BaseTool):
    name = "bottom_n"
    description = "Retrieve the bottom N records with lowest values in a specified column."
    input_schema = BottomNInput

    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> ToolResult:
        col = arguments.get("column")
        n = arguments.get("n", 5)

        if df.empty or col not in df.columns:
            return ToolResult(success=False, tool_name=self.name, error=f"Column '{col}' not found.")

        num_s = pd.to_numeric(df[col], errors="coerce")
        if num_s.notna().sum() > 0:
            bottom_df = df.iloc[num_s.sort_values(ascending=True).index].head(n)
        else:
            bottom_df = df.sort_values(by=col, ascending=True).head(n)

        records = bottom_df.replace({np.nan: None}).to_dict(orient="records")
        return ToolResult(
            success=True,
            tool_name=self.name,
            operation="BOTTOM_N",
            results=records,
            row_count=len(records),
        )

