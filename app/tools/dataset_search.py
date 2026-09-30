"""Tool: search_dataset - Search dataset values, text, or entities using lexical and fuzzy matching."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.llm.schemas import ToolResult
from app.tools.base import BaseTool


class SearchDatasetInput(BaseModel):
    query: str = Field(..., description="Text query or entity to search for")
    column: Optional[str] = Field(None, description="Optional specific column to search in")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum matching records to return")


class SearchDatasetTool(BaseTool):
    name = "search_dataset"
    description = "Search dataset for specific entity names, text phrases, or key terms across columns."
    input_schema = SearchDatasetInput

    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> ToolResult:
        query = arguments.get("query", "").strip().lower()
        column = arguments.get("column")
        limit = arguments.get("limit", 10)

        if df.empty or not query:
            return ToolResult(
                success=True,
                tool_name=self.name,
                operation="SEARCH",
                results=[],
                row_count=0,
            )

        search_cols = [column] if column and column in df.columns else list(df.columns)
        matched_mask = pd.Series(False, index=df.index)

        for col in search_cols:
            col_str = df[col].astype(str).str.lower()
            matched_mask = matched_mask | col_str.str.contains(query, regex=False, na=False)

        matched_df = df[matched_mask].head(limit)
        clean_records = matched_df.replace({np.nan: None}).to_dict(orient="records")

        return ToolResult(
            success=True,
            tool_name=self.name,
            operation="SEARCH",
            results=clean_records,
            row_count=len(matched_df),
            metadata={"query": query, "search_columns": search_cols},
        )

