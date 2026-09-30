"""Tool: lookup - Retrieve attribute values for a specific resolved entity."""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.llm.schemas import ToolResult
from app.tools.base import BaseTool


class LookupInput(BaseModel):
    entity_name: str = Field(..., description="Entity value or name to lookup")
    entity_column: Optional[str] = Field(None, description="Optional column containing the entity")
    target_attribute: Optional[str] = Field(None, description="Specific attribute/column value to retrieve")


class LookupTool(BaseTool):
    name = "lookup"
    description = "Lookup attribute value or details for a resolved entity name."
    input_schema = LookupInput

    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> ToolResult:
        entity_name = arguments.get("entity_name", "").strip().lower()
        e_col = arguments.get("entity_column")
        t_attr = arguments.get("target_attribute")

        if df.empty or not entity_name:
            return ToolResult(success=True, tool_name=self.name, operation="LOOKUP", results=[], row_count=0)

        search_cols = [e_col] if e_col and e_col in df.columns else list(df.columns)
        matched_mask = pd.Series(False, index=df.index)

        for c in search_cols:
            matched_mask = matched_mask | (df[c].astype(str).str.lower() == entity_name)

        matched_df = df[matched_mask]
        if matched_df.empty:
            # Fallback to contains
            for c in search_cols:
                matched_mask = matched_mask | df[c].astype(str).str.lower().str.contains(entity_name, regex=False, na=False)
            matched_df = df[matched_mask]

        records = matched_df.head(5).replace({np.nan: None}).to_dict(orient="records")
        attr_val = None
        if not matched_df.empty and t_attr and t_attr in matched_df.columns:
            attr_val = matched_df[t_attr].iloc[0]

        return ToolResult(
            success=True,
            tool_name=self.name,
            operation="LOOKUP",
            results=records,
            aggregation={"attribute": t_attr, "value": attr_val} if t_attr else None,
            row_count=len(records),
        )

