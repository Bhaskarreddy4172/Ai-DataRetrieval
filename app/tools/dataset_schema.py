"""Tool: schema_inspection - Inspect dataset columns, data types, nulls, and sample distinct values."""

from typing import Any, Dict
import pandas as pd
from pydantic import BaseModel, Field

from app.llm.schemas import ToolResult
from app.tools.base import BaseTool


class SchemaInspectionInput(BaseModel):
    pass


class SchemaInspectionTool(BaseTool):
    name = "schema_inspection"
    description = "Inspect dataset structure, column names, data types, distinct value counts, and sample values."
    input_schema = SchemaInspectionInput

    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> ToolResult:
        if df.empty:
            return ToolResult(success=True, tool_name=self.name, operation="SCHEMA", results=[], row_count=0)

        columns_profile = []
        for col in df.columns:
            s = df[col]
            num_s = pd.to_numeric(s, errors="coerce")
            col_type = "numeric" if num_s.notna().sum() > len(df) * 0.5 else "text"
            sample_vals = [str(v) for v in s.dropna().unique()[:5]]
            columns_profile.append({
                "name": col,
                "type": col_type,
                "null_count": int(s.isna().sum()),
                "unique_count": int(s.nunique()),
                "sample_values": sample_vals,
            })

        return ToolResult(
            success=True,
            tool_name=self.name,
            operation="SCHEMA",
            results=columns_profile,
            row_count=len(df),
            metadata={"columns_count": len(df.columns), "total_rows": len(df)},
        )

