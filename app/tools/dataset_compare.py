"""Tool: compare - Side-by-side comparison of two entities, groups, or metrics."""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.llm.schemas import ToolResult
from app.tools.base import BaseTool


class CompareInput(BaseModel):
    entity_column: str = Field(..., description="Entity column identifier")
    entity_1: str = Field(..., description="First entity name")
    entity_2: str = Field(..., description="Second entity name")
    metric_column: str = Field(..., description="Metric column to compare on")


class CompareTool(BaseTool):
    name = "compare"
    description = "Compare two entities or groups side-by-side on a numerical or categorical metric."
    input_schema = CompareInput

    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> ToolResult:
        e_col = arguments.get("entity_column")
        e1 = arguments.get("entity_1", "").strip().lower()
        e2 = arguments.get("entity_2", "").strip().lower()
        m_col = arguments.get("metric_column")

        if df.empty or e_col not in df.columns or m_col not in df.columns:
            return ToolResult(success=False, tool_name=self.name, error="Invalid comparison columns.")

        row_1 = df[df[e_col].astype(str).str.lower() == e1]
        row_2 = df[df[e_col].astype(str).str.lower() == e2]

        val_1 = row_1[m_col].iloc[0] if not row_1.empty else None
        val_2 = row_2[m_col].iloc[0] if not row_2.empty else None

        records = []
        if not row_1.empty:
            records.append(row_1.iloc[0].replace({np.nan: None}).to_dict())
        if not row_2.empty:
            records.append(row_2.iloc[0].replace({np.nan: None}).to_dict())

        diff = None
        try:
            if val_1 is not None and val_2 is not None:
                diff = float(val_1) - float(val_2)
        except (ValueError, TypeError):
            pass

        return ToolResult(
            success=True,
            tool_name=self.name,
            operation="COMPARE",
            results=records,
            aggregation={
                "entity_1": e1,
                "val_1": val_1,
                "entity_2": e2,
                "val_2": val_2,
                "difference": diff,
                "metric": m_col,
            },
            row_count=len(records),
        )

