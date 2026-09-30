"""Tool: relationship - Resolve relationships, candidate keys, and correlations between columns."""

from typing import Any, Dict
import pandas as pd
from pydantic import BaseModel, Field

from app.llm.schemas import ToolResult
from app.tools.base import BaseTool


class RelationshipInput(BaseModel):
    pass


class RelationshipTool(BaseTool):
    name = "relationship"
    description = "Resolve column relationships, primary candidate keys, and correlations in the dataset."
    input_schema = RelationshipInput

    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> ToolResult:
        if df.empty:
            return ToolResult(success=True, tool_name=self.name, operation="RELATIONSHIP", results=[], row_count=0)

        candidate_keys = []
        relationships = []

        for col in df.columns:
            if df[col].nunique() == len(df) and len(df) > 0:
                candidate_keys.append(col)

        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        correlations = {}
        if len(num_cols) > 1:
            corr_df = df[num_cols].corr()
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    val = corr_df.loc[c1, c2]
                    if not pd.isna(val):
                        correlations[f"{c1}_vs_{c2}"] = round(float(val), 3)

        payload = {
            "candidate_keys": candidate_keys,
            "correlations": correlations,
            "column_count": len(df.columns),
        }

        return ToolResult(
            success=True,
            tool_name=self.name,
            operation="RELATIONSHIP",
            results=[payload],
            aggregation=payload,
            row_count=len(df),
        )

