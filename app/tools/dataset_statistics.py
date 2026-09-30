"""Tool: statistics - Statistical distributions, null counts, and column summaries."""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.llm.schemas import ToolResult
from app.tools.base import BaseTool


class StatisticsInput(BaseModel):
    column: Optional[str] = Field(default=None, description="Target column for statistics, or None for entire dataset")


class StatisticsTool(BaseTool):
    name = "statistics"
    description = "Compute statistical distributions, null counts, mean, median, min, max, and quantiles."
    input_schema = StatisticsInput

    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> ToolResult:
        col = arguments.get("column")

        if df.empty:
            return ToolResult(success=True, tool_name=self.name, operation="STATISTICS", results=[], row_count=0)

        stats_summary: Dict[str, Any] = {}

        if col and col in df.columns:
            raw_s = df[col]
            s: pd.Series = raw_s.iloc[:, 0] if isinstance(raw_s, pd.DataFrame) else raw_s
            converted = pd.to_numeric(s, errors="coerce")
            if isinstance(converted, pd.Series) and converted.notna().sum() > 0:
                num_arr = converted.dropna().to_numpy(dtype=float)
                stats_summary[col] = {
                    "count": int(len(num_arr)),
                    "null_count": int(s.isna().sum()),
                    "mean": float(np.mean(num_arr)),
                    "std": float(np.std(num_arr, ddof=1)) if len(num_arr) > 1 else 0.0,
                    "min": float(np.min(num_arr)),
                    "max": float(np.max(num_arr)),
                    "median": float(np.median(num_arr)),
                }
            else:
                stats_summary[col] = {
                    "count": int(s.count()),
                    "null_count": int(s.isna().sum()),
                    "unique_values": int(s.nunique()),
                }
        else:
            for c in df.columns:
                col_name = str(c)
                raw_s = df[col_name]
                s = raw_s.iloc[:, 0] if isinstance(raw_s, pd.DataFrame) else raw_s
                converted = pd.to_numeric(s, errors="coerce")
                if isinstance(converted, pd.Series) and converted.notna().sum() > len(df) * 0.5:
                    num_arr = converted.dropna().to_numpy(dtype=float)
                    if len(num_arr) > 0:
                        stats_summary[col_name] = {
                            "count": int(len(num_arr)),
                            "mean": float(np.mean(num_arr)),
                            "min": float(np.min(num_arr)),
                            "max": float(np.max(num_arr)),
                        }
                else:
                    stats_summary[col_name] = {
                        "count": int(s.count()),
                        "unique": int(s.nunique()),
                    }

        return ToolResult(
            success=True,
            tool_name=self.name,
            operation="STATISTICS",
            results=[stats_summary],
            aggregation=stats_summary,
            row_count=len(df),
        )


statistics_tool = StatisticsTool()
