"""AgentPlanner constructing structured single and multi-step query execution plans."""

from typing import Any, Dict, List, Optional
import pandas as pd
from app.llm.schemas import ToolCall


class QueryPlanStep:
    def __init__(self, tool_name: str, arguments: Dict[str, Any], description: Optional[str] = None):
        self.tool_name = tool_name
        self.arguments = arguments
        self.description = description


class QueryPlan:
    def __init__(self, goal: str, steps: List[QueryPlanStep]):
        self.goal = goal
        self.steps = steps


class AgentPlanner:
    """Builds structured query execution plans based on dataset schema intelligence."""

    def build_plan(self, question: str, columns: List[str], df: Optional[pd.DataFrame] = None) -> QueryPlan:
        q_lower = question.lower()
        cols_lower = {c.lower(): c for c in columns}

        # Find target numerical and categorical columns
        num_cols = []
        if df is not None and not df.empty:
            for c in columns:
                num_s = pd.to_numeric(df[c], errors="coerce")
                if num_s.notna().sum() > len(df) * 0.5:
                    num_cols.append(c)

        target_col = None
        for c in columns:
            if c.lower() in q_lower:
                target_col = c
                break

        if not target_col:
            target_col = num_cols[0] if num_cols else (columns[0] if columns else "value")

        steps: List[QueryPlanStep] = []

        # Check for multi-condition / ranking questions
        if "highest" in q_lower or "maximum" in q_lower or "max" in q_lower or "most" in q_lower:
            steps.append(QueryPlanStep(
                tool_name="aggregate_dataset",
                arguments={"operation": "MAX", "column": target_col},
                description=f"Find maximum value in column '{target_col}'",
            ))
        elif "lowest" in q_lower or "minimum" in q_lower or "min" in q_lower or "least" in q_lower:
            steps.append(QueryPlanStep(
                tool_name="aggregate_dataset",
                arguments={"operation": "MIN", "column": target_col},
                description=f"Find minimum value in column '{target_col}'",
            ))
        elif "average" in q_lower or "avg" in q_lower or "mean" in q_lower:
            steps.append(QueryPlanStep(
                tool_name="aggregate_dataset",
                arguments={"operation": "AVG", "column": target_col},
                description=f"Find average value in column '{target_col}'",
            ))
        elif "how many" in q_lower or "count" in q_lower:
            steps.append(QueryPlanStep(
                tool_name="aggregate_dataset",
                arguments={"operation": "COUNT", "column": target_col},
                description=f"Count records for '{target_col}'",
            ))
        else:
            steps.append(QueryPlanStep(
                tool_name="search_dataset",
                arguments={"query": question},
                description=f"Search dataset for '{question}'",
            ))

        return QueryPlan(goal=question, steps=steps)


agent_planner = AgentPlanner()

