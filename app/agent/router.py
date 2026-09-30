"""AgentRouter classifying request types (DATASET, GENERAL_KNOWLEDGE, HYBRID, AMBIGUOUS, UNSUPPORTED)."""

import re
from typing import Any, Dict, List, Optional
import pandas as pd


class RouterDecision:
    def __init__(
        self,
        question_type: str,
        confidence: float = 1.0,
        explanation: Optional[str] = None,
        dataset_subquery: Optional[str] = None,
        general_subquery: Optional[str] = None,
    ):
        self.question_type = question_type
        self.confidence = confidence
        self.explanation = explanation
        self.dataset_subquery = dataset_subquery
        self.general_subquery = general_subquery


class AgentRouter:
    """Classifies incoming user messages to determine execution strategy."""

    GREETINGS = {"hi", "hello", "hey", "good morning", "good evening", "greetings", "who are you", "what can you do"}

    def route(
        self,
        question: str,
        available_columns: List[str],
        df: Optional[pd.DataFrame] = None,
    ) -> RouterDecision:
        q_clean = question.strip().lower()

        # 1. Check for general greetings
        if q_clean in self.GREETINGS:
            return RouterDecision(
                question_type="GENERAL_KNOWLEDGE",
                confidence=1.0,
                explanation="General greeting.",
                general_subquery=question,
            )

        # 2. Check for explicit dataset references or column matches
        cols_lower = [c.lower() for c in available_columns]
        matches = [c for c in cols_lower if c in q_clean]

        if matches or any(w in q_clean for w in ["salary", "employee", "dataset", "table", "row", "column", "count", "average", "highest", "lowest"]):
            return RouterDecision(
                question_type="DATASET",
                confidence=0.95,
                dataset_subquery=question,
            )

        # Default to DATASET query
        return RouterDecision(
            question_type="DATASET",
            confidence=0.8,
            dataset_subquery=question,
        )


agent_router = AgentRouter()

