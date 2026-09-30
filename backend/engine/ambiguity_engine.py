from typing import Any, Dict, List, Optional
import pandas as pd
from app.query.no_match_engine import no_match_engine
from app.dataset.loader import dataset_loader


class AmbiguityEngine:
    """Ambiguity Detection and Clarification Generator."""

    def analyze(
        self,
        question: str,
        df: Optional[pd.DataFrame] = None,
        available_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Check if user question is ambiguous and formulate clarification options."""
        active_df = df if df is not None else dataset_loader.dataframe
        cols = available_columns or dataset_loader.get_columns()

        diag = no_match_engine.diagnose(question, cols, active_df)

        if diag and diag.category == "AMBIGUOUS":
            return {
                "is_ambiguous": True,
                "reason": diag.explanation,
                "clarification_question": diag.user_message,
                "target_column": diag.target_column,
                "needs_user_input": True
            }

        return {
            "is_ambiguous": False,
            "reason": "Unambiguous query",
            "clarification_question": None,
            "target_column": None,
            "needs_user_input": False
        }


ambiguity_engine = AmbiguityEngine()
