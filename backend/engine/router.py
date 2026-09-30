"""General Knowledge and Question Router (Deliverable 16).

Classifies inquiries into DATASET, GENERAL_KNOWLEDGE, HYBRID, or UNSUPPORTED.
Guarantees client dataset queries never mix with pre-trained LLM memory.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from app.router.question_router import question_router
from app.dataset.loader import dataset_loader


class QuestionRouterEngine:
    """Intelligent Question Router."""

    def route(
        self,
        question: str,
        df: Optional[pd.DataFrame] = None,
        available_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Classify question and return routing decision."""
        active_df = df if df is not None else dataset_loader.dataframe
        cols = available_columns or dataset_loader.get_columns()
        schema_info = dataset_loader.schema_intelligence

        decision = question_router.route(
            question=question,
            available_columns=cols,
            schema_intelligence=schema_info,
            df=active_df
        )

        return {
            "question": question,
            "target": decision.question_type,  # DATASET, GENERAL_KNOWLEDGE, HYBRID, UNSUPPORTED, AMBIGUOUS
            "is_boolean": decision.is_boolean,
            "boolean_assertion": decision.boolean_assertion,
            "dataset_grounding_required": decision.question_type in {"DATASET", "HYBRID"},
            "allow_general_knowledge": decision.question_type == "GENERAL_KNOWLEDGE"
        }


question_router_engine = QuestionRouterEngine()

