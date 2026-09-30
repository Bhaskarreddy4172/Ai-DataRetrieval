"""Query Intent Detector (Deliverable 9).

Classifies natural language user questions into structured intent JSON
covering 100+ query operations and compositional patterns.
"""

from typing import Any, Dict, List, Optional
from app.ai.intent_parser import intent_parser
from app.dataset.loader import dataset_loader


class QueryIntentDetector:
    """Comprehensive Intent Detection Engine."""

    def detect_intent(
        self,
        question: str,
        available_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detect core operation intent, entity keywords, and operator attributes."""
        cols = available_columns or dataset_loader.get_columns()
        parsed_intent = intent_parser.parse(question, cols)

        return {
            "intent": parsed_intent.operation,
            "target_column": parsed_intent.target_column,
            "entity": parsed_intent.entity,
            "filter_column": parsed_intent.filter_column,
            "filter_value": parsed_intent.filter_value,
            "operator": parsed_intent.operator,
            "limit": parsed_intent.limit,
            "confidence": parsed_intent.confidence,
            "is_aggregation": parsed_intent.operation in {"COUNT", "SUM", "AVG", "MAX", "MIN"},
            "is_ranking": parsed_intent.operation in {"TOP_N", "BOTTOM_N", "MAX", "MIN"},
            "is_boolean": parsed_intent.operation == "BOOLEAN_CHECK",
        }


query_intent_detector = QueryIntentDetector()

