"""Structured Failure Analysis and Telemetry Logger.

Captures, categorizes, and provides diagnostic telemetry for unmapped columns,
unresolved entities, empty results, ambiguity collisions, and out-of-scope queries.
"""

import time
from collections import deque
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.utils.logger import logger


class FailureRecord(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    category: str = Field(..., description="Failure category: UNMAPPED_COLUMN, UNRESOLVED_ENTITY, EMPTY_RESULT, AMBIGUOUS_QUERY, OUT_OF_SCOPE, HALLUCINATION_DETECTED, SYNTAX_ERROR")
    question: str
    dataset_name: str
    operation: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    candidates_considered: Optional[List[Any]] = None
    suggested_remediation: Optional[str] = None


class FailureAnalysisLogger:
    """In-memory telemetry and categorization engine for query execution anomalies."""

    def __init__(self, max_records: int = 200):
        self.max_records = max_records
        self.records: deque = deque(maxlen=max_records)
        self.category_counts: Dict[str, int] = {
            "UNMAPPED_COLUMN": 0,
            "UNRESOLVED_ENTITY": 0,
            "EMPTY_RESULT": 0,
            "AMBIGUOUS_QUERY": 0,
            "OUT_OF_SCOPE": 0,
            "HALLUCINATION_DETECTED": 0,
            "SYNTAX_ERROR": 0,
        }
        self.total_failures = 0

    def log_failure(
        self,
        category: str,
        question: str,
        dataset_name: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        candidates_considered: Optional[List[Any]] = None,
        suggested_remediation: Optional[str] = None,
    ) -> FailureRecord:
        """Record and categorize an execution anomaly."""
        cat = category if category in self.category_counts else "OUT_OF_SCOPE"
        self.category_counts[cat] = self.category_counts.get(cat, 0) + 1
        self.total_failures += 1

        record = FailureRecord(
            category=cat,
            question=question,
            dataset_name=dataset_name,
            operation=operation,
            details=details or {},
            candidates_considered=candidates_considered,
            suggested_remediation=suggested_remediation,
        )
        self.records.appendleft(record)
        logger.warning(f"Failure logged [{cat}] for question: '{question}' on dataset '{dataset_name}'")
        return record

    def get_summary(self) -> Dict[str, Any]:
        """Return diagnostic metrics and summary of failures."""
        return {
            "total_failures": self.total_failures,
            "category_breakdown": dict(self.category_counts),
            "recent_failures_count": len(self.records),
            "recent_records": [r.model_dump() for r in list(self.records)[:20]],
        }

    def clear(self) -> None:
        """Reset telemetry logs."""
        self.records.clear()
        for k in self.category_counts:
            self.category_counts[k] = 0
        self.total_failures = 0


failure_logger = FailureAnalysisLogger()

