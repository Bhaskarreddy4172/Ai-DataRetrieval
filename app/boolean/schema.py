"""Pydantic schema definitions for Boolean and Fact-Checking queries."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BooleanAssertion(BaseModel):
    """Structured representation of a customer fact-check assertion."""
    intent: str = "BOOLEAN_CHECK"
    assertion_type: str = Field(
        ...,
        description="EQUALITY | INEQUALITY | EXISTENCE | NON_EXISTENCE | MEMBERSHIP | COMPARISON | RANKING | AGGREGATION | CROSS_ROW | DUPLICATE | MISSING_FIELD | UNKNOWN"
    )
    source_target: str = Field(
        "DATASET",
        description="DATASET | GENERAL_KNOWLEDGE | HYBRID"
    )
    subject: Optional[str] = Field(None, description="Primary entity or attribute under verification")
    relationship: Optional[str] = Field(None, description="Semantic relationship e.g. BELONGS_TO_STATE, WORKS_IN, GREATER_THAN, SAME_AS")
    object: Optional[Any] = Field(None, description="Target value, state, threshold, or entity being asserted")
    attribute: Optional[str] = Field(None, description="Target column or attribute being tested")
    comparison_op: Optional[str] = Field(None, description="Comparison operator: =, !=, >, <, >=, <=")
    secondary_subject: Optional[str] = Field(None, description="Secondary entity for cross-row comparisons")
    conditions: List[Dict[str, Any]] = Field(default_factory=list, description="Filter conditions for multi-condition or existence checks")
    raw_question: str = Field("", description="Raw input question")
    explanation: Optional[str] = Field(None, description="Contextual explanation")


class BooleanResult(BaseModel):
    """Evaluation result of a Boolean check with evidence tracking."""
    intent: str = "BOOLEAN_CHECK"
    result: Optional[bool] = Field(None, description="True, False, or None if unverified/unknown")
    status: str = Field(..., description="VERIFIED | CANNOT_VERIFY | AMBIGUOUS")
    answer: str = Field(..., description="User-facing explanation matching verified fact")
    source: str = Field(..., description="DATASET | GENERAL_KNOWLEDGE | HYBRID")
    confidence: float = 1.0
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Audit evidence: row_ids, columns, actual values")
    subject: Optional[str] = None
    attribute: Optional[str] = None
    actual_value: Optional[Any] = None
    expected_value: Optional[Any] = None

