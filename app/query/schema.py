"""Pydantic schemas for safe, structured query representation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FilterCondition(BaseModel):
    column: str = Field(..., description="Target dataset column name")
    operator: str = Field(
        default="=",
        description="Filter operator: =, !=, >, <, >=, <=, contains, not_contains, starts_with, ends_with, in, between"
    )
    value: Any = Field(..., description="Value to filter on")


class StructuredQuery(BaseModel):
    """Safe, controlled query specification to execute against pandas DataFrame."""
    operation: str = Field(
        default="FILTER",
        description="Operation: FILTER, LOOKUP, COUNT, SUM, AVERAGE, MIN, MAX, SORT, GROUP, DISTINCT, SEARCH, TOP_N, BOTTOM_N, UNKNOWN, UNSUPPORTED_QUERY, CLARIFICATION, AMBIGUOUS"
    )
    conditions: List[FilterCondition] = Field(default_factory=list, description="List of filter conditions")
    logical_operator: str = Field(default="AND", description="Combination logic: AND or OR")
    target_column: Optional[str] = Field(default=None, description="Column to aggregate, calculate, or retrieve")
    group_by_column: Optional[str] = Field(default=None, description="Column for grouping aggregations")
    select_columns: Optional[List[str]] = Field(default=None, description="Specific columns to include in results")
    sort_column: Optional[str] = Field(default=None, description="Column to sort by")
    sort_order: str = Field(default="ASC", description="Sort direction: ASC or DESC")
    limit: int = Field(default=50, ge=1, le=200, description="Max result rows to return")
    rank_offset: Optional[int] = Field(default=None, ge=1, le=50, description="1-based rank offset for ordinal queries like 2nd highest")
    missing_column: Optional[str] = Field(default=None, description="Name of any column requested that is missing from dataset")
    missing_entity: Optional[str] = Field(default=None, description="Name of any entity requested that is missing from dataset")
    no_match_type: Optional[str] = Field(default=None, description="Category of no-match: ENTITY_NOT_FOUND, COLUMN_NOT_FOUND, NO_MATCHING_ROWS, NULL_VALUE, AMBIGUOUS, UNSUPPORTED")
    explanation: Optional[str] = Field(default=None, description="Brief explanation of the parsed intent")
    clarification_prompt: Optional[str] = Field(default=None, description="Clarification request if question is ambiguous")
    debug_trace: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Pipeline debug trace")
