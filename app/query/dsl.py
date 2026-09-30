"""Safe Query DSL conforming strictly to Section 28 specifications."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DSLFilter(BaseModel):
    column: str
    operator: str  # '=', '!=', '>', '<', '>=', '<=', 'contains', 'in', 'between'
    value: Any


class DSLAggregation(BaseModel):
    column: str
    function: str  # 'SUM', 'AVERAGE', 'COUNT', 'MIN', 'MAX', 'MEDIAN', 'STD', 'IQR_OUTLIERS'


class DSLSort(BaseModel):
    column: str
    direction: str = "ASC"  # 'ASC' or 'DESC'


class SafeQueryDSL(BaseModel):
    """Safe, controlled query abstract syntax tree (AST)."""
    operation: str = "FILTER"
    source: Optional[str] = None
    group_by: Optional[List[str]] = None
    aggregation: Optional[DSLAggregation] = None
    filters: List[DSLFilter] = Field(default_factory=list)
    sort: Optional[DSLSort] = None
    limit: Optional[int] = None
    distinct: bool = False
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)
