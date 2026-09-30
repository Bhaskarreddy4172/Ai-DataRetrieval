"""Data transfer and schema models for database entities."""

from pydantic import BaseModel
from typing import Optional


class QueryHistoryRecord(BaseModel):
    id: int
    timestamp: str
    question: str
    intent: Optional[str] = None
    result_count: int = 0
    processing_time: float = 0.0
    status: str = "success"
