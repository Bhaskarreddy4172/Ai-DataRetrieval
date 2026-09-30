"""Abstract base class for all deterministic dataset tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type
import pandas as pd
from pydantic import BaseModel
from app.llm.schemas import ToolDefinition, ToolResult


class BaseTool(ABC):
    """Base class for every deterministic dataset tool."""

    name: str
    description: str
    input_schema: Type[BaseModel]

    def get_definition(self) -> ToolDefinition:
        """Return schema definition for LLM tool discovery."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_schema.model_json_schema(),
        )

    @abstractmethod
    def execute(self, df: pd.DataFrame, arguments: Dict[str, Any]) -> ToolResult:
        """Execute deterministic computation on Pandas DataFrame."""
        pass

