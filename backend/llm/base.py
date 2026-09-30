"""Abstract base class for LLM Providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
from backend.llm.schemas import LLMResponse, ToolCall, ToolDefinition


class LLMProvider(ABC):
    """Abstract interface decoupling application from underlying LLM engine."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def select_tool(
        self,
        question: str,
        available_tools: List[ToolDefinition],
        dataset_schema: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        pass

