"""Concrete Ollama LLM provider implementation with deterministic fallback."""

import json
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, ValidationError

from app.ai.ollama_client import ollama_client
from app.llm.base import LLMProvider
from app.llm.schemas import LLMResponse, ToolCall, ToolDefinition
from app.utils.logger import logger


class OllamaProvider(LLMProvider):
    """Ollama LLM provider with dynamic tool selection and offline fallback."""

    def __init__(self, model_name: Optional[str] = None):
        if model_name:
            ollama_client.set_model(model_name)

    def get_status_message(self, status: str) -> str:
        """Return standardized user-facing error message based on failure classification."""
        if status == "UNAVAILABLE":
            return "The AI service is temporarily unavailable. Please try again."
        elif status == "MODEL_NOT_FOUND":
            return "The configured AI model is currently unavailable. Please check the AI service configuration."
        elif status in ("TIMEOUT", "ERROR"):
            return "I couldn't process that request. Please try again."
        elif status == "MISSING_DATA":
            return "This information is not available in the uploaded dataset."
        return "I couldn't process that request. Please try again."

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        res = ollama_client.generate(prompt=prompt, system=system, temperature=temperature)
        return res or ""

    def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        health = ollama_client.check_health()
        if not health.get("available"):
            return {}

        raw_res = ollama_client.generate(prompt=prompt, system=system, json_format=True, temperature=temperature)
        if raw_res:
            try:
                parsed = json.loads(raw_res)
                validated = schema.model_validate(parsed)
                return validated.model_dump()
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Structured output validation failed: {e}")

        # Fallback to empty default matching schema
        return {}

    def select_tool(
        self,
        question: str,
        available_tools: List[ToolDefinition],
        dataset_schema: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Prompt Ollama to select a tool call or construct deterministic plan if Ollama is unavailable."""
        health = ollama_client.check_health()
        
        if not health.get("available"):
            # Ollama offline: return deterministic tool selection
            return self._deterministic_tool_selection(question, available_tools, dataset_schema, context)

        # Ollama available: build prompt with tool definitions
        tools_str = json.dumps([t.model_dump() for t in available_tools], indent=2)
        system_prompt = (
            "You are an AI Dataset Assistant tool selector. Analyze the user question and select the exact tool "
            "and arguments required to answer it. Output JSON matching:\n"
            '{"tool_name": "...", "arguments": {...}, "explanation": "..."}'
        )
        user_prompt = (
            f"Active Dataset Schema:\n{json.dumps(dataset_schema, indent=2)}\n\n"
            f"Available Tools:\n{tools_str}\n\n"
            f"User Question: {question}\n\n"
            "Output JSON tool request:"
        )

        res_str = ollama_client.generate(prompt=user_prompt, system=system_prompt, json_format=True, temperature=0.0)
        if res_str:
            try:
                data = json.loads(res_str)
                tool_name = data.get("tool_name", "")
                args = data.get("arguments", {})
                if tool_name and any(t.name == tool_name for t in available_tools):
                    return LLMResponse(
                        content="",
                        tool_calls=[ToolCall(tool_name=tool_name, arguments=args, explanation=data.get("explanation"))],
                        is_final=False,
                    )
            except Exception as e:
                logger.warning(f"Ollama tool parsing failed ({e}); falling back to deterministic selection.")

        return self._deterministic_tool_selection(question, available_tools, dataset_schema, context)

    def _deterministic_tool_selection(
        self,
        question: str,
        available_tools: List[ToolDefinition],
        dataset_schema: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Deterministic NLP planner fallback when LLM is unavailable or unparseable."""
        q_lower = question.lower()
        cols = [c.get("name") if isinstance(c, dict) else str(c) for c in dataset_schema.get("columns", [])]
        num_cols = [c.get("name") for c in dataset_schema.get("columns", []) if isinstance(c, dict) and c.get("type") in ["number", "integer", "float"]]

        target_col = None
        for c in num_cols or cols:
            if c and c.lower() in q_lower:
                target_col = c
                break
        if not target_col and num_cols:
            target_col = num_cols[0]
        elif not target_col and cols:
            target_col = cols[0]

        # Aggregate max / min / count / sum
        if any(w in q_lower for w in ["highest", "maximum", "max", "most", "top"]):
            return LLMResponse(
                tool_calls=[ToolCall(tool_name="aggregate_dataset", arguments={"operation": "MAX", "column": target_col})],
                is_final=False,
            )
        if any(w in q_lower for w in ["lowest", "minimum", "min", "least", "bottom"]):
            return LLMResponse(
                tool_calls=[ToolCall(tool_name="aggregate_dataset", arguments={"operation": "MIN", "column": target_col})],
                is_final=False,
            )
        if any(w in q_lower for w in ["average", "avg", "mean"]):
            return LLMResponse(
                tool_calls=[ToolCall(tool_name="aggregate_dataset", arguments={"operation": "AVG", "column": target_col})],
                is_final=False,
            )
        if any(w in q_lower for w in ["total", "sum"]):
            return LLMResponse(
                tool_calls=[ToolCall(tool_name="aggregate_dataset", arguments={"operation": "SUM", "column": target_col})],
                is_final=False,
            )
        if any(w in q_lower for w in ["how many", "count"]):
            return LLMResponse(
                tool_calls=[ToolCall(tool_name="aggregate_dataset", arguments={"operation": "COUNT", "column": target_col})],
                is_final=False,
            )

        # Default fallback to dataset search / filter
        return LLMResponse(
            tool_calls=[ToolCall(tool_name="search_dataset", arguments={"query": question})],
            is_final=False,
        )

