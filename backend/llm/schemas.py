"""Pydantic schemas for LLM providers, tool calls, and agent execution events."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="Tool capability description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="JSON schema of input arguments")


class ToolCall(BaseModel):
    tool_name: str = Field(..., description="Name of tool to execute")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments to pass to the tool")
    explanation: Optional[str] = Field(None, description="Reasoning for tool selection")


class ToolResult(BaseModel):
    success: bool = Field(..., description="Whether tool execution succeeded")
    tool_name: str = Field(..., description="Name of executed tool")
    operation: str = Field(default="EXECUTE", description="Operation performed")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved rows/records")
    aggregation: Optional[Dict[str, Any]] = Field(None, description="Aggregated metric result if applicable")
    row_count: int = Field(default=0, description="Total matched row count")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional tool metadata")


class LLMResponse(BaseModel):
    content: str = Field(default="", description="Generated text content")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tool calls requested by model")
    is_final: bool = Field(default=True, description="Whether response is final answer")
    raw_response: Optional[Any] = Field(None, description="Raw provider response")


class AgentEvent(BaseModel):
    event_type: str = Field(..., description="Event type: UNDERSTANDING, TOOL_SELECTION, TOOL_EXECUTION, FIREWALL, RESPONSE")
    message: str = Field(..., description="Human-readable event description")
    details: Dict[str, Any] = Field(default_factory=dict, description="Event payload")
    timestamp: float = Field(..., description="Event epoch timestamp")

