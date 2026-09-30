"""Central Tool Registry managing deterministic dataset tools."""

from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import ValidationError

from app.llm.schemas import ToolDefinition, ToolResult
from app.tools.base import BaseTool
from app.utils.logger import logger


class ToolRegistry:
    """Centralized Tool Registry for tool discovery, validation, and safe execution."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a new dataset tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool '{tool.name}'.")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def get_definitions(self) -> List[ToolDefinition]:
        """Return list of JSON schema definitions for all registered tools."""
        return [tool.get_definition() for tool in self._tools.values()]

    def execute(self, tool_name: str, arguments: Dict[str, Any], df: pd.DataFrame) -> ToolResult:
        """Validate input arguments and execute tool deterministically on DataFrame."""
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' not found in registry. Available tools: {list(self._tools.keys())}",
            )

        # Validate arguments using Pydantic schema
        try:
            validated_args = tool.input_schema.model_validate(arguments)
            arg_dict = validated_args.model_dump()
        except ValidationError as e:
            logger.warning(f"Tool argument validation failed for '{tool_name}': {e}")
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"Invalid arguments for tool '{tool_name}': {str(e)}",
            )

        # Safe execution
        try:
            return tool.execute(df, arg_dict)
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"Tool execution failed: {str(e)}",
            )


tool_registry = ToolRegistry()

