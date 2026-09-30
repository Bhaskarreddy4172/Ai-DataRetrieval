"""AdaptiveToolLoop managing step-by-step tool execution, observation, and plan repair."""

import time
from typing import Any, Dict, List, Optional
import pandas as pd

from app.agent.state import AgentStateEnum, RequestContext
from app.llm.model_manager import model_manager
from app.llm.schemas import ToolCall, ToolResult
from app.tools.registry import tool_registry
from app.utils.logger import logger


class AdaptiveToolLoop:
    """Executes multi-step tool calls, observes step results, and repairs plan if needed."""

    def __init__(self, max_steps: int = 5, timeout: float = 10.0):
        self.max_steps = max_steps
        self.timeout = timeout

    def run_loop(
        self,
        ctx: RequestContext,
        df: pd.DataFrame,
        dataset_schema: Dict[str, Any],
    ) -> ToolResult:
        step_count = 0
        start_time = time.perf_counter()
        provider = model_manager.get_provider()
        available_tools = tool_registry.get_definitions()
        last_result: Optional[ToolResult] = None

        while step_count < self.max_steps:
            elapsed = time.perf_counter() - start_time
            if elapsed > self.timeout:
                logger.warning(f"AdaptiveToolLoop timeout after {elapsed:.2f}s.")
                break

            step_count += 1
            ctx.transition_to(
                AgentStateEnum.TOOL_SELECTION,
                f"Selecting tool for step {step_count}...",
                details={"step": step_count},
            )

            # Request tool call from LLM / provider
            question_to_send = ctx.resolved_message or ctx.user_message
            llm_res = provider.select_tool(
                question=question_to_send,
                available_tools=available_tools,
                dataset_schema=dataset_schema,
            )

            if not llm_res.tool_calls or llm_res.is_final:
                break

            tool_call = llm_res.tool_calls[0]

            ctx.transition_to(
                AgentStateEnum.EXECUTING,
                f"Executing tool '{tool_call.tool_name}'",
                details={"tool_name": tool_call.tool_name, "arguments": tool_call.arguments},
            )

            # Safe tool execution
            res = tool_registry.execute(tool_call.tool_name, tool_call.arguments, df)
            ctx.transition_to(
                AgentStateEnum.OBSERVING,
                f"Observed result from tool '{tool_call.tool_name}'",
                details={"success": res.success, "row_count": res.row_count, "error": res.error},
            )

            if not res.success and res.error:
                # Plan repair: retry with search_dataset fallback if tool failed
                ctx.transition_to(
                    AgentStateEnum.REPLANNING,
                    f"Tool error ({res.error}); repairing plan via search_dataset fallback...",
                )
                res = tool_registry.execute("search_dataset", {"query": question_to_send}, df)

            last_result = res

            # Sufficiency check: if results or aggregation are non-empty, we have sufficient data
            if res.results or res.aggregation:
                break

        # Fallback to search if loop produced no results
        if not last_result or (not last_result.results and not last_result.aggregation):
            question_to_send = ctx.resolved_message or ctx.user_message
            last_result = tool_registry.execute("search_dataset", {"query": question_to_send}, df)

        return last_result


adaptive_tool_loop = AdaptiveToolLoop()

