"""DatasetAgent orchestrating the tool-calling loop, execution safeguards, and grounded answers."""

import time
from typing import Any, Dict, List, Optional
import pandas as pd

from app.agent.firewall import hallucination_firewall
from app.agent.memory import conversation_memory
from app.ai.response_generator import response_generator
from app.dataset.loader import dataset_loader
from app.llm.model_manager import model_manager
from app.llm.schemas import AgentEvent, ToolCall, ToolResult
from app.tools.registry import tool_registry
from app.utils.logger import logger


class DatasetAgent:
    """Tool-using AI Agent orchestrating language understanding, tool selection, safe execution, and verified response generation."""

    def __init__(self, max_tool_calls: int = 5, timeout: float = 10.0):
        self.max_tool_calls = max_tool_calls
        self.timeout = timeout

    def process_query(
        self,
        question: str,
        session_id: str = "default",
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Execute the agent tool-calling loop for a natural language question."""
        start_time = time.perf_counter()
        events: List[AgentEvent] = []

        def log_event(event_type: str, message: str, details: Optional[Dict[str, Any]] = None):
            events.append(AgentEvent(
                event_type=event_type,
                message=message,
                details=details or {},
                timestamp=round(time.time(), 3),
            ))

        log_event("UNDERSTANDING", f"Analyzing question: '{question}'")

        active_df = df if df is not None else dataset_loader.dataframe
        if active_df.empty:
            return {
                "question": question,
                "operation": "EMPTY_DATASET",
                "results": [],
                "aggregation": None,
                "result_count": 0,
                "answer": "No active dataset available. Please upload a dataset first.",
                "dataset_name": dataset_loader.dataset_name,
                "processing_time": round(time.perf_counter() - start_time, 4),
                "grounded": True,
                "is_clarification": False,
                "events": [e.model_dump() for e in events],
            }

        dataset_schema = {
            "dataset_name": dataset_loader.dataset_name if df is None else "custom_dataset",
            "columns": list(active_df.columns),
            "row_count": len(active_df),
        }

        # Retrieve conversation memory & provider
        memory_state = conversation_memory.get_state(session_id)
        provider = model_manager.get_provider()
        available_tools = tool_registry.get_definitions()

        # Tool execution loop
        step_count = 0
        last_result: Optional[ToolResult] = None
        executed_tools: List[str] = []

        while step_count < self.max_tool_calls:
            elapsed = time.perf_counter() - start_time
            if elapsed > self.timeout:
                logger.warning(f"Agent execution timeout reached after {elapsed:.2f}s.")
                break

            step_count += 1
            log_event("TOOL_SELECTION", f"Selecting tool call for step {step_count}...")

            llm_res = provider.select_tool(
                question=question,
                available_tools=available_tools,
                dataset_schema=dataset_schema,
                context=memory_state,
            )

            if not llm_res.tool_calls or llm_res.is_final:
                break

            tool_call = llm_res.tool_calls[0]
            log_event(
                "TOOL_EXECUTION",
                f"Executing tool '{tool_call.tool_name}' with arguments {tool_call.arguments}",
                details={"tool_name": tool_call.tool_name, "arguments": tool_call.arguments},
            )

            executed_tools.append(tool_call.tool_name)
            tool_res = tool_registry.execute(tool_call.tool_name, tool_call.arguments, active_df)

            if not tool_res.success and tool_res.error:
                logger.warning(f"Tool execution warning: {tool_res.error}")
                # Try fallback or next step
                break

            last_result = tool_res

            # If tool returned specific matched results or aggregation, we can finish or chain next step
            if tool_res.results or tool_res.aggregation:
                break

        # Fallback to search if no tool executed successfully
        if not last_result or not last_result.success:
            last_result = tool_registry.execute("search_dataset", {"query": question}, active_df)
            executed_tools.append("search_dataset")

        log_event("FIREWALL", "Verifying factual claims against tool results...")

        # 5. Generate Natural Language Grounded Answer
        operation = last_result.operation if last_result else "SEARCH"
        retrieved_results = last_result.results if last_result else []
        aggregation_result = last_result.aggregation if last_result else None

        answer = response_generator.generate(
            question=question,
            operation=operation,
            results=retrieved_results,
            aggregation=aggregation_result,
        )

        # 6. Pass through Hallucination Firewall
        is_grounded, discrepancies = hallucination_firewall.verify(answer, retrieved_results, aggregation_result)
        if not is_grounded:
            logger.warning(f"Firewall caught discrepancies ({discrepancies}); using grounded fallback.")
            answer = response_generator.generate_grounded_fallback(
                question, operation, retrieved_results, aggregation_result
            )

        elapsed_time = round(time.perf_counter() - start_time, 4)

        # Record conversation turn
        conversation_memory.add_turn(
            session_id=session_id,
            question=question,
            tool_name=executed_tools[-1] if executed_tools else "search_dataset",
            operation=operation,
            results=retrieved_results,
            aggregation=aggregation_result,
            answer=answer,
        )

        log_event("RESPONSE", "Final response generated successfully.", details={"processing_time": elapsed_time})

        return {
            "question": question,
            "operation": operation,
            "conditions": last_result.metadata.get("conditions", []) if last_result else [],
            "results": retrieved_results,
            "aggregation": aggregation_result,
            "result_count": last_result.row_count if last_result else len(retrieved_results),
            "answer": answer,
            "dataset_name": dataset_loader.dataset_name,
            "processing_time": elapsed_time,
            "grounded": is_grounded,
            "is_clarification": False,
            "debug_trace": {
                "executed_tools": executed_tools,
                "step_count": step_count,
                "verified": is_grounded,
                "events": [e.model_dump() for e in events],
            },
        }


dataset_agent = DatasetAgent()
