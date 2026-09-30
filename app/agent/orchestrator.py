"""AgentOrchestrator coordinating the complete request lifecycle state machine."""

import time
from typing import Any, Dict, List, Optional
import pandas as pd

from app.agent.clarifier import ambiguity_clarifier
from app.agent.context import context_resolver
from app.agent.memory import conversation_memory
from app.agent.planner import agent_planner
from app.agent.router import agent_router
from app.agent.state import AgentStateEnum, RequestContext
from app.agent.tool_loop import adaptive_tool_loop
from app.agent.verifier import result_verifier
from app.ai.response_generator import response_generator
from app.dataset.loader import dataset_loader
from app.utils.logger import logger


class AgentOrchestrator:
    """Master Orchestrator managing request intake, context, planning, tools, verification, and answer delivery."""

    def process_request(
        self,
        user_message: str,
        session_id: str = "default",
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        active_df = df if df is not None else dataset_loader.dataframe

        # 1. Request Intake & State Initialization
        ctx = RequestContext(
            conversation_id=session_id,
            dataset_id=dataset_loader.dataset_name if df is None else "custom_dataset",
            user_message=user_message,
        )
        ctx.transition_to(AgentStateEnum.RECEIVED, "Request received.")

        if active_df.empty:
            ctx.transition_to(AgentStateEnum.COMPLETED, "No active dataset available.")
            return {
                "question": user_message,
                "operation": "EMPTY_DATASET",
                "results": [],
                "aggregation": None,
                "result_count": 0,
                "answer": "No active dataset available. Please upload a dataset first.",
                "dataset_name": dataset_loader.dataset_name,
                "processing_time": round(time.perf_counter() - start_time, 4),
                "grounded": True,
                "is_clarification": False,
                "debug_trace": {"events": [e.model_dump() for e in ctx.trace_events]},
            }

        cols = list(active_df.columns)
        dataset_schema = {
            "dataset_name": dataset_loader.dataset_name if df is None else "custom_dataset",
            "columns": cols,
            "row_count": len(active_df),
        }

        # 2. Language Understanding & Normalization
        ctx.transition_to(AgentStateEnum.UNDERSTANDING, "Normalizing language & intent...")
        ctx.normalized_message = user_message.strip()

        # 3. Context Resolution (Pronouns & References)
        ctx.transition_to(AgentStateEnum.CONTEXT_RESOLUTION, "Resolving pronouns and session references...")
        resolved_q, res_meta = context_resolver.resolve(ctx.normalized_message, session_id, cols)
        ctx.resolved_message = resolved_q
        ctx.metadata.update(res_meta)

        # 4. Request Routing
        ctx.transition_to(AgentStateEnum.ROUTING, "Classifying request strategy...")
        route_dec = agent_router.route(ctx.resolved_message, cols, active_df)

        # 5. Planning
        ctx.transition_to(AgentStateEnum.PLANNING, f"Constructing execution plan for strategy '{route_dec.question_type}'...")
        plan = agent_planner.build_plan(ctx.resolved_message, cols, active_df)

        # 6. Adaptive Tool Execution Loop
        tool_result = adaptive_tool_loop.run_loop(ctx, active_df, dataset_schema)

        # 7. Ambiguity Clarification Check
        amb_check = ambiguity_clarifier.check_ambiguity(
            ctx.resolved_message, active_df, entity_name=res_meta.get("resolved_entity")
        )
        if amb_check:
            ctx.transition_to(AgentStateEnum.CLARIFICATION_REQUIRED, amb_check["message"])
            return {
                "question": user_message,
                "operation": "CLARIFICATION",
                "results": amb_check["candidate_records"],
                "aggregation": None,
                "result_count": len(amb_check["candidate_records"]),
                "answer": amb_check["message"] + "\n\n" + "\n".join(amb_check["options"]),
                "dataset_name": dataset_loader.dataset_name,
                "processing_time": round(time.perf_counter() - start_time, 4),
                "grounded": True,
                "is_clarification": True,
                "debug_trace": {"events": [e.model_dump() for e in ctx.trace_events]},
            }

        # 8. Result Verification & Hallucination Firewall
        ctx.transition_to(AgentStateEnum.VERIFICATION, "Verifying results against Hallucination Firewall...")
        retrieved_results = tool_result.results if tool_result else []
        aggregation_result = tool_result.aggregation if tool_result else None
        operation = tool_result.operation if tool_result else "SEARCH"

        # 9. Answer Generation
        ctx.transition_to(AgentStateEnum.ANSWER_GENERATION, "Generating natural language grounded answer...")
        answer = response_generator.generate(
            question=ctx.resolved_message,
            operation=operation,
            results=retrieved_results,
            aggregation=aggregation_result,
        )

        is_grounded, discrepancies = result_verifier.verify_answer(answer, retrieved_results, aggregation_result)
        if not is_grounded:
            logger.warning(f"Hallucination Firewall caught discrepancies ({discrepancies}); using grounded fallback.")
            answer = response_generator.generate_grounded_fallback(
                ctx.resolved_message, operation, retrieved_results, aggregation_result
            )

        elapsed_time = round(time.perf_counter() - start_time, 4)

        # Record conversation turn in memory
        conversation_memory.add_turn(
            session_id=session_id,
            question=user_message,
            tool_name=tool_result.tool_name if tool_result else "search_dataset",
            operation=operation,
            results=retrieved_results,
            aggregation=aggregation_result,
            answer=answer,
        )

        ctx.transition_to(AgentStateEnum.COMPLETED, "Request completed successfully.")

        return {
            "question": user_message,
            "operation": operation,
            "conditions": tool_result.metadata.get("conditions", []) if tool_result else [],
            "results": retrieved_results,
            "aggregation": aggregation_result,
            "result_count": tool_result.row_count if tool_result else len(retrieved_results),
            "answer": answer,
            "dataset_name": dataset_loader.dataset_name,
            "processing_time": elapsed_time,
            "grounded": is_grounded,
            "is_clarification": False,
            "debug_trace": {
                "request_id": ctx.request_id,
                "conversation_id": ctx.conversation_id,
                "dataset_id": ctx.dataset_id,
                "operation": operation,
                "verified": is_grounded,
                "events": [e.model_dump() for e in ctx.trace_events],
            },
        }


agent_orchestrator = AgentOrchestrator()

