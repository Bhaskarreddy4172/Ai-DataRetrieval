"""REST API Endpoints for dataset upload, multi-sheet, schema, query pipeline, and debug trace."""

import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field

from app.ai.ollama_client import ollama_client
from app.ai.response_generator import response_generator
from app.config import settings
from app.conversation.context import conversation_manager
from app.dataset.loader import dataset_loader
from app.dataset.metadata import DatasetMetadata
from app.dataset.suggestions import dataset_suggestion_engine
from app.database.database import get_query_history, log_query
from app.knowledge.general_knowledge import general_knowledge_engine
from app.router.question_router import question_router
from app.boolean.verifier import boolean_verifier
from app.boolean.schema import BooleanAssertion
import numpy as np
import pandas as pd
from app.query.schema import StructuredQuery
from app.query.executor import query_executor
from app.query.parser import query_parser
from app.query.decomposer import multi_question_decomposer
from app.verification.answer_validator import answer_validator
from app.verification.result_validator import result_validator
from app.verification.failure_logger import failure_logger
from app.query.analytics import dataset_analytics
from app.query.cache import query_cache
from app.utils.logger import logger
from app.utils.security import sanitize_filename, sanitize_user_input, validate_file_upload

router = APIRouter()


class ClearConversationRequest(BaseModel):
    session_id: Optional[str] = Field("default", description="Conversation session ID to clear, or empty to clear all")


class QueryRequest(BaseModel):
    question: str = Field(..., max_length=400, json_schema_extra={"example": "Who spent more than 10000?"})
    session_id: str = Field(default="default", description="Conversation session ID for follow-up context and session isolation")


class QueryResponse(BaseModel):
    question: str
    operation: str
    conditions: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    aggregation: Optional[Dict[str, Any]] = None
    result_count: int
    answer: str
    dataset_name: str
    processing_time: float
    grounded: bool = True
    is_clarification: bool = False
    source_type: str = "DATASET"
    dataset_used: bool = True
    general_knowledge_used: bool = False
    confidence: float = 1.0
    debug_trace: Optional[Dict[str, Any]] = None
    source_transparency: Optional[Dict[str, Any]] = None
    evidence: Optional[Dict[str, Any]] = None
    chart_spec: Optional[Dict[str, Any]] = None
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    hierarchy_scope: Optional[str] = None
    child_dataset: Optional[str] = None
    provenance: Optional[Dict[str, Any]] = None
    calculation_details: Optional[Dict[str, Any]] = None
    normalized_question: Optional[str] = None


class CompareRequest(BaseModel):
    entity_column: str
    entity_1: str
    entity_2: str
    metric_column: str


@router.get("/health")
def get_health() -> Dict[str, Any]:
    """System health check, dataset status, sheets, and Ollama status."""
    ollama_info = ollama_client.check_health()
    active_dataset = dataset_loader.dataset_name
    row_count = len(dataset_loader.dataframe)
    ollama_status = ollama_info.get("status", "READY" if ollama_info.get("available") else "UNAVAILABLE")

    from app.dataset.registry import parent_child_registry
    registry_health = parent_child_registry.get_health_report()

    return {
        "status": "healthy" if (ollama_info.get("available") and row_count > 0) else "ready",
        "active_dataset": active_dataset,
        "rows": row_count,
        "columns": dataset_loader.get_columns(),
        "sheets": dataset_loader.sheet_names,
        "active_sheet": dataset_loader.active_sheet,
        "ollama": "online" if ollama_info.get("available") else "offline (deterministic engine active)",
        "ollama_status": ollama_status,
        "ollama_details": ollama_info,
        "child_datasets_active": parent_child_registry.is_parent_child_active(),
        "child_datasets_count": registry_health.get("child_datasets_registered", 0),
        "registry_health": registry_health,
    }


@router.get("/dataset/health-dashboard")
def get_health_dashboard() -> Dict[str, Any]:
    """Return comprehensive health and readiness dashboard across all discovered and registered datasets."""
    from app.dataset.registry import parent_child_registry
    return parent_child_registry.get_health_dashboard()


@router.get("/dataset/all-metadata")
def get_all_dataset_metadata() -> List[Dict[str, Any]]:
    """Return catalog metadata for all registered datasets across the platform."""
    from app.dataset.registry import parent_child_registry
    return parent_child_registry.get_all_dataset_metadata()


@router.get("/dataset/samples")
def get_dataset_samples() -> List[Dict[str, Any]]:
    """List available preloaded sample datasets (states, customers, products, employees)."""
    return dataset_loader.get_available_samples()


@router.post("/dataset/sample/{sample_id}")
def load_sample_dataset(sample_id: str) -> Dict[str, Any]:
    """Switch active dataset to one of the built-in samples."""
    success, msg = dataset_loader.load_sample(sample_id)
    if not success:
        raise HTTPException(status_code=400, detail=msg)

    df = dataset_loader.dataframe
    suggestions = dataset_suggestion_engine.generate_suggestions(
        df, dataset_loader.dataset_name, dataset_loader.profile
    )
    summary = dataset_suggestion_engine.generate_summary(
        df, dataset_loader.dataset_name, dataset_loader.profile
    )

    return {
        "status": "success",
        "message": msg,
        "active_dataset": dataset_loader.dataset_name,
        "rows": len(df),
        "columns": dataset_loader.get_columns(),
        "profile": dataset_loader.profile,
        "sheets": dataset_loader.sheet_names,
        "suggestions": suggestions,
        "summary": summary,
    }


@router.post("/dataset/sheet/{sheet_name}")
def switch_excel_sheet(sheet_name: str) -> Dict[str, Any]:
    """Switch active sheet in multi-sheet Excel dataset."""
    success, msg = dataset_loader.switch_sheet(sheet_name)
    if not success:
        raise HTTPException(status_code=400, detail=msg)

    return {
        "status": "success",
        "message": f"Switched to sheet '{sheet_name}'.",
        "active_sheet": dataset_loader.active_sheet,
        "rows": len(dataset_loader.dataframe),
        "columns": dataset_loader.get_columns(),
    }


@router.post("/upload")
@router.post("/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload client CSV, Excel, or JSON dataset, profile it, index it, clear prior sessions, and set as active."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")
    clean_name = sanitize_filename(file.filename)
    dest_path = settings.UPLOADS_DIR / clean_name

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    file_size = os.path.getsize(dest_path)
    is_valid, val_msg = validate_file_upload(clean_name, file_size)
    if not is_valid:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=val_msg)

    success, msg = dataset_loader.load_dataset(dest_path, custom_name=clean_name)
    if not success:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=msg)

    # Strictly enforce multi-dataset isolation by clearing conversation history and caches
    conversation_manager.clear_all()
    try:
        query_cache.clear()
    except Exception:
        pass

    df = dataset_loader.dataframe
    suggestions = dataset_suggestion_engine.generate_suggestions(
        df, dataset_loader.dataset_name, dataset_loader.profile
    )
    summary = dataset_suggestion_engine.generate_summary(
        df, dataset_loader.dataset_name, dataset_loader.profile
    )

    return {
        "status": "success",
        "message": f"Dataset '{clean_name}' successfully uploaded and active.",
        "dataset_name": dataset_loader.dataset_name,
        "rows": len(df),
        "columns": dataset_loader.get_columns(),
        "profile": dataset_loader.profile,
        "relationships": dataset_loader.relationships,
        "sheets": dataset_loader.sheet_names,
        "suggestions": suggestions,
        "summary": summary,
    }


@router.get("/dataset/suggestions")
def get_dataset_suggestions() -> Dict[str, Any]:
    """Return dynamic, dataset-agnostic suggested questions based on active dataset."""
    df = dataset_loader.dataframe
    suggestions = dataset_suggestion_engine.generate_suggestions(
        df, dataset_loader.dataset_name, dataset_loader.profile
    )
    return {
        "status": "success",
        "dataset_name": dataset_loader.dataset_name,
        "suggestions": suggestions,
    }


@router.get("/dataset/summary")
def get_dataset_summary() -> Dict[str, Any]:
    """Return friendly natural language dataset summary based on active dataset."""
    df = dataset_loader.dataframe
    summary = dataset_suggestion_engine.generate_summary(
        df, dataset_loader.dataset_name, dataset_loader.profile
    )
    from app.dataset.registry import parent_child_registry
    registry_health = parent_child_registry.get_health_report()
    child_count = registry_health.get("child_datasets_registered", 0)
    if parent_child_registry.is_parent_child_active() and child_count > 0:
        summary += f" Hierarchical child datasets are active with {child_count} child datasets available across registered states."

    return {
        "status": "success",
        "dataset_name": dataset_loader.dataset_name,
        "rows": len(df),
        "columns": dataset_loader.get_columns(),
        "summary": summary,
        "child_datasets_count": child_count,
    }


@router.post("/clear-chat")
@router.post("/conversation/clear")
def clear_conversation(payload: Optional[ClearConversationRequest] = None) -> Dict[str, Any]:
    """Clear conversation history for a given session or all sessions."""
    session_id = payload.session_id if payload and payload.session_id else None
    if session_id:
        conversation_manager.clear_session(session_id)
        msg = f"Conversation cleared for session '{session_id}'."
    else:
        conversation_manager.clear_all()
        msg = "All conversation sessions cleared."

    # Also invalidate query cache
    try:
        query_cache.clear()
    except Exception:
        pass

    return {
        "status": "success",
        "message": msg,
        "session_id": session_id,
    }


@router.post("/new-session")
def create_new_session() -> Dict[str, Any]:
    """Generate a new isolated conversation session ID and reset its dialogue state."""
    session_id = str(uuid.uuid4())
    conversation_manager.clear_session(session_id)
    return {
        "status": "success",
        "session_id": session_id,
        "message": f"New session '{session_id}' initialized."
    }


@router.get("/history")
def get_chat_history(session_id: str = "default") -> Dict[str, Any]:
    """Retrieve full conversation dialogue history for a session."""
    raw_history = conversation_manager.get_history(session_id)
    history_items = []
    for turn in raw_history:
        q_obj = turn.get("query")
        op = getattr(q_obj, "operation", str(q_obj)) if q_obj else "QUERY"
        history_items.append({
            "question": turn.get("question", ""),
            "answer": turn.get("answer", ""),
            "operation": op,
            "result_count": turn.get("result_count", 0),
            "timestamp": turn.get("timestamp", 0.0),
        })
    return {
        "status": "success",
        "session_id": session_id,
        "turns_count": len(history_items),
        "history": history_items,
    }


@router.get("/download-chat")
def download_chat_history(session_id: str = "default", format: str = "json") -> Any:
    """Export and download conversation transcript for a session as JSON or plain text."""
    raw_history = conversation_manager.get_history(session_id)
    safe_session = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)

    if format.lower() in {"txt", "text"}:
        lines = [f"=== Universal Dataset Chatbot Dialogue Log ==="]
        lines.append(f"Session: {session_id}")
        lines.append(f"Active Dataset: {dataset_loader.dataset_name}")
        lines.append(f"Total Turns: {len(raw_history)}\n" + "="*50 + "\n")
        for idx, turn in enumerate(raw_history, 1):
            lines.append(f"[{idx}] USER:")
            lines.append(f"    {turn.get('question', '')}\n")
            lines.append(f"    ASSISTANT:")
            lines.append(f"    {turn.get('answer', '')}\n")
            lines.append("-" * 50 + "\n")

        content = "\n".join(lines)
        return PlainTextResponse(
            content=content,
            headers={"Content-Disposition": f'attachment; filename="chat_{safe_session}.txt"'}
        )

    # JSON default
    formatted_turns = []
    for idx, turn in enumerate(raw_history, 1):
        q_obj = turn.get("query")
        op = getattr(q_obj, "operation", str(q_obj)) if q_obj else "QUERY"
        formatted_turns.append({
            "turn": idx,
            "question": turn.get("question", ""),
            "answer": turn.get("answer", ""),
            "operation": op,
            "result_count": turn.get("result_count", 0),
            "timestamp": turn.get("timestamp", 0.0)
        })

    return JSONResponse(
        content={
            "session_id": session_id,
            "dataset_name": dataset_loader.dataset_name,
            "turns_count": len(formatted_turns),
            "turns": formatted_turns,
        },
        headers={"Content-Disposition": f'attachment; filename="chat_{safe_session}.json"'}
    )


@router.get("/dataset/profile")
def get_dataset_profile() -> Dict[str, Any]:
    """Return statistical profile and column metadata of the currently active dataset."""
    profile = dataset_loader.profile
    if not profile:
        raise HTTPException(status_code=404, detail="No active dataset loaded.")
    
    df = dataset_loader.dataframe
    col_types = {}
    for c in profile.get("columns", []):
        col_types[c.get("name", c.get("column_name", ""))] = c.get("type", "text")

    res = dict(profile)
    res["dataset_name"] = dataset_loader.dataset_name
    res["rows"] = len(df)
    res["row_count"] = len(df)
    res["columns"] = profile.get("columns", [])
    res["column_types"] = col_types
    res["entities"] = [c.get("name") for c in profile.get("columns", []) if c.get("is_entity") or c.get("type") == "text"]
    res["relationships"] = dataset_loader.relationships or []

    from app.dataset.registry import parent_child_registry
    registry_health = parent_child_registry.get_health_report()
    res["child_datasets_active"] = parent_child_registry.is_parent_child_active()
    res["child_datasets_count"] = registry_health.get("child_datasets_registered", 0)
    res["available_states"] = registry_health.get("available_states", [])
    return res


@router.get("/dataset/relationships")
def get_dataset_relationships() -> Dict[str, Any]:
    """Return discovered structural relationships, candidate keys, and correlations."""
    return dataset_loader.relationships


@router.post("/dataset/compare")
def compare_entities_endpoint(payload: CompareRequest) -> Dict[str, Any]:
    """Compare two entities side-by-side on a metric column."""
    df = dataset_loader.dataframe
    if df.empty:
        raise HTTPException(status_code=400, detail="No active dataset.")
    return query_executor.execute_comparison(
        df, payload.entity_column, payload.entity_1, payload.entity_2, payload.metric_column
    )


@router.get("/dataset/schema")
def get_dataset_schema() -> Dict[str, Any]:
    """Return compact LLM-friendly schema for query understanding."""
    profile = dataset_loader.profile
    if not profile:
        raise HTTPException(status_code=404, detail="No active dataset loaded.")

    return DatasetMetadata.get_compact_schema(
        columns_profile=profile.get("columns", []),
        dataset_name=dataset_loader.dataset_name,
        row_count=len(dataset_loader.dataframe)
    )


@router.post("/chat", response_model=QueryResponse)
@router.post("/query", response_model=QueryResponse)
def process_query(payload: QueryRequest) -> QueryResponse:
    """Execute 14-step strict pipeline: normalize, resolve context, map columns/values, execute, verify, explain."""
    start_time = time.perf_counter()

    clean_q = sanitize_user_input(payload.question)
    if not clean_q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Dedicated Multi-Stage Normalization Pipeline
    from app.nlp.normalizer import text_normalizer
    norm_res = text_normalizer.normalize_question(clean_q)
    normalized_q = norm_res.normalized_question

    df = dataset_loader.dataframe
    from app.dataset.registry import parent_child_registry
    from app.dataset.hierarchical_engine import hierarchical_query_engine

    # Dynamic dataset resolution for queries targeting standalone or registered datasets
    is_hierarchical = hierarchical_query_engine.can_handle(clean_q, payload.session_id) or hierarchical_query_engine.can_handle(normalized_q, payload.session_id)
    if not is_hierarchical:
        active_stem = Path(dataset_loader.dataset_name).stem.lower() if dataset_loader.dataset_name else ""
        if not (active_stem and active_stem in clean_q.lower()):
            resolved = parent_child_registry.resolve_dataset_for_query(clean_q) or parent_child_registry.resolve_dataset_for_query(normalized_q)
            if resolved:
                res_name, res_path, _ = resolved
                if df.empty or (res_path.name != dataset_loader.dataset_name and not parent_child_registry.is_village_level_query(clean_q)):
                    dataset_loader.load_dataset(res_path, custom_name=res_name)
                    df = dataset_loader.dataframe

    if df.empty and not is_hierarchical:
        # Fallback to main dataset if present
        main_path = parent_child_registry.get_main_dataset_path()
        if main_path and main_path.exists():
            dataset_loader.load_dataset(main_path, custom_name="india_states_capitals")
            df = dataset_loader.dataframe

    if df.empty and not is_hierarchical:
        raise HTTPException(status_code=400, detail="No active dataset available. Please upload a dataset first.")

    available_cols = dataset_loader.get_columns()
    schema_info = dataset_loader.schema_intelligence
    recent_history = conversation_manager.get_history(payload.session_id)

    # Sentiment analysis (independent, zero side-effects on execution)
    from app.nlp.sentiment import sentiment_analyzer
    sent_res = sentiment_analyzer.analyze(clean_q)
    user_sentiment = sent_res.sentiment
    user_sentiment_score = sent_res.score

    # 0. Contextualize follow-ups first if any
    clarified_q = normalized_q
    if recent_history:
        clarified_q, _ = conversation_manager.contextualize_and_resolve(
            normalized_q, payload.session_id, available_cols
        )

    # 0.4. Check Hierarchical Parent-Child dataset queries
    from app.dataset.hierarchical_engine import hierarchical_query_engine
    h_candidate = clarified_q if hierarchical_query_engine.can_handle(clarified_q, payload.session_id) else (clean_q if hierarchical_query_engine.can_handle(clean_q, payload.session_id) else None)
    if h_candidate is not None:
        h_res = hierarchical_query_engine.execute(h_candidate, payload.session_id)
        # Validate consistency of h_res
        from app.query.validator import answer_consistency_validator
        is_consistent, reason = answer_consistency_validator.validate_answer(
            question=clean_q,
            operation=h_res.get("operation", "LOOKUP"),
            scope=h_res.get("scope", ""),
            answer=h_res.get("answer", ""),
            results=h_res.get("results", []),
            session_id=payload.session_id
        )
        if not is_consistent:
            logger.warning(f"AnswerConsistencyValidator triggered re-plan: {reason}")
            from app.query.global_aggregation_engine import universal_global_aggregation_engine
            h_res = universal_global_aggregation_engine.execute(clean_q, payload.session_id)

        elapsed = round(time.perf_counter() - start_time, 4)
        h_query = StructuredQuery(
            operation=h_res.get("operation", "LOOKUP"),
            explanation=h_res.get("answer", ""),
            limit=max(1, h_res.get("result_count", 1))
        )
        conversation_manager.add_turn(
            session_id=payload.session_id,
            question=clean_q,
            query=h_query,
            result_count=h_res.get("result_count", 0),
            results=h_res.get("results", []),
            answer=h_res.get("answer", ""),
            metric=h_res.get("metric"),
            intent=h_res.get("operation"),
            return_entity=h_res.get("return_entity"),
            scope=h_res.get("scope")
        )
        cols_used = h_res.get("columns_used", [])
        return QueryResponse(
            question=clean_q,
            operation=h_res.get("operation", "LOOKUP"),
            conditions=[],
            results=h_res.get("results", []),
            aggregation=h_res.get("comparison_details") or h_res.get("aggregation"),
            result_count=h_res.get("result_count", 0),
            answer=h_res.get("answer", ""),
            dataset_name=dataset_loader.dataset_name,
            processing_time=elapsed,
            grounded=True,
            is_clarification=False,
            source_type="DATASET",
            dataset_used=True,
            general_knowledge_used=False,
            confidence=1.0 if h_res.get("verification_status") == "PASS" else 0.98,
            debug_trace={
                "hierarchical": h_res,
                "scope": h_res.get("scope"),
                "verification": h_res.get("verification_checks")
            },
            source_transparency={
                "dataset": h_res.get("child_dataset", dataset_loader.dataset_name),
                "scope": h_res.get("scope"),
                "rows_matched": h_res.get("result_count", 0),
                "columns_used": cols_used,
                "verification_status": h_res.get("verification_status")
            },
            sentiment=user_sentiment,
            sentiment_score=user_sentiment_score,
            hierarchy_scope=h_res.get("scope"),
            child_dataset=h_res.get("child_dataset"),
            provenance={
                "dataset_id": dataset_loader.dataset_name,
                "dataset_version": dataset_loader.dataset_hash,
                "source_dataset": dataset_loader.dataset_name,
                "child_dataset": h_res.get("child_dataset"),
                "columns_used": cols_used,
                "operations": [h_res.get("operation", "LOOKUP")],
                "calculation_formula": h_res.get("calculation_formula"),
                "verification_status": h_res.get("verification_status"),
            },
            calculation_details=h_res.get("comparison_details") or h_res.get("verification_checks") or h_res.get("aggregation"),
            normalized_question=normalized_q
        )

    # 0.5. Route query: DATASET, GENERAL_KNOWLEDGE, HYBRID, AMBIGUOUS, UNSUPPORTED
    decision = question_router.route(clarified_q, available_cols, schema_info, df=df)
    logger.info(f"Routing decision: {decision.question_type} (is_boolean={decision.is_boolean}) for '{clarified_q}'")

    # Handle Boolean / Fact-Check questions directly with zero hallucinations
    if decision.is_boolean and decision.boolean_assertion:
        elapsed = round(time.perf_counter() - start_time, 4)
        assertion = BooleanAssertion(**decision.boolean_assertion)
        b_res = boolean_verifier.verify(assertion, df)

        results_list = []
        if b_res.evidence.get("row_ids") and not df.empty:
            matched_df = df[df["_internal_row_id"].isin(b_res.evidence["row_ids"])]
            clean_df = matched_df.replace({np.nan: None})
            if isinstance(clean_df, pd.DataFrame):
                results_list = clean_df.to_dict(orient="records")
        elif b_res.evidence.get("row_id") is not None and not df.empty:
            matched_df = df[df["_internal_row_id"] == b_res.evidence["row_id"]]
            clean_df = matched_df.replace({np.nan: None})
            if isinstance(clean_df, pd.DataFrame):
                results_list = clean_df.to_dict(orient="records")

        # Save turn to conversation history
        bool_query = StructuredQuery(
            operation="BOOLEAN_CHECK",
            explanation=b_res.answer,
            limit=max(1, len(results_list))
        )
        conversation_manager.add_turn(
            payload.session_id,
            clean_q,
            bool_query,
            len(results_list),
            results_list,
            answer=b_res.answer
        )
        log_query(
            dataset_name=dataset_loader.dataset_name,
            question=clean_q,
            operation="BOOLEAN_CHECK",
            result_count=len(results_list),
            processing_time=elapsed,
            status="success",
            session_id=payload.session_id
        )

        return QueryResponse(
            question=clean_q,
            operation="BOOLEAN_CHECK",
            conditions=assertion.conditions,
            results=results_list,
            aggregation=None,
            result_count=len(results_list),
            answer=b_res.answer,
            dataset_name=dataset_loader.dataset_name,
            processing_time=elapsed,
            grounded=True,
            is_clarification=(b_res.status == "AMBIGUOUS"),
            source_type=b_res.source,
            dataset_used=(b_res.source in ["DATASET", "HYBRID"]),
            general_knowledge_used=(b_res.source in ["GENERAL_KNOWLEDGE", "HYBRID"]),
            confidence=b_res.confidence,
            debug_trace={
                "routing": decision.model_dump(),
                "boolean_assertion": assertion.model_dump(),
                "boolean_result": b_res.model_dump()
            },
            source_transparency={
                "source_type": b_res.source,
                "dataset_used": (b_res.source in ["DATASET", "HYBRID"]),
                "general_knowledge_used": (b_res.source in ["GENERAL_KNOWLEDGE", "HYBRID"]),
                "boolean_status": b_res.status,
                "boolean_result": b_res.result,
                "evidence": b_res.evidence
            },
            sentiment=user_sentiment,
            sentiment_score=user_sentiment_score
        )

    if decision.question_type == "GENERAL_KNOWLEDGE":
        elapsed = round(time.perf_counter() - start_time, 4)
        gk_res = general_knowledge_engine.answer_question(decision.normalized_question)
        ans_text = gk_res["answer"]
        return QueryResponse(
            question=clean_q,
            operation="GENERAL_KNOWLEDGE",
            conditions=[],
            results=[],
            aggregation=None,
            result_count=0,
            answer=ans_text,
            dataset_name=dataset_loader.dataset_name,
            processing_time=elapsed,
            grounded=True,
            is_clarification=False,
            source_type="GENERAL_KNOWLEDGE",
            dataset_used=False,
            general_knowledge_used=True,
            confidence=gk_res.get("confidence", 1.0),
            debug_trace={"routing": decision.model_dump(), "knowledge": gk_res},
            source_transparency={
                "source_type": "GENERAL_KNOWLEDGE",
                "dataset_used": False,
                "general_knowledge_used": True,
                "topic": gk_res.get("topic")
            },
            sentiment=user_sentiment,
            sentiment_score=user_sentiment_score
        )

    if decision.question_type == "HYBRID":
        # 1. Execute dataset subquery
        data_q = decision.dataset_subquery or clean_q
        parsed_q = query_parser.parse(data_q, available_cols, schema_info, recent_history)
        exec_res = query_executor.execute(parsed_q, df)
        data_results = exec_res.get("results", [])

        # 2. Execute general knowledge subquery
        gen_q = decision.general_subquery or clean_q
        gk_res = general_knowledge_engine.answer_question(gen_q)

        # 3. Combine responses
        ans_text = response_generator.generate_hybrid_response(
            data_results, gk_res["answer"], clean_q, decision.target_entity
        )
        elapsed = round(time.perf_counter() - start_time, 4)
        return QueryResponse(
            question=clean_q,
            operation="HYBRID",
            conditions=[c.model_dump() for c in parsed_q.conditions],
            results=data_results,
            aggregation=exec_res.get("aggregation"),
            result_count=len(data_results),
            answer=ans_text,
            dataset_name=dataset_loader.dataset_name,
            processing_time=elapsed,
            grounded=True,
            is_clarification=False,
            source_type="HYBRID",
            dataset_used=True,
            general_knowledge_used=True,
            confidence=0.95,
            debug_trace={"routing": decision.model_dump(), "dataset_exec": exec_res, "knowledge": gk_res},
            source_transparency={
                "source_type": "HYBRID",
                "dataset_used": True,
                "general_knowledge_used": True,
                "dataset": dataset_loader.dataset_name,
                "rows_matched": len(data_results)
            },
            sentiment=user_sentiment,
            sentiment_score=user_sentiment_score
        )

    if decision.question_type == "AMBIGUOUS":
        elapsed = round(time.perf_counter() - start_time, 4)
        failure_logger.log_failure(
            category="AMBIGUOUS_QUERY",
            question=clean_q,
            dataset_name=dataset_loader.dataset_name,
            operation="AMBIGUOUS",
            details={"explanation": decision.explanation}
        )
        return QueryResponse(
            question=clean_q,
            operation="CLARIFICATION",
            conditions=[],
            results=[],
            aggregation=None,
            result_count=0,
            answer=decision.explanation or "Your question is ambiguous. Could you please clarify your request?",
            dataset_name=dataset_loader.dataset_name,
            processing_time=elapsed,
            grounded=True,
            is_clarification=True,
            source_type="AMBIGUOUS",
            dataset_used=False,
            general_knowledge_used=False,
            confidence=decision.confidence,
            debug_trace={"routing": decision.model_dump()},
            source_transparency={"source_type": "AMBIGUOUS", "dataset_used": False, "general_knowledge_used": False},
            sentiment=user_sentiment,
            sentiment_score=user_sentiment_score
        )

    # 0.9. Check for compound / multi-question inquiries
    if multi_question_decomposer.is_compound_question(clarified_q):
        decomp_res = multi_question_decomposer.execute_decomposed(
            clarified_q, df, available_cols, schema_info, recent_history
        )
        elapsed = round(time.perf_counter() - start_time, 4)
        conversation_manager.add_turn(
            session_id=payload.session_id,
            question=clean_q,
            query=StructuredQuery(
                operation="MULTI_QUESTION",
                explanation=decomp_res["answer"],
                limit=max(1, decomp_res["result_count"])
            ),
            result_count=decomp_res["result_count"],
            results=decomp_res["combined_results"],
            answer=decomp_res["answer"]
        )
        log_query(
            dataset_name=dataset_loader.dataset_name,
            question=clean_q,
            operation="MULTI_QUESTION",
            result_count=decomp_res["result_count"],
            processing_time=elapsed,
            status="success",
            session_id=payload.session_id
        )
        return QueryResponse(
            question=clean_q,
            operation="MULTI_QUESTION",
            conditions=decomp_res.get("conditions", []),
            results=decomp_res["combined_results"],
            aggregation=decomp_res.get("aggregation"),
            result_count=decomp_res["result_count"],
            answer=decomp_res["answer"],
            dataset_name=dataset_loader.dataset_name,
            processing_time=elapsed,
            grounded=True,
            is_clarification=False,
            source_type="DATASET",
            dataset_used=True,
            general_knowledge_used=False,
            confidence=1.0,
            debug_trace=decomp_res.get("debug_trace", {}),
            source_transparency={
                "dataset": dataset_loader.dataset_name,
                "rows_matched": decomp_res["result_count"],
                "operation": "MULTI_QUESTION",
                "sub_questions": decomp_res["sub_questions"]
            },
            sentiment=user_sentiment,
            sentiment_score=user_sentiment_score
        )

    # 0.95. Check for Universal Comparison queries on active dataset (e.g. Rahul vs Priya)
    from app.query.comparison_engine import universal_comparison_engine
    if universal_comparison_engine.is_comparison_query(clarified_q) or universal_comparison_engine.is_comparison_query(clean_q):
        comp_plan = universal_comparison_engine.parse_and_plan(clarified_q, df=df, session_id=payload.session_id) or universal_comparison_engine.parse_and_plan(clean_q, df=df, session_id=payload.session_id)
        if comp_plan is not None:
            c_res = universal_comparison_engine.execute(comp_plan, df=df)
            if c_res.status != "NO_MATCH":
                elapsed = round(time.perf_counter() - start_time, 4)
                c_results = [{
                    "left_entity": c_res.left_entity,
                    "left_value": c_res.left_value,
                    "right_entity": c_res.right_entity,
                    "right_value": c_res.right_value,
                    "absolute_difference": c_res.absolute_difference,
                    "directed_difference": c_res.directed_difference,
                    "higher_entity": c_res.higher_entity,
                    "lower_entity": c_res.lower_entity,
                    "attribute": c_res.attribute,
                    "is_equal": c_res.is_equal,
                    "boolean_result": c_res.boolean_result,
                    "left_parent": c_res.left_parent,
                    "right_parent": c_res.right_parent,
                    "scope": c_res.scope,
                }] if c_res.left_entity else (c_res.rankings or [])
                c_query = StructuredQuery(
                    operation=c_res.operation,
                    explanation=c_res.answer,
                    limit=max(1, len(c_results))
                )
                conversation_manager.add_turn(
                    session_id=payload.session_id,
                    question=clean_q,
                    query=c_query,
                    result_count=len(c_results),
                    results=c_results,
                    answer=c_res.answer
                )
                return QueryResponse(
                    question=clean_q,
                    operation=c_res.operation,
                    conditions=[],
                    results=c_results,
                    aggregation=c_res.to_dict() if c_res.left_entity else None,
                    result_count=len(c_results),
                    answer=c_res.answer,
                    dataset_name=dataset_loader.dataset_name,
                    processing_time=elapsed,
                    grounded=True,
                    is_clarification=False,
                    source_type="DATASET",
                    dataset_used=True,
                    general_knowledge_used=False,
                    confidence=0.99,
                    debug_trace={
                        "comparison": c_res.to_dict(),
                        "verification": c_res.verification_checks
                    },
                    source_transparency={
                        "dataset": dataset_loader.dataset_name,
                        "rows_matched": len(c_results),
                        "columns_used": c_res.columns_used,
                        "verification_status": c_res.verification_status
                    },
                    sentiment=user_sentiment,
                    sentiment_score=user_sentiment_score,
                    provenance={
                        "dataset_id": dataset_loader.dataset_name,
                        "dataset_version": dataset_loader.dataset_hash,
                        "source_dataset": dataset_loader.dataset_name,
                        "columns_used": c_res.columns_used,
                        "operations": [c_res.operation],
                        "calculation_formula": c_res.calculation_formula,
                        "verification_status": c_res.verification_status,
                    },
                    normalized_question=normalized_q
                )

    # 1. Parse natural language into safe StructuredQuery + Debug Trace
    parsed_query = query_parser.parse(clean_q, available_cols, schema_info, recent_history)
    debug_trace = parsed_query.debug_trace or {}

    # 2. Contextualize follow-ups and pronouns (e.g. 'What about in hyd?', 'What is their salary?')
    clarified_q, context_query = conversation_manager.contextualize_and_resolve(
        clean_q, payload.session_id, available_cols, parsed_query
    )
    final_query: StructuredQuery = context_query if context_query is not None else parsed_query

    if clarified_q != clean_q and context_query is None:
        final_query = query_parser.parse(clarified_q, available_cols, schema_info, recent_history)
        debug_trace = final_query.debug_trace or debug_trace

    # Check if a clarification question is required (Ambiguity Guard)
    if final_query.operation == "CLARIFICATION":
        elapsed = round(time.perf_counter() - start_time, 4)
        failure_logger.log_failure(
            category="AMBIGUOUS_QUERY",
            question=clean_q,
            dataset_name=dataset_loader.dataset_name,
            operation="CLARIFICATION",
            details={"explanation": final_query.explanation}
        )
        return QueryResponse(
            question=clean_q,
            operation="CLARIFICATION",
            conditions=[],
            results=[],
            aggregation=None,
            result_count=0,
            answer=final_query.explanation or "Could you please clarify your request?",
            dataset_name=dataset_loader.dataset_name,
            processing_time=elapsed,
            grounded=True,
            is_clarification=True,
            debug_trace=debug_trace,
            source_transparency={"dataset": dataset_loader.dataset_name, "rows_matched": 0, "operation": "CLARIFICATION"},
            sentiment=user_sentiment,
            sentiment_score=user_sentiment_score
        )

    # Check if unmapped column or unsupported query can be resolved via Child Datasets
    if final_query.operation in {"UNSUPPORTED_QUERY", "UNKNOWN"} or final_query.missing_column:
        from app.dataset.hierarchical_engine import hierarchical_query_engine
        h_candidate = clarified_q if hierarchical_query_engine.can_handle(clarified_q, payload.session_id) else (clean_q if hierarchical_query_engine.can_handle(clean_q, payload.session_id) else None)
        if h_candidate is not None:
            h_res = hierarchical_query_engine.execute(h_candidate, payload.session_id)
            if h_res.get("operation") != "NO_MATCH":
                elapsed = round(time.perf_counter() - start_time, 4)
                cols_used = h_res.get("columns_used", [])
                h_query = StructuredQuery(
                    operation=h_res.get("operation", "LOOKUP"),
                    explanation=h_res.get("answer", ""),
                    limit=max(1, h_res.get("result_count", 1))
                )
                conversation_manager.add_turn(
                    session_id=payload.session_id,
                    question=clean_q,
                    query=h_query,
                    result_count=h_res.get("result_count", 0),
                    results=h_res.get("results", []),
                    answer=h_res.get("answer", "")
                )
                return QueryResponse(
                    question=clean_q,
                    operation=h_res.get("operation", "LOOKUP"),
                    conditions=[],
                    results=h_res.get("results", []),
                    aggregation=h_res.get("aggregation"),
                    result_count=h_res.get("result_count", 0),
                    answer=h_res.get("answer", ""),
                    dataset_name=dataset_loader.dataset_name,
                    processing_time=elapsed,
                    grounded=True,
                    is_clarification=False,
                    source_type="DATASET",
                    dataset_used=True,
                    general_knowledge_used=False,
                    confidence=0.98,
                    debug_trace={"hierarchical": h_res, "scope": h_res.get("scope")},
                    source_transparency={
                        "dataset": h_res.get("child_dataset", dataset_loader.dataset_name),
                        "scope": h_res.get("scope"),
                        "rows_matched": h_res.get("result_count", 0),
                        "columns_used": cols_used
                    },
                    sentiment=user_sentiment,
                    sentiment_score=user_sentiment_score,
                    hierarchy_scope=h_res.get("scope"),
                    child_dataset=h_res.get("child_dataset"),
                    provenance={
                        "dataset_id": dataset_loader.dataset_name,
                        "dataset_version": dataset_loader.dataset_hash,
                        "source_dataset": dataset_loader.dataset_name,
                        "child_dataset": h_res.get("child_dataset"),
                        "columns_used": cols_used,
                        "operations": [h_res.get("operation", "LOOKUP")],
                        "calculation_formula": h_res.get("calculation_formula"),
                    },
                    calculation_details=h_res.get("aggregation"),
                    normalized_question=normalized_q
                )

        failure_logger.log_failure(
            category="UNMAPPED_COLUMN" if final_query.missing_column else "OUT_OF_SCOPE",
            question=clean_q,
            dataset_name=dataset_loader.dataset_name,
            operation=final_query.operation,
            details={"missing_column": final_query.missing_column}
        )

    # 3. Deterministic Python execution on Pandas DataFrame
    exec_result = query_executor.execute(final_query, df)
    retrieved_results = exec_result.get("results", [])
    aggregation_result = exec_result.get("aggregation")
    result_count = exec_result.get("result_count", len(retrieved_results))

    if result_count == 0 and final_query.conditions:
        failure_logger.log_failure(
            category="EMPTY_RESULT",
            question=clean_q,
            dataset_name=dataset_loader.dataset_name,
            operation=final_query.operation,
            details={"conditions": [c.model_dump() for c in final_query.conditions]}
        )

    # 4. Result Verification (checks bounds, types, constraints)
    is_res_valid, res_val_msg = result_validator.validate(final_query, exec_result)
    if not is_res_valid:
        logger.warning(f"Result validation warning: {res_val_msg}")

    # 5. Generate Grounded Natural Language Answer
    answer = response_generator.generate(
        question=clean_q,
        operation=final_query.operation,
        results=retrieved_results,
        aggregation=aggregation_result,
        missing_column=final_query.missing_column,
        missing_entity=final_query.missing_entity,
        explanation=exec_result.get("explanation") or final_query.explanation
    )

    # 6. Strict Answer Verification (ensure zero hallucinations of entities or numbers)
    is_ans_valid, ans_discrepancies = answer_validator.verify_answer(
        answer, retrieved_results, aggregation_result, final_query.operation
    )
    if not is_ans_valid:
        failure_logger.log_failure(
            category="HALLUCINATION_DETECTED",
            question=clean_q,
            dataset_name=dataset_loader.dataset_name,
            operation=final_query.operation,
            details={"discrepancies": ans_discrepancies}
        )
        logger.warning(f"Answer verification failed ({ans_discrepancies}); falling back to factual template.")
        answer = response_generator.generate_grounded_fallback(
            clean_q,
            final_query.operation,
            retrieved_results,
            aggregation_result,
            final_query.missing_column,
            final_query.missing_entity,
            exec_result.get("explanation") or final_query.explanation
        )

    elapsed = round(time.perf_counter() - start_time, 4)

    # 7. Record Conversation Turn & Audit Trail
    conversation_manager.add_turn(
        session_id=payload.session_id,
        question=clean_q,
        query=final_query,
        result_count=result_count,
        results=retrieved_results,
        answer=answer
    )
    log_query(
        dataset_name=dataset_loader.dataset_name,
        question=clean_q,
        operation=final_query.operation,
        result_count=result_count,
        processing_time=elapsed,
        status="success" if final_query.operation not in {"UNSUPPORTED_QUERY", "UNKNOWN"} else "out_of_scope",
        session_id=payload.session_id
    )

    # 8. Assemble Source Transparency & Debug Trace
    cols_used = [c.column for c in final_query.conditions]
    if final_query.target_column:
        cols_used.append(final_query.target_column)

    source_transparency = {
        "dataset": dataset_loader.dataset_name,
        "rows_matched": result_count,
        "columns_used": sorted(list(set(cols_used))),
        "operation": final_query.operation,
        "sheet": dataset_loader.active_sheet or "Default"
    }

    debug_trace["executed_operation"] = final_query.operation
    debug_trace["result_count"] = result_count
    debug_trace["verified"] = is_ans_valid

    chart_spec = dataset_analytics.generate_chart_spec(
        operation=final_query.operation,
        results=retrieved_results,
        aggregation=aggregation_result,
        columns=dataset_loader.get_columns()
    )
    evidence = dataset_analytics.build_evidence_payload(
        operation=final_query.operation,
        results=retrieved_results,
        columns_referenced=cols_used,
        aggregation=aggregation_result,
        dataset_name=dataset_loader.dataset_name
    )

    return QueryResponse(
        question=clean_q,
        operation=final_query.operation,
        conditions=[c.model_dump() for c in final_query.conditions],
        results=retrieved_results,
        aggregation=aggregation_result,
        result_count=result_count,
        answer=answer,
        dataset_name=dataset_loader.dataset_name,
        processing_time=elapsed,
        grounded=True,
        is_clarification=False,
        source_type="DATASET",
        dataset_used=True,
        general_knowledge_used=False,
        confidence=float(debug_trace.get("confidence", 1.0)),
        debug_trace=debug_trace,
        source_transparency=source_transparency,
        evidence=evidence,
        chart_spec=chart_spec,
        sentiment=user_sentiment,
        sentiment_score=user_sentiment_score,
        hierarchy_scope="MAIN_ONLY",
        provenance={
            "dataset_id": dataset_loader.dataset_name,
            "dataset_version": dataset_loader.dataset_hash,
            "source_dataset": dataset_loader.dataset_name,
            "columns_used": cols_used,
            "filters_applied": [c.model_dump() for c in final_query.conditions],
            "operations": [final_query.operation],
            "calculation_formula": aggregation_result.get("metric") if aggregation_result else None,
            "calculation_result": aggregation_result.get("value") if aggregation_result else None,
        },
        normalized_question=normalized_q
    )


@router.get("/failure-analysis")
def get_failure_analysis() -> Dict[str, Any]:
    """Retrieve structured telemetry and category breakdown of query execution failures."""
    return failure_logger.get_summary()


@router.get("/dataset/fingerprint")
def get_dataset_fingerprint() -> Dict[str, Any]:
    """Return dataset ID, schema hash, and version fingerprint of active dataset."""
    prof = dataset_loader.profile
    if not prof:
        raise HTTPException(status_code=404, detail="No active dataset loaded.")
    return prof.get("fingerprint", {})


@router.get("/query/history")
def get_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieve query history audit trail."""
    return get_query_history(limit=limit)


# =========================================================================
# Active Learning Feedback & Human Correction Loop (Sections 47, 48, 60)
# =========================================================================

class FeedbackRequest(BaseModel):
    question: str
    is_correct: bool
    operation: Optional[str] = None
    answer: Optional[str] = None
    user_correction: Optional[Dict[str, Any]] = None


@router.post("/feedback")
def submit_feedback(payload: FeedbackRequest) -> Dict[str, Any]:
    """Record customer feedback (Thumb Up / Down) with ground truth correction for active learning."""
    from app.training.feedback import feedback_manager
    entry = feedback_manager.record_feedback(
        question=payload.question,
        is_correct=payload.is_correct,
        system_operation=payload.operation,
        system_answer=payload.answer,
        user_correction=payload.user_correction
    )
    return {"status": "success", "feedback": entry}


@router.get("/training/feedback/stats")
def get_feedback_stats() -> Dict[str, Any]:
    """Get active learning feedback metrics and review queue stats."""
    from app.training.feedback import feedback_manager
    return feedback_manager.get_stats()


# =========================================================================
# Statistical Analytics & Outlier Detection (Sections 24 & 25)
# =========================================================================

@router.get("/dataset/analytics/{column_name}")
def get_column_analytics(column_name: str) -> Dict[str, Any]:
    """Compute rigorous statistics (mean, median, mode, IQR, std, percentiles) for a column."""
    from app.query.analytics import dataset_analytics
    df = dataset_loader.dataframe
    if df.empty:
        raise HTTPException(status_code=400, detail="No active dataset loaded.")
    if column_name not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column_name}' not found.")
    raw_col = df[column_name]
    col_series: pd.Series = raw_col.iloc[:, 0] if isinstance(raw_col, pd.DataFrame) else raw_col
    return dataset_analytics.calculate_statistics(col_series, column_name)


@router.get("/dataset/outliers/{column_name}")
def get_column_outliers(column_name: str, method: str = "iqr", threshold: float = 1.5) -> Dict[str, Any]:
    """Detect statistical outliers deterministically via IQR or z-score."""
    from app.query.analytics import dataset_analytics
    df = dataset_loader.dataframe
    if df.empty:
        raise HTTPException(status_code=400, detail="No active dataset loaded.")
    if column_name not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column_name}' not found.")

    if method.lower() == "zscore" or method.lower() == "z-score":
        return dataset_analytics.detect_outliers_zscore(df, column_name, threshold=threshold if threshold != 1.5 else 2.5)
    return dataset_analytics.detect_outliers_iqr(df, column_name, k=threshold)


# =========================================================================
# Synthetic Training Data Generation & Evaluation (Sections 33, 52, 54)
# =========================================================================

@router.post("/training/generate")
def generate_training_data(max_samples: int = 100) -> Dict[str, Any]:
    """Synthesize diverse training pairs with linguistic augmentations from active dataset."""
    from app.training.generator import training_generator
    df = dataset_loader.dataframe
    if df.empty:
        raise HTTPException(status_code=400, detail="No active dataset loaded.")
    samples = training_generator.generate_from_dataframe(df, dataset_loader.dataset_name, max_samples=max_samples)
    return {
        "status": "success",
        "dataset": dataset_loader.dataset_name,
        "sample_count": len(samples),
        "samples": samples
    }

