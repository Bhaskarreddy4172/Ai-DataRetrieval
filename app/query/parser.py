"""Translates natural language questions into StructuredQuery using Ollama or semantic fallback."""

from typing import Any, Dict, List, Optional
from app.ai.ollama_client import ollama_client, OllamaClient
from app.ai.prompts import build_query_understanding_prompt
from app.query.planner import query_planner
from app.query.schema import StructuredQuery
from app.query.validator import query_validator
from app.utils.fuzzy_match import expand_abbreviations
from app.utils.logger import logger
from app.utils.validators import safe_json_extract, sanitize_text


class QueryParser:
    """Parses natural language into validated StructuredQuery with dynamic schema awareness and fuzzy matching."""

    def __init__(self, ai_client: OllamaClient = ollama_client):
        self.client = ai_client

    def parse(
        self,
        question: str,
        available_columns: List[str],
        schema: Dict[str, Any],
        recent_history: Optional[List[Dict[str, Any]]] = None
    ) -> StructuredQuery:
        clean_q = sanitize_text(question)
        expanded_q = expand_abbreviations(clean_q)

        # Universal Dataset-Aware Spell Checker & Typo Resolution
        try:
            from app.dataset.spell_checker import dataset_spell_checker
            corr = dataset_spell_checker.correct_full_question(expanded_q)
            if corr and corr.get("corrected_question"):
                expanded_q = corr["corrected_question"]
        except Exception:
            pass

        # 1. Try Ollama if available
        health = self.client.check_health()
        if health.get("available"):
            prompt = build_query_understanding_prompt(expanded_q, schema, recent_history)
            raw_response = self.client.generate(prompt=prompt, json_format=True)
            if raw_response:
                parsed_json = safe_json_extract(raw_response)
                if parsed_json and "operation" in parsed_json:
                    try:
                        structured = StructuredQuery(**parsed_json)
                        is_valid, norm_query, _ = query_validator.validate_and_normalize(structured, available_columns)
                        if is_valid:
                            norm_query.debug_trace = {
                                "original_question": question,
                                "normalized_question": clean_q,
                                "intent": norm_query.operation,
                                "source": "ollama",
                                "confidence": 0.95
                            }
                            logger.info(f"Ollama structured query: {norm_query.model_dump()}")
                            return norm_query
                    except Exception as err:
                        logger.warning(f"Error validating Ollama response: {err}")

        # 2. High-speed deterministic QueryPlanner
        planned_query = query_planner.plan_query(expanded_q, available_columns, schema)
        trace = planned_query.debug_trace
        _, norm_query, _ = query_validator.validate_and_normalize(planned_query, available_columns)
        norm_query.debug_trace = trace
        return norm_query


query_parser = QueryParser()
