"""Multi-Question Decomposition Engine for compound natural language dataset inquiries.

Decomposes complex inquiries containing multiple sub-intents or questions joined by
conjunctions (e.g. 'Who earns > 10L in Hyderabad and what department are they in?')
while strictly protecting single inquiries with multiple conditions (e.g. 'in IT and salary > 50k').
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from app.query.schema import StructuredQuery
from app.query.parser import query_parser
from app.query.executor import query_executor
from app.ai.response_generator import response_generator
from app.utils.logger import logger


SPLIT_CONJUNCTION_PATTERN = re.compile(
    r"(?:;\s*|\?\s*|\b(?:and|also|as well as|additionally)\s+(?=(?:what|who|where|which|how|tell me|show|list|find|give me|is|are|can you)\b))",
    re.IGNORECASE
)

PRONOUN_PATTERN = re.compile(r"\b(?:they|them|their|theirs|those|these|he|she|his|her|that person|those people)\b", re.IGNORECASE)


class MultiQuestionDecomposer:
    """Detects, splits, and executes compound dataset queries sequentially with result chaining."""

    def is_compound_question(self, question: str) -> bool:
        """Determine if a query consists of multiple distinct questions or sub-intents."""
        if not question:
            return False

        sub_qs = self.split_question(question)
        return len(sub_qs) > 1

    def split_question(self, question: str) -> List[str]:
        """Split a compound question into atomic sub-questions while preserving conditions."""
        if not question:
            return []

        q = question.strip()

        # Split on questions separated by question marks or semicolons first
        initial_parts = [p.strip() for p in re.split(r"[?;]\s*", q) if p.strip()]

        atomic_parts: List[str] = []
        for part in initial_parts:
            # Check for conjunction introducing a question clause
            split_sub = SPLIT_CONJUNCTION_PATTERN.split(part)
            for s in split_sub:
                s_clean = s.strip()
                # Clean leading conjunctions if any left
                s_clean = re.sub(r"^(?:and|also|as well as|additionally)\s+", "", s_clean, flags=re.IGNORECASE).strip()
                if s_clean and len(s_clean) >= 3:
                    atomic_parts.append(s_clean)

        return atomic_parts if len(atomic_parts) > 1 else [q]

    def execute_decomposed(
        self,
        question: str,
        df: pd.DataFrame,
        available_cols: List[str],
        schema_info: Dict[str, Any],
        recent_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Execute each decomposed sub-query with entity/row chaining and synthesize a composite answer."""
        sub_questions = self.split_question(question)
        if len(sub_questions) <= 1:
            # Single question execution
            parsed = query_parser.parse(question, available_cols, schema_info, recent_history)
            exec_res = query_executor.execute(parsed, df)
            ans = response_generator.generate(
                question=question,
                operation=parsed.operation,
                results=exec_res.get("results", []),
                aggregation=exec_res.get("aggregation"),
                missing_column=parsed.missing_column
            )
            return {
                "is_compound": False,
                "sub_questions": [question],
                "sub_results": [exec_res.get("results", [])],
                "sub_answers": [ans],
                "combined_results": exec_res.get("results", []),
                "answer": ans,
                "operation": parsed.operation,
                "result_count": exec_res.get("result_count", len(exec_res.get("results", []))),
                "aggregation": exec_res.get("aggregation"),
                "conditions": [c.model_dump() for c in parsed.conditions],
                "debug_trace": parsed.debug_trace or {}
            }

        logger.info(f"Decomposing compound question into {len(sub_questions)} sub-questions: {sub_questions}")

        sub_executions: List[Dict[str, Any]] = []
        sub_answers: List[str] = []
        all_results: List[Dict[str, Any]] = []
        all_conditions: List[Dict[str, Any]] = []
        operations: List[str] = []

        current_df = df
        chained_row_ids: Optional[List[int]] = None

        for idx, sub_q in enumerate(sub_questions):
            # Check if this sub-question references previous rows via pronouns
            has_pronoun = bool(PRONOUN_PATTERN.search(sub_q))
            sub_working_df = current_df

            if has_pronoun and chained_row_ids is not None and len(chained_row_ids) > 0:
                # Scope query execution to the rows matching previous sub-query
                scoped_df = df[df["_internal_row_id"].isin(chained_row_ids)]
                if not scoped_df.empty:
                    sub_working_df = scoped_df

            # Parse sub-query
            parsed = query_parser.parse(sub_q, available_cols, schema_info, recent_history)
            operations.append(parsed.operation)
            for c in parsed.conditions:
                all_conditions.append(c.model_dump())

            # Execute sub-query
            exec_res = query_executor.execute(parsed, sub_working_df)
            res_list = exec_res.get("results", [])
            sub_executions.append(exec_res)

            # Update chained row ids if results were returned with internal IDs
            if res_list and any("_internal_row_id" in r for r in res_list):
                chained_row_ids = [r["_internal_row_id"] for r in res_list if "_internal_row_id" in r]

            # Generate individual sub-answer
            sub_ans = response_generator.generate(
                question=sub_q,
                operation=parsed.operation,
                results=res_list,
                aggregation=exec_res.get("aggregation"),
                missing_column=parsed.missing_column
            )
            sub_answers.append(sub_ans)

            for r in res_list:
                if r not in all_results:
                    all_results.append(r)

        # Synthesize clear, structured composite answer
        answer_parts = []
        for i, (sq, ans) in enumerate(zip(sub_questions, sub_answers), start=1):
            answer_parts.append(f"{i}. {ans}")

        combined_answer = "\n".join(answer_parts)

        return {
            "is_compound": True,
            "sub_questions": sub_questions,
            "sub_results": [e.get("results", []) for e in sub_executions],
            "sub_answers": sub_answers,
            "combined_results": all_results,
            "answer": combined_answer,
            "operations": operations,
            "result_count": len(all_results),
            "aggregation": sub_executions[-1].get("aggregation") if sub_executions else None,
            "conditions": all_conditions,
            "debug_trace": {
                "sub_questions": sub_questions,
                "operations": operations,
                "chained_row_count": len(chained_row_ids) if chained_row_ids else 0
            }
        }


multi_question_decomposer = MultiQuestionDecomposer()

