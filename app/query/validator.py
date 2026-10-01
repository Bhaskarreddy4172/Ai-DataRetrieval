"""Validates and normalizes StructuredQuery against the active dataset schema."""

from typing import List, Optional, Tuple
import pandas as pd
from app.dataset.metadata import DatasetMetadata
from app.query.schema import StructuredQuery
from app.utils.logger import logger


class QueryValidator:
    """Ensures queries reference valid columns, operators, and types before execution."""

    VALID_OPERATORS = {
        "=", "==", "!=", "<>", ">", "<", ">=", "<=",
        "contains", "not_contains", "starts_with", "ends_with",
        "in", "not_in", "between"
    }

    def validate_and_normalize(self, query: StructuredQuery, available_columns: List[str]) -> Tuple[bool, StructuredQuery, str]:
        """Normalize column names and ensure query safety."""
        if query.operation in {"UNKNOWN", "AMBIGUOUS"}:
            return True, query, "Special intent"

        # Check target_column
        if query.target_column:
            matched = DatasetMetadata.match_column(query.target_column, available_columns)
            if matched:
                query.target_column = matched
            else:
                query.operation = "UNKNOWN"
                query.missing_column = query.target_column
                return False, query, f"The dataset does not contain a '{query.target_column}' column."

        # Check group_by_column
        if query.group_by_column:
            matched = DatasetMetadata.match_column(query.group_by_column, available_columns)
            if matched:
                query.group_by_column = matched
            else:
                query.operation = "UNKNOWN"
                query.missing_column = query.group_by_column
                return False, query, f"The dataset does not contain a '{query.group_by_column}' column."

        # Check sort_column
        if query.sort_column:
            matched = DatasetMetadata.match_column(query.sort_column, available_columns)
            if matched:
                query.sort_column = matched
            else:
                query.sort_column = None

        # Check select_columns
        if query.select_columns:
            valid_select = []
            for sc in query.select_columns:
                matched = DatasetMetadata.match_column(sc, available_columns)
                if matched:
                    valid_select.append(matched)
            query.select_columns = valid_select or None

        # Validate and normalize conditions
        valid_conditions = []
        for cond in query.conditions:
            matched_col = DatasetMetadata.match_column(cond.column, available_columns)
            if not matched_col:
                # Column does not exist in dataset -> Anti-hallucination trigger!
                query.operation = "UNKNOWN"
                query.missing_column = cond.column
                return False, query, f"The dataset does not contain a '{cond.column}' column."

            cond.column = matched_col

            # Normalize operator
            op = cond.operator.strip().lower()
            if op == "==":
                op = "="
            elif op == "<>":
                op = "!="
            if op not in self.VALID_OPERATORS:
                op = "="
            cond.operator = op
            valid_conditions.append(cond)

        query.conditions = valid_conditions
        return True, query, "Valid query"


query_validator = QueryValidator()


class AnswerConsistencyValidator:
    """Validates that execution results and answers semantically satisfy the user's question before returning."""

    @staticmethod
    def validate_answer(
        question: str,
        operation: str,
        scope: str,
        answer: str,
        results: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str]]:
        """Validate answer consistency against question intent.

        Returns (is_valid, failure_reason).
        """
        q_lower = question.lower()
        import re
        from app.query.fast_classifier import fast_query_classifier

        # Rule 1: If question asks for state ranking / extreme, answer MUST NOT be a single-state total
        if fast_query_classifier.is_global_query(question):
            is_extreme_q = any(re.search(r"\b" + re.escape(w) + r"\b", q_lower) for w in [
                "highest", "lowest", "more", "less", "most", "least", "maximum", "minimum", "larger", "smaller"
            ])
            has_state_target = bool(re.search(r"\b(?:state|states|who)\b", q_lower))
            if is_extreme_q and has_state_target:
                if operation == "SUM" and scope == "SINGLE_STATE":
                    return False, "Question requires comparison across states, but single state SUM was returned."
                if "across all recorded villages in" in answer and "total" in answer.lower() and not any(w in answer.lower() for w in ["highest", "lowest", "most", "least"]):
                    return False, "Answer failed to identify extreme state."

        # Rule 2: If question asks for comparison, must not be a direct single-value lookup
        if any(w in q_lower for w in ["compare", "difference between", "how much more", "how much less"]):
            if operation in ["LOOKUP"] and len(results) == 1 and not results[0].get("left_entity"):
                return False, "Question requires comparison, but single lookup returned."

        return True, None


answer_consistency_validator = AnswerConsistencyValidator()
