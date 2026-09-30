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
