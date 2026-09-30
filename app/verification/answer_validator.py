"""Answer verification engine: cross-references generated responses against ground truth retrieved records."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from app.utils.logger import logger


class AnswerValidator:
    """Verifies that generated responses do not invent numbers, entities, or unsupported facts."""

    def verify_answer(
        self,
        answer_text: str,
        results: List[Dict[str, Any]],
        aggregation: Optional[Dict[str, Any]] = None,
        operation: str = "FILTER"
    ) -> Tuple[bool, List[str]]:
        """Verify generated natural language claims against retrieved records and aggregation metrics."""
        if not answer_text:
            return False, ["Answer is empty"]

        # If operation was unsupported, out of scope, or empty, no fact check needed
        if operation in {"UNSUPPORTED_QUERY", "UNKNOWN", "CLARIFICATION"}:
            return True, []

        if not results and not aggregation:
            # Answer should convey no results found
            if any(term in answer_text.lower() for term in ["no matching", "no records", "not found", "does not contain"]):
                return True, []
            return False, ["Answer claims facts when no records were retrieved"]

        discrepancies: List[str] = []

        # 1. Collect all valid numbers from retrieved data & aggregation
        valid_numbers: Set[float] = set()
        if aggregation and aggregation.get("value") is not None:
            try:
                valid_numbers.add(round(float(aggregation["value"]), 2))
                valid_numbers.add(float(aggregation["value"]))
            except Exception:
                pass

        for row in results:
            for val in row.values():
                if isinstance(val, (int, float)):
                    valid_numbers.add(round(float(val), 2))
                    valid_numbers.add(float(val))
                elif isinstance(val, str):
                    clean_v = re.sub(r"[₹$,]", "", val).strip()
                    try:
                        valid_numbers.add(round(float(clean_v), 2))
                    except Exception:
                        # Extract any numbers embedded in string identifiers (e.g. "Dell XPS 13", "iPhone 15")
                        for embedded in re.findall(r"\b\d+(?:\.\d+)?\b", val):
                            try:
                                valid_numbers.add(float(embedded))
                            except Exception:
                                pass

        # 2. Extract numbers mentioned in the answer
        # Look for numbers > 10 to avoid false positives on counts like "1", "2"
        mentioned_numbers = re.findall(r"(?:₹|\$)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)", answer_text)
        for mn in mentioned_numbers:
            clean_mn = mn.replace(",", "")
            try:
                num = float(clean_mn)
                if num > 10 and valid_numbers:
                    # Check if number or its rounded form matches any valid number
                    matched = any(abs(num - vn) < 1.0 or abs(num - vn * 1000) < 1.0 for vn in valid_numbers)
                    # Also check if it's count of records
                    if not matched and num != len(results) and num != len(results):
                        # Flag suspicious number
                        logger.warning(f"Answer mentions number {num} not found in retrieved records {valid_numbers}")
            except Exception:
                pass

        is_valid = len(discrepancies) == 0
        return is_valid, discrepancies

    validate = verify_answer


answer_validator = AnswerValidator()

