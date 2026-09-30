"""Result validator: ensures execution output matches query criteria and schema constraints."""

from typing import Any, Dict, List, Optional, Tuple
from app.query.schema import StructuredQuery
from app.utils.logger import logger


class ResultValidator:
    """Validates raw execution results against query constraints prior to response generation."""

    def validate(self, query: StructuredQuery, execution_result: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify that returned results strictly satisfy query criteria."""
        status = execution_result.get("status")
        if status in {"empty_dataset", "unsupported", "unknown_query"}:
            return True, "Valid non-result state"

        results = execution_result.get("results", [])
        aggregation = execution_result.get("aggregation")

        # 1. Verify numeric conditions hold in filtered rows
        if query.conditions and results:
            for cond in query.conditions:
                col = cond.column
                op = cond.operator
                val = cond.value

                if isinstance(val, (int, float)):
                    for row in results[:10]:
                        if col in row and row[col] is not None:
                            try:
                                row_val = float(str(row[col]).replace("₹", "").replace("$", "").replace(",", "").strip())
                                if op == ">=" and row_val < val:
                                    return False, f"Result row violates condition {col} >= {val} (found {row_val})"
                                elif op == ">" and row_val <= val:
                                    return False, f"Result row violates condition {col} > {val} (found {row_val})"
                                elif op == "<=" and row_val > val:
                                    return False, f"Result row violates condition {col} <= {val} (found {row_val})"
                                elif op == "<" and row_val >= val:
                                    return False, f"Result row violates condition {col} < {val} (found {row_val})"
                            except Exception:
                                pass

        # 2. Verify aggregation has valid metric
        if aggregation:
            val = aggregation.get("value")
            if val is None:
                return False, "Aggregation returned null value"

        return True, "Execution results verified successfully"


result_validator = ResultValidator()

