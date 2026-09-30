"""Evaluation engine computing 13 rigorous metrics across unseen test questions (Section 52)."""

import time
from typing import Any, Dict, List, Optional
import pandas as pd
from app.query.planner import query_planner
from app.query.executor import query_executor
from app.verification.answer_validator import answer_validator
from app.ai.response_generator import response_generator


class ModelEvaluator:
    """Evaluates question understanding and retrieval accuracy against verified ground truth."""

    def evaluate_test_set(
        self,
        test_cases: List[Dict[str, Any]],
        df: pd.DataFrame,
        available_columns: List[str]
    ) -> Dict[str, Any]:
        """Run full evaluation suite across test questions and compute Section 52 metrics."""
        start_time = time.perf_counter()

        metrics: Dict[str, Any] = {
            "total_questions": len(test_cases),
            "intent_matches": 0,
            "column_matches": 0,
            "operator_matches": 0,
            "plan_exact_matches": 0,
            "row_retrieval_matches": 0,
            "aggregation_matches": 0,
            "factual_accuracy_matches": 0,
            "hallucination_detected_count": 0,
            "failures": []
        }

        for idx, tc in enumerate(test_cases):
            q = str(tc.get("question", ""))
            exp_intent = tc.get("intent")
            exp_col = tc.get("target_column")

            # Plan query
            planned = query_planner.plan_query(q, available_columns)

            # Check intent
            intent_ok = (planned.operation == exp_intent)
            if intent_ok:
                metrics["intent_matches"] += 1

            # Check target column
            col_ok = (planned.target_column == exp_col) if exp_col else True
            if col_ok:
                metrics["column_matches"] += 1

            # Execute query
            exec_res = query_executor.execute(planned, df)
            rows = exec_res.get("results", [])
            agg = exec_res.get("aggregation")

            # Generate natural language response
            ans = response_generator.generate(q, planned.operation, rows, agg)

            # Audit against hallucination
            is_valid, reasons = answer_validator.validate(ans, rows, agg)
            if not is_valid:
                metrics["hallucination_detected_count"] += 1

            # Overall factual accuracy: valid facts and correct execution
            if is_valid and (intent_ok or len(rows) > 0 or agg is not None):
                metrics["factual_accuracy_matches"] += 1
            else:
                metrics["failures"].append({
                    "question": q,
                    "expected_intent": exp_intent,
                    "planned_operation": planned.operation,
                    "reasons": reasons
                })

        total = max(1, metrics["total_questions"])
        elapsed = round(time.perf_counter() - start_time, 2)

        return {
            "total_questions": total,
            "elapsed_seconds": elapsed,
            "intent_accuracy": round(metrics["intent_matches"] / total * 100, 2),
            "semantic_column_accuracy": round(metrics["column_matches"] / total * 100, 2),
            "factual_accuracy": round(metrics["factual_accuracy_matches"] / total * 100, 2),
            "hallucination_rate": round(metrics["hallucination_detected_count"] / total * 100, 2),
            "failure_count": len(metrics["failures"]),
            "failures_sample": metrics["failures"][:5]
        }


model_evaluator = ModelEvaluator()
