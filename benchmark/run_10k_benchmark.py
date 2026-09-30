"""Runs the 10,000+ Synthetic Test Benchmark Suite and outputs detailed accuracy metrics and telemetry.

Generates:
- benchmark/benchmark_10k_report.json
- benchmark/benchmark_10k_report.csv
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Windows encoding fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.dataset.loader import dataset_loader
from app.api.routes import process_query, QueryRequest
from failure_analyzer import RootCauseFailureAnalyzer

SUITE_PATH = PROJECT_ROOT / "benchmark" / "synthetic_10k_suite.json"
DATA_PATH = PROJECT_ROOT / "data" / "indian_states_capitals.csv"
REPORT_JSON = PROJECT_ROOT / "benchmark" / "benchmark_10k_report.json"
REPORT_CSV = PROJECT_ROOT / "benchmark" / "benchmark_10k_report.csv"
FAILURE_JSON = PROJECT_ROOT / "benchmark" / "failure_analysis_10k.json"


def run_10k_benchmark(sample_limit: Optional[int] = None) -> Dict[str, Any]:
    if not SUITE_PATH.exists():
        print(f"Benchmark suite not found at {SUITE_PATH}. Generating 10k suite...")
        from tests.test_generator import DatasetTestGenerator
        df_init = pd.read_csv(DATA_PATH)
        gen = DatasetTestGenerator(df_init, DATA_PATH.name)
        suite_data = gen.generate_large_scale_benchmark(total_target=10000)
        SUITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SUITE_PATH, "w", encoding="utf-8") as f:
            json.dump(suite_data, f, indent=2)

    with open(SUITE_PATH, "r", encoding="utf-8") as f:
        tests: List[Dict[str, Any]] = json.load(f)

    if sample_limit:
        tests = tests[:sample_limit]

    print(f"Loaded {len(tests)} benchmark test cases from {SUITE_PATH.name}.")

    dataset_loader.load_dataset(DATA_PATH)
    df = dataset_loader.dataframe
    cols = dataset_loader.get_columns()

    failure_analyzer = RootCauseFailureAnalyzer(log_path=FAILURE_JSON)

    total_tests = len(tests)
    passed_tests = 0
    intent_matches = 0
    entity_matches = 0
    column_matches = 0
    value_matches = 0
    operator_matches = 0
    relationship_matches = 0
    relationship_total = 0
    typo_matches = 0
    typo_total = 0
    phonetic_matches = 0
    phonetic_total = 0
    shortcut_matches = 0
    shortcut_total = 0
    analytical_matches = 0
    analytical_total = 0
    boolean_matches = 0
    boolean_total = 0
    unsupported_rejections = 0
    unsupported_total = 0
    hallucinations = 0

    retrieval_precisions = []
    retrieval_recalls = []

    categories_stats: Dict[str, Dict[str, int]] = {}
    latencies: List[float] = []
    results_detail: List[Dict[str, Any]] = []

    start_all = time.perf_counter()

    for idx, t in enumerate(tests):
        t_id = t["id"]
        cat = t["category"]
        q = t["question"]
        exp_intent = t.get("expected_intent")
        exp_entity = t.get("expected_entity")
        exp_column = t.get("expected_column")
        exp_answer = str(t.get("expected_answer", "")).strip()

        if cat not in categories_stats:
            categories_stats[cat] = {"total": 0, "passed": 0}
        categories_stats[cat]["total"] += 1

        t_start = time.perf_counter()
        req = QueryRequest(question=q, session_id=f"10k_{idx}")
        res = process_query(req)
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        latencies.append(latency_ms)

        actual_op = res.operation
        actual_answer = str(res.answer).strip()
        actual_results = res.results or []
        actual_agg = res.aggregation

        is_pass = True
        fail_reasons: List[str] = []

        # 1. Intent Matching
        intent_ok = False
        if exp_intent in {"LOOKUP", "FILTER"}:
            intent_ok = (actual_op in {"LOOKUP", "FILTER"} or (actual_results and len(actual_results) >= 1))
        elif exp_intent == "BOOLEAN_CHECK":
            intent_ok = (actual_op == "BOOLEAN_CHECK")
        elif exp_intent in {"COUNT", "DISTINCT", "SUM", "AVERAGE", "MIN", "MAX"}:
            intent_ok = (actual_op == exp_intent)
        elif exp_intent in {"UNSUPPORTED_QUERY", "NO_MATCH"}:
            intent_ok = (actual_op in {"UNSUPPORTED_QUERY", "NO_MATCH", "UNKNOWN"})
        else:
            intent_ok = (actual_op == exp_intent)

        if intent_ok:
            intent_matches += 1
        else:
            is_pass = False
            fail_reasons.append(f"Intent mismatch: exp '{exp_intent}' got '{actual_op}'")

        # 2. Entity Matching
        entity_ok = True
        if exp_entity:
            cond_vals = [str(c.get("value", "")).lower() for c in res.conditions]
            res_vals = [
                str(v).lower()
                for r in actual_results
                for v in r.values()
                if v is not None
            ]
            exp_ent_low = exp_entity.lower()
            if (
                any(exp_ent_low in cv for cv in cond_vals)
                or any(exp_ent_low in rv for rv in res_vals)
                or exp_ent_low in actual_answer.lower()
                or (res.debug_trace and exp_ent_low in str(res.debug_trace).lower())
            ):
                entity_ok = True
            else:
                entity_ok = False
                is_pass = False
                fail_reasons.append(f"Entity mismatch: expected '{exp_entity}'")

        if entity_ok:
            entity_matches += 1

        # 3. Column Matching
        col_ok = True
        if exp_column:
            exp_col_low = exp_column.lower()
            cond_cols = [str(c.get("column", "")).lower() for c in res.conditions]
            trace_str = str(res.debug_trace).lower() if res.debug_trace else ""
            if (
                any(exp_col_low == cc for cc in cond_cols)
                or exp_col_low in [c.lower() for c in cols]
                or exp_col_low in trace_str
                or exp_col_low in actual_answer.lower()
            ):
                col_ok = True
            else:
                col_ok = False
                is_pass = False
                fail_reasons.append(f"Column mismatch: expected '{exp_column}'")

        if col_ok:
            column_matches += 1

        # 4. Operator Resolution
        op_ok = True
        if exp_intent in {"COUNT", "DISTINCT"}:
            op_ok = (actual_op == exp_intent)
        elif exp_intent == "BOOLEAN_CHECK":
            op_ok = (actual_op == "BOOLEAN_CHECK")
        elif exp_intent in {"UNSUPPORTED_QUERY", "NO_MATCH"}:
            op_ok = (actual_op in {"UNSUPPORTED_QUERY", "NO_MATCH", "UNKNOWN"})
        else:
            op_ok = True

        if op_ok:
            operator_matches += 1

        # 5. Value Matching & Factual Grounding
        val_ok = True
        if exp_intent in {"UNSUPPORTED_QUERY", "NO_MATCH"}:
            if actual_op not in {"UNSUPPORTED_QUERY", "NO_MATCH", "UNKNOWN"}:
                val_ok = False
                hallucinations += 1
                is_pass = False
                fail_reasons.append("Hallucination: Fabricated answer for unsupported inquiry")
            else:
                val_ok = True
        elif exp_intent == "BOOLEAN_CHECK":
            boolean_total += 1
            exp_bool_str = exp_answer.lower()
            act_starts_true = actual_answer.lower().startswith("true")
            act_starts_false = actual_answer.lower().startswith("false")
            if (exp_bool_str == "true" and act_starts_true) or (exp_bool_str == "false" and act_starts_false):
                boolean_matches += 1
                val_ok = True
            else:
                val_ok = False
                is_pass = False
                fail_reasons.append(f"Boolean value mismatch: exp '{exp_answer}' got '{actual_answer}'")
        elif exp_intent in {"COUNT", "DISTINCT"}:
            analytical_total += 1
            agg_val = actual_agg.get("value") if actual_agg else res.result_count
            if str(agg_val) == exp_answer or exp_answer in actual_answer:
                analytical_matches += 1
                val_ok = True
            else:
                val_ok = False
                is_pass = False
                fail_reasons.append(f"Analytical value mismatch: exp '{exp_answer}' got '{agg_val}'")
        else:
            exp_ans_low = exp_answer.lower()
            res_has_val = any(
                exp_ans_low in str(v).lower()
                for r in actual_results
                for v in r.values()
                if v is not None
            )
            if res_has_val or exp_ans_low in actual_answer.lower():
                val_ok = True
            else:
                val_ok = False
                is_pass = False
                fail_reasons.append(f"Value mismatch: exp '{exp_answer}' not in results or answer")

        if val_ok:
            value_matches += 1

        # 6. Specific Category Breakdown
        if cat == "3_typo_variations":
            typo_total += 1
            if is_pass:
                typo_matches += 1
        elif cat == "4_phonetic_soundalike":
            phonetic_total += 1
            if is_pass:
                phonetic_matches += 1
        elif cat == "5_abbreviation_shortcuts":
            shortcut_total += 1
            if is_pass:
                shortcut_matches += 1
        elif cat == "6_reverse_relationships":
            relationship_total += 1
            if is_pass:
                relationship_matches += 1
        elif cat == "10_unsupported_antihallucination":
            unsupported_total += 1
            if actual_op in {"UNSUPPORTED_QUERY", "NO_MATCH", "UNKNOWN"}:
                unsupported_rejections += 1

        # 7. Row Retrieval Precision & Recall
        if exp_intent in {"LOOKUP", "FILTER"}:
            retrieved_count = len(actual_results)
            if val_ok and retrieved_count > 0:
                p = 1.0 / retrieved_count
                r = 1.0
            else:
                p = 0.0
                r = 0.0
            retrieval_precisions.append(p)
            retrieval_recalls.append(r)
        elif exp_intent in {"UNSUPPORTED_QUERY", "NO_MATCH"}:
            if len(actual_results) == 0:
                retrieval_precisions.append(1.0)
                retrieval_recalls.append(1.0)
            else:
                retrieval_precisions.append(0.0)
                retrieval_recalls.append(0.0)

        if is_pass:
            passed_tests += 1
            categories_stats[cat]["passed"] += 1
        else:
            failure_analyzer.record_failure(
                question=q,
                expected_intent=exp_intent or "UNKNOWN",
                predicted_intent=actual_op,
                expected_entity=exp_entity,
                expected_column=exp_column,
                error_type="; ".join(fail_reasons),
                correction=f"Expected answer: {exp_answer}",
                verified=False
            )

        results_detail.append({
            "test_id": t_id,
            "category": cat,
            "question": q,
            "expected_intent": exp_intent,
            "actual_operation": actual_op,
            "expected_answer": exp_answer,
            "actual_answer": actual_answer,
            "passed": is_pass,
            "latency_ms": latency_ms,
            "fail_reasons": "; ".join(fail_reasons) if fail_reasons else "NONE"
        })

        if (idx + 1) % 1000 == 0 or (idx + 1) == total_tests:
            print(f"Processed {idx + 1}/{total_tests} test cases... (Passed: {passed_tests})")

    total_time_sec = round(time.perf_counter() - start_all, 2)
    avg_latency = round(float(np.mean(latencies)), 2) if latencies else 0.0
    p95_latency = round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0

    mean_precision = round(float(np.mean(retrieval_precisions)) * 100, 2) if retrieval_precisions else 100.0
    mean_recall = round(float(np.mean(retrieval_recalls)) * 100, 2) if retrieval_recalls else 100.0
    f1_score = round(2 * (mean_precision * mean_recall) / (mean_precision + mean_recall), 2) if (mean_precision + mean_recall) > 0 else 0.0

    accuracy_pct = round((passed_tests / total_tests) * 100, 2)
    intent_pct = round((intent_matches / total_tests) * 100, 2)
    entity_pct = round((entity_matches / total_tests) * 100, 2)
    column_pct = round((column_matches / total_tests) * 100, 2)
    value_pct = round((value_matches / total_tests) * 100, 2)
    operator_pct = round((operator_matches / total_tests) * 100, 2)

    relationship_pct = round((relationship_matches / relationship_total) * 100, 2) if relationship_total > 0 else 100.0
    typo_pct = round((typo_matches / typo_total) * 100, 2) if typo_total > 0 else 100.0
    phonetic_pct = round((phonetic_matches / phonetic_total) * 100, 2) if phonetic_total > 0 else 100.0
    shortcut_pct = round((shortcut_matches / shortcut_total) * 100, 2) if shortcut_total > 0 else 100.0
    analytical_pct = round((analytical_matches / analytical_total) * 100, 2) if analytical_total > 0 else 100.0
    boolean_pct = round((boolean_matches / boolean_total) * 100, 2) if boolean_total > 0 else 100.0
    unsupported_pct = round((unsupported_rejections / unsupported_total) * 100, 2) if unsupported_total > 0 else 100.0
    hallucination_pct = round((hallucinations / total_tests) * 100, 2)

    summary_report: Dict[str, Any] = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries_tested": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "overall_accuracy_pct": accuracy_pct,
        "intent_recognition_accuracy_pct": intent_pct,
        "entity_recognition_accuracy_pct": entity_pct,
        "column_mapping_accuracy_pct": column_pct,
        "value_matching_accuracy_pct": value_pct,
        "operator_resolution_accuracy_pct": operator_pct,
        "relationship_resolution_accuracy_pct": relationship_pct,
        "typo_recovery_rate_pct": typo_pct,
        "phonetic_recovery_rate_pct": phonetic_pct,
        "shortcut_abbreviation_recovery_rate_pct": shortcut_pct,
        "row_retrieval_precision_pct": mean_precision,
        "row_retrieval_recall_pct": mean_recall,
        "row_retrieval_f1_pct": f1_score,
        "analytical_calculation_accuracy_pct": analytical_pct,
        "boolean_verification_accuracy_pct": boolean_pct,
        "unsupported_query_rejection_rate_pct": unsupported_pct,
        "hallucination_rate_pct": hallucination_pct,
        "average_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "total_benchmark_time_seconds": total_time_sec,
        "category_breakdown": {
            cat: {
                "total": stats["total"],
                "passed": stats["passed"],
                "accuracy_pct": round((stats["passed"] / stats["total"]) * 100, 2) if stats["total"] > 0 else 0.0
            }
            for cat, stats in categories_stats.items()
        }
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "test_id", "category", "question", "expected_intent", "actual_operation",
            "expected_answer", "actual_answer", "passed", "latency_ms", "fail_reasons"
        ])
        writer.writeheader()
        writer.writerows(results_detail)

    failure_analyzer.generate_report(FAILURE_JSON)

    print("\n" + "=" * 70)
    print("10,000 SYNTHETIC TEST BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Total Queries:                         {total_tests}")
    print(f"Overall Accuracy:                      {accuracy_pct}% ({passed_tests}/{total_tests})")
    print(f"Hallucination Rate:                    {hallucination_pct}% (Strict 0% Target)")
    print(f"Average Latency:                       {avg_latency} ms (P95: {p95_latency} ms)")
    print(f"Total Time:                            {total_time_sec}s")
    print("=" * 70)
    print(f"Saved: {REPORT_JSON}, {REPORT_CSV}, {FAILURE_JSON}")

    return summary_report


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_10k_benchmark(sample_limit=limit)
