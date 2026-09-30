"""Failure Analyzer and Continuous Improvement Engine (Section 44, 45).

Groups failures by exact required categories:
- Spelling
- Phonetic
- Abbreviation
- Entity
- Column
- Intent
- Operator
- Relationship
- Query Plan
- Execution
- Conversation
- Ambiguity
- Answer Generation
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


class RootCauseFailureAnalyzer:
    """Classifies, aggregates, and reports on failure taxonomy with remediation guidance."""

    CATEGORIES = [
        "Spelling",
        "Phonetic",
        "Abbreviation",
        "Entity",
        "Column",
        "Intent",
        "Operator",
        "Relationship",
        "Query Plan",
        "Execution",
        "Conversation",
        "Ambiguity",
        "Answer Generation"
    ]

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or (Path(__file__).resolve().parent / "logs" / "query_failures.json")
        self.failures: List[Dict[str, Any]] = []

    def classify(self, failure_item: Dict[str, Any]) -> str:
        """Classify a failure record into one of the 13 required categories."""
        q = str(failure_item.get("question", "")).lower()
        reason = str(failure_item.get("reason", failure_item.get("reasons", ""))).lower()
        err_type = str(failure_item.get("error_type", failure_item.get("category", ""))).lower()
        exp = str(failure_item.get("expected", failure_item.get("expected_intent", "")))
        act = str(failure_item.get("actual", failure_item.get("predicted_intent", "")))

        if "spell" in err_type or "typo" in err_type or "edit_distance" in reason:
            return "Spelling"
        if "phonetic" in err_type or "soundex" in reason or "metaphone" in reason:
            return "Phonetic"
        if "abbr" in err_type or "shortcut" in reason or "code" in reason:
            return "Abbreviation"
        if "entity" in err_type or "entity_not_found" in err_type or "missing_entity" in reason:
            return "Entity"
        if "column" in err_type or "column_not_found" in err_type or "unmapped_column" in err_type or "target_column" in reason:
            return "Column"
        if "intent" in err_type or (exp and act and exp != act):
            return "Intent"
        if "operator" in err_type or "comparison" in reason or "filter_op" in reason:
            return "Operator"
        if "relationship" in err_type or "possessive" in reason or "reverse_lookup" in reason:
            return "Relationship"
        if "plan" in err_type or "query_plan" in reason:
            return "Query Plan"
        if "exec" in err_type or "timeout" in reason or "duckdb" in reason or "sql" in reason:
            return "Execution"
        if "convers" in err_type or "followup" in reason or "turn" in reason or "pronoun" in reason:
            return "Conversation"
        if "ambig" in err_type or "clarification" in reason:
            return "Ambiguity"
        if "answer" in err_type or "hallucinat" in reason or "grounding" in reason:
            return "Answer Generation"

        return "Execution"

    def record_failure(
        self,
        question: str,
        expected_intent: str,
        predicted_intent: str,
        expected_entity: Optional[str] = None,
        predicted_entity: Optional[str] = None,
        expected_column: Optional[str] = None,
        error_type: Optional[str] = None,
        correction: Optional[str] = None,
        verified: bool = False
    ) -> Dict[str, Any]:
        """Record and format a failure entry adhering to Section 44 schema."""
        raw_record = {
            "question": question,
            "expected_intent": expected_intent,
            "predicted_intent": predicted_intent,
            "expected_entity": expected_entity,
            "predicted_entity": predicted_entity,
            "expected_column": expected_column,
            "error_type": error_type or "UNKNOWN",
            "correction": correction,
            "verified": verified
        }
        category = self.classify(raw_record)
        raw_record["grouped_category"] = category
        self.failures.append(raw_record)
        return raw_record

    def summarize_failures(self) -> Dict[str, Any]:
        """Aggregate failures by category with counts and percentages."""
        breakdown = {cat: 0 for cat in self.CATEGORIES}
        for f in self.failures:
            cat = f.get("grouped_category", "Execution")
            if cat in breakdown:
                breakdown[cat] += 1
            else:
                breakdown["Execution"] += 1

        total = len(self.failures)
        return {
            "total_failures": total,
            "category_counts": breakdown,
            "category_percentages": {k: round(v / total * 100, 2) if total > 0 else 0.0 for k, v in breakdown.items()}
        }

    def generate_report(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        summary = self.summarize_failures()
        report = {
            "summary": summary,
            "detailed_failures": self.failures
        }
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        return report


failure_analyzer = RootCauseFailureAnalyzer()


if __name__ == "__main__":
    # Self-test
    analyzer = RootCauseFailureAnalyzer()
    analyzer.record_failure(
        question="wat is ts captal",
        expected_intent="LOOKUP",
        predicted_intent="FILTER",
        expected_entity="Telangana",
        predicted_entity=None,
        expected_column="Capital",
        error_type="ENTITY_RESOLUTION",
        correction="TS -> Telangana, captal -> capital",
        verified=False
    )
    res = analyzer.summarize_failures()
    print("Failure Analyzer Self-Test:")
    print(json.dumps(res, indent=2))

