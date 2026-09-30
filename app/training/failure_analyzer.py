"""Automated failure analysis taxonomy and diagnostic reporter (Section 51).

Classifies every query failure into exact root-cause categories and generates
structured diagnostics, root causes, and actionable recommendations.
"""

from typing import Any, Dict, List, Optional


class FailureAnalyzer:
    """Categorizes failed questions into actionable 14-category taxonomy with remediation advice."""

    FAILURE_CATEGORIES = [
        "UNMAPPED_COLUMN",
        "INCORRECT_OPERATION",
        "VALUE_NOT_FOUND",
        "PHONETIC_MATCH_FAILED",
        "FILTER_PARSING_ERROR",
        "AGGREGATION_ERROR",
        "RANKING_ERROR",
        "COMPARISON_ERROR",
        "AMBIGUITY_DETECTION_ERROR",
        "TIMEOUT",
        "SCHEMA_MISMATCH",
        "NORMALIZATION_ERROR",
        "LLM_GROUNDING_VIOLATION",
        "OUT_OF_SCOPE_MISCLASSIFICATION",
        "OTHER",
    ]

    def classify_failure(self, failure: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a single failure record into Section 51 failure schema."""
        question = str(failure.get("question", ""))
        reasons = str(failure.get("reasons", failure.get("reason", ""))).lower()
        details = str(failure.get("details", "")).lower()
        expected = str(failure.get("expected", ""))
        actual = str(failure.get("actual", failure.get("operation", "")))
        cat = failure.get("category", "")

        # Heuristic classification
        if cat in self.FAILURE_CATEGORIES:
            category = cat
        elif "grounding" in reasons or "hallucinat" in reasons or "unverified" in reasons:
            category = "LLM_GROUNDING_VIOLATION"
        elif "column" in reasons or "unmapped" in reasons:
            category = "UNMAPPED_COLUMN"
        elif "phonetic" in reasons or "soundex" in reasons or "metaphone" in reasons:
            category = "PHONETIC_MATCH_FAILED"
        elif "not found" in reasons or "value" in reasons or "empty" in reasons:
            category = "VALUE_NOT_FOUND"
        elif "ambiguous" in reasons or "competing" in reasons:
            category = "AMBIGUITY_DETECTION_ERROR"
        elif "timeout" in reasons:
            category = "TIMEOUT"
        elif "schema" in reasons or "sheet" in reasons:
            category = "SCHEMA_MISMATCH"
        elif "rank" in reasons or "offset" in reasons:
            category = "RANKING_ERROR"
        elif "compare" in reasons or "comparison" in reasons:
            category = "COMPARISON_ERROR"
        elif "agg" in reasons or "sum" in reasons or "avg" in reasons or "average" in reasons:
            category = "AGGREGATION_ERROR"
        elif "filter" in reasons or "condition" in reasons or "operator" in reasons:
            category = "FILTER_PARSING_ERROR"
        elif "norm" in reasons:
            category = "NORMALIZATION_ERROR"
        elif "operation" in reasons or (expected and actual and expected != actual):
            category = "INCORRECT_OPERATION"
        elif "out_of_scope" in reasons or "general knowledge" in reasons:
            category = "OUT_OF_SCOPE_MISCLASSIFICATION"
        else:
            category = "OTHER"

        # Generate root cause and suggested fix
        fixes = {
            "UNMAPPED_COLUMN": ("Column name/synonym not recognized by semantic mapper.", "Add synonym or ontological alias to UNIVERSAL_ONTOLOGY.", True),
            "INCORRECT_OPERATION": ("Intent classification deviated from requested operation.", "Add pattern or exemplar to few-shot training pool.", True),
            "VALUE_NOT_FOUND": ("Entity value was not found in dataset index.", "Verify spelling, case, or fuzzy threshold.", False),
            "PHONETIC_MATCH_FAILED": ("Phonetic key (Double Metaphone / Soundex) did not align.", "Add domain-specific abbreviation or phonetic synonym to AliasResolver.", True),
            "FILTER_PARSING_ERROR": ("Failed to parse comparison condition or numeric bound.", "Refine numeric/range regex in normalization.py.", True),
            "AGGREGATION_ERROR": ("Mathematical aggregation mismatch or type error.", "Ensure target column is cast to numeric without currency symbols.", True),
            "RANKING_ERROR": ("Ranking offset or tie-break order misaligned.", "Review ordinal rank extraction and DESC/ASC direction.", True),
            "COMPARISON_ERROR": ("Side-by-side entity comparison failed to resolve target entities.", "Check column existence for both entities in dataset.", False),
            "AMBIGUITY_DETECTION_ERROR": ("Ambiguity guard either under-triggered or over-triggered.", "Adjust ambiguity confidence margin threshold.", True),
            "TIMEOUT": ("Query execution exceeded deadline.", "Optimize DuckDB query plan or add caching.", True),
            "SCHEMA_MISMATCH": ("Active sheet does not contain expected schema.", "Switch to correct sheet or verify dataset format.", False),
            "NORMALIZATION_ERROR": ("Text or date normalizer failed to standardize format.", "Update dateparser configuration or slang expansions.", True),
            "LLM_GROUNDING_VIOLATION": ("Answer validator detected numerical or entity discrepancy.", "Enforce deterministic template fallback.", True),
            "OUT_OF_SCOPE_MISCLASSIFICATION": ("General knowledge question misrouted to dataset or vice versa.", "Refine QuestionRouter context heuristic.", True),
            "OTHER": ("Unclassified execution anomaly.", "Inspect execution trace and audit logs.", False),
        }

        root_cause, suggested_fix, auto_fixable = fixes.get(category, ("Unknown anomaly.", "Investigate logs.", False))

        return {
            "question": question,
            "expected_intent": expected or "UNKNOWN",
            "actual_intent": actual or "UNKNOWN",
            "failure_category": category,
            "root_cause": root_cause,
            "suggested_fix": suggested_fix,
            "auto_fixable": auto_fixable,
        }

    def analyze(self, failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a list of failure records and return categorized diagnostics."""
        breakdown = {cat: 0 for cat in self.FAILURE_CATEGORIES}
        diagnostics = []

        for f in failures:
            classified = self.classify_failure(f)
            cat = classified["failure_category"]
            breakdown[cat] = breakdown.get(cat, 0) + 1
            diagnostics.append(classified)

        recommendations = []
        if breakdown["UNMAPPED_COLUMN"] > 0:
            recommendations.append("Expand Universal Ontology synonyms for unmapped column concepts.")
        if breakdown["FILTER_PARSING_ERROR"] > 0:
            recommendations.append("Review regex pattern bindings for relational and numerical bounds.")
        if breakdown["PHONETIC_MATCH_FAILED"] > 0:
            recommendations.append("Review Entity Alias & Abbreviation Resolution mappings.")
        if breakdown["LLM_GROUNDING_VIOLATION"] > 0:
            recommendations.append("Answer Firewall prevented hallucination; ensure fallback generation is calibrated.")
        if not recommendations:
            recommendations.append("All query operations healthy.")

        return {
            "total_failures": len(failures),
            "category_breakdown": breakdown,
            "diagnostics": diagnostics[:50],
            "recommendations": recommendations,
        }


failure_analyzer = FailureAnalyzer()
