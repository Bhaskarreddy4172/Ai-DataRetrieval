"""Robust intent classifier supporting 30+ analytical and retrieval intents."""

import re
from typing import Any, Dict, List, Optional, Tuple
from app.ai.ollama_client import ollama_client, OllamaClient
from app.utils.logger import logger
from app.utils.validators import safe_json_extract


INTENT_PATTERNS: List[Tuple[str, List[str]]] = [
    ("UNSUPPORTED_QUERY", [
        "weather", "temperature", "president", "prime minister", "population of",
        "who is the pm", "who won", "crime rate", "gdp of", "capital of usa", "capital of india",
        "stock price", "revenue for q4", "profit margin", "who is the ceo", "icc world cup"
    ]),
    ("DUPLICATES", ["duplicate", "dups", "dupes", "repeated records", "redundant rows"]),
    ("EXISTS", ["is there any", "do we have", "does any", "are there any", "exists"]),
    ("DISTINCT", ["distinct", "unique values", "list all different", "what types of", "what categories"]),
    ("MEDIAN", ["median", "middle value"]),
    ("COUNT", ["how many", "count ", "count the ", "total count", "count of", "total number of", "number of records", "number of rows", "count all", "headcount", "workforce size", "count staff", "how mny", "how many ppl"]),
    ("SUM", ["total ", "sum of", "overall ", "calculate total", "how much total", "total revenue", "total salary", "total payroll", "total remuneration"]),
    ("AVERAGE", ["average ", "avg ", "mean ", "average compensation"]),
    ("COMPARE", ["compare ", " vs ", " versus ", "who earns more", "which pays more", "higher than"]),
    ("RANK", ["second highest", "third highest", "second lowest", "third lowest", "rank"]),
    ("OUTLIER", ["outlier", "outliers", "unusually high", "unusually low", "anomaly", "anomalies", "abnormal"]),
    ("TOP_N", [
        "top ", "highest ", "best ", "maximum ", "max ", "costliest", "most expensive",
        "makes the most", "paid the most", "paid most", "earns most", "earning most",
        "earning highest", "highest pay", "max pay", "biggest money", "biggest pay",
        "biggest paycheck", "makes the biggest", "making the biggest", "making biggest",
        "most money", "most cash", "making bank", "top earner", "rockstar", "top talent",
        "highest compensation", "highest paid", "top paid", "makes maximum", "biggest salary",
        "earning most amount", "gets paid highest", "more money", "earning maximum",
        "sabse jyada", "highest wala", "on top", "who's on top", "who is on top"
    ]),
    ("BOTTOM_N", [
        "bottom", "lowest", "cheapest", "least paid", "paid least", "earns least",
        "earning lowest", "lowest pay", "min pay", "minimum", "min", "makes the least",
        "lowest compensation", "lowest earner", "sabse kam", "least money"
    ]),
    ("MISSING_DATA", ["missing", "null", "blank", "empty", "without", "no city", "no salary"]),
    ("DATASET_SUMMARY", ["dataset summary", "overview", "describe dataset", "profile"]),
    ("GROUP", ["group by", "per department", "per city", "by department", "by city", "by category", "distribution"]),
    ("LOOKUP", [
        "what is", "where is", "which is", "who is", "what's", "whats",
        "details for", "info for", "tell me", "show me", "capital of", "salary of",
        "batao", "dikhao", "cheppu", "chudu"
    ]),
    ("BOOLEAN_CHECK", [
        "is ", "are ", "does ", "did ", "do ", "was ", "were ", "can ", "could ",
        "has ", "have ", "will ", "would ", "should ", "check if", "verify if",
        "right?", " na?", " true or false", " correct?", "belongs to", "belong to",
        "comes under", "under ", "is it ", "isn't it"
    ]),
    ("SEARCH", ["search", "anything related to", "mentioning"])
]

BOOLEAN_INTENT_ALIASES = {
    "true_false": "BOOLEAN_CHECK",
    "yes_no": "BOOLEAN_CHECK",
    "fact_check": "BOOLEAN_CHECK",
    "relationship_check": "BOOLEAN_CHECK",
    "condition_check": "BOOLEAN_CHECK",
    "assertion_validation": "BOOLEAN_CHECK",
    "boolean": "BOOLEAN_CHECK",
    "boolean_check": "BOOLEAN_CHECK",
}


class IntentParser:
    """Classifies user question into structured intent with confidence scoring."""

    def __init__(self, ai_client: OllamaClient = ollama_client):
        self.client = ai_client

    def classify_with_rules(self, text: str, available_columns: List[str]) -> Tuple[str, float]:
        """Deterministic intent classification using token and syntax pattern matching."""
        t_lower = text.lower().strip()
        cols_lower = [c.lower() for c in available_columns]

        # 0. Check out of scope
        for kw in INTENT_PATTERNS[0][1]:
            if kw in t_lower and not any(kw in c for c in cols_lower):
                return "UNSUPPORTED_QUERY", 1.0

        # Guard: 'at least' or 'at most' is a comparison threshold, not MIN / MAX
        if "at least" in t_lower or "at most" in t_lower:
            has_multi = any(prep in t_lower for prep in [" in ", " from ", " for ", " and ", " with "])
            return ("MULTI_FILTER" if has_multi else "FILTER"), 0.95

        # Check for extreme intent with inverted syntax: e.g. "salary highest who?", "salary max who?", "who salary highest?"
        if re.search(r"\b(?:highest|maximum|max|top|best)\b", t_lower) and any(w in t_lower for w in ["who", "which", "person", "guy", "employee", "student"]):
            return "TOP_N", 0.96
        if re.search(r"\b(?:lowest|minimum|min|bottom|cheapest)\b", t_lower) and any(w in t_lower for w in ["who", "which", "person", "guy", "employee", "student"]):
            return "BOTTOM_N", 0.96

        # Check explicit patterns in precedence order
        for intent, patterns in INTENT_PATTERNS[1:]:
            # Guard: Wh-questions (what, which, who, where, when, why, how) are not BOOLEAN_CHECK
            # unless ending with a confirmation tag
            if intent == "BOOLEAN_CHECK":
                if any(t_lower.startswith(wh) for wh in ["what ", "which ", "who ", "where ", "when ", "why ", "how ", "what's ", "whats "]):
                    if not any(tag in t_lower for tag in ["true or false", "right?", "correct?"]):
                        continue

            for pat in patterns:
                if pat in t_lower:
                    # Confidence adjustment based on specificity
                    conf = 0.98 if len(pat) > 4 else 0.88
                    # Refine filter vs multi-filter
                    if intent in {"MULTI_FILTER", "FILTER"}:
                        has_num = bool(re.search(r"\d+", t_lower))
                        has_multi = " and " in t_lower or (has_num and any(prep in t_lower for prep in [" in ", " from ", " for "]))
                        return ("MULTI_FILTER" if has_multi else "FILTER"), 0.92
                    return intent, conf

        return "FILTER", 0.75

    def classify(self, text: str, available_columns: List[str]) -> Tuple[str, float]:
        """Classify intent using Ollama if online; otherwise deterministic rules."""
        health = self.client.check_health()
        if health.get("available"):
            prompt = (
                f"Classify the intent of this question for a dataset query engine:\n"
                f"Question: \"{text}\"\n"
                f"Columns: {available_columns}\n"
                f"Supported intents: LOOKUP, FILTER, MULTI_FILTER, COUNT, SUM, AVERAGE, MIN, MAX, MEDIAN, "
                f"TOP_N, BOTTOM_N, GROUP, COMPARE, DUPLICATES, DISTINCT, EXISTS, DATE_FILTER, MISSING_DATA, "
                f"BOOLEAN_CHECK, UNSUPPORTED_QUERY, CLARIFICATION.\n"
                f"Return JSON: {{\"intent\": \"<INTENT>\", \"confidence\": <float 0-1>}}"
            )
            raw = self.client.generate(prompt=prompt, json_format=True)
            if raw:
                parsed = safe_json_extract(raw)
                if parsed and "intent" in parsed and parsed["intent"] != "UNKNOWN":
                    raw_intent = str(parsed["intent"]).strip()
                    norm_intent = BOOLEAN_INTENT_ALIASES.get(raw_intent.lower(), raw_intent)
                    return norm_intent, float(parsed.get("confidence", 0.9))

        return self.classify_with_rules(text, available_columns)


intent_parser = IntentParser()

