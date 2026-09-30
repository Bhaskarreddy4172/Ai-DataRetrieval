"""Natural language query parser combining Ollama AI understanding and deterministic fallback."""

import re
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator
from app.ai.ollama_client import ollama_client, OllamaClient
from app.ai.prompts import QUERY_PARSER_SYSTEM_PROMPT
from app.utils.logger import logger
from app.utils.validators import safe_json_extract, sanitize_text, validate_question


class StructuredQuery(BaseModel):
    intent: str = Field(..., description="Query retrieval intent")
    state: Optional[str] = Field(None, description="Extracted state name")
    capital: Optional[str] = Field(None, description="Extracted capital city name")
    prefix: Optional[str] = Field(None, description="Prefix character or string")
    limit: int = Field(10, ge=1, le=100, description="Max results limit")

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        allowed = {
            "GET_CAPITAL",
            "GET_STATE",
            "LIST_ALL",
            "COUNT",
            "SEARCH_STATE",
            "SEARCH_CAPITAL",
            "FILTER_BY_PREFIX",
            "UNKNOWN",
        }
        upper_v = v.strip().upper()
        if upper_v not in allowed:
            return "UNKNOWN"
        return upper_v


class QueryParser:
    """Parses user question into StructuredQuery using Ollama with rule-based fallback."""

    # Keywords that indicate non-dataset questions (hallucination prevention)
    OUT_OF_SCOPE_WORDS = [
        "prime minister", "president", "chief minister", "cm", "pm",
        "population", "area", "weather", "temperature", "car", "honda",
        "service", "warranty", "flight", "train", "hotel", "crime", "gdp"
    ]

    def __init__(self, ai_client: OllamaClient = ollama_client):
        self.client = ai_client

    def parse_with_rules(self, question: str) -> StructuredQuery:
        """Deterministic rule-based parser for fallback and verification."""
        clean_q = sanitize_text(question).rstrip("?").strip()
        lower_q = clean_q.lower()

        # Check out-of-scope first
        for word in self.OUT_OF_SCOPE_WORDS:
            if word in lower_q:
                return StructuredQuery(intent="UNKNOWN")

        # Count queries
        if any(p in lower_q for p in ["how many states", "count of states", "total states", "number of states"]):
            return StructuredQuery(intent="COUNT")

        # List all queries
        if any(p in lower_q for p in ["show all", "list all", "all states", "all capitals", "all the states", "every state"]):
            return StructuredQuery(intent="LIST_ALL", limit=100)

        # Prefix search: states starting with X
        prefix_match = re.search(r"(?:states|capitals)?\s*(?:starting with|begins with|starts with)\s+([a-zA-Z])\b", lower_q)
        if prefix_match:
            letter = prefix_match.group(1).upper()
            return StructuredQuery(intent="FILTER_BY_PREFIX", prefix=letter, limit=50)

        # Reverse lookup: which state has X as capital
        m_rev1 = re.search(r"which state has\s+([a-zA-Z\s]+?)\s+as(?:\s+its)?\s+capital", lower_q)
        if m_rev1:
            cap = sanitize_text(m_rev1.group(1))
            return StructuredQuery(intent="GET_STATE", capital=cap)

        m_rev2 = re.search(r"([a-zA-Z\s]+?)\s+is the capital of which state", lower_q)
        if m_rev2:
            cap = sanitize_text(m_rev2.group(1))
            return StructuredQuery(intent="GET_STATE", capital=cap)

        m_rev3 = re.search(r"state (?:for|of)\s+([a-zA-Z\s]+)", lower_q)
        if m_rev3:
            cap = sanitize_text(m_rev3.group(1))
            return StructuredQuery(intent="GET_STATE", capital=cap)

        # Direct lookup: capital of X
        m_dir1 = re.search(r"capital of\s+([a-zA-Z\s]+)", lower_q)
        if m_dir1:
            st = sanitize_text(m_dir1.group(1))
            if st not in ["the", "india", "country"]:
                return StructuredQuery(intent="GET_CAPITAL", state=st)

        m_dir2 = re.search(r"what is\s+([a-zA-Z\s]+?)(?:'s)?\s+capital", lower_q)
        if m_dir2:
            st = sanitize_text(m_dir2.group(1))
            if st not in ["the", "india", "country"]:
                return StructuredQuery(intent="GET_CAPITAL", state=st)

        # Fallback keyword match
        cleaned_words = re.findall(r"\b[a-zA-Z]{3,}\b", lower_q)
        stop_words = {"what", "which", "tell", "show", "find", "capital", "state", "city", "india", "please"}
        keywords = [w for w in cleaned_words if w not in stop_words]
        if keywords:
            return StructuredQuery(intent="SEARCH_STATE", state=keywords[0])

        return StructuredQuery(intent="UNKNOWN")

    def parse(self, question: str) -> StructuredQuery:
        """Parse natural language question using Ollama when available, falling back to rule parser."""
        is_valid, validated_or_err = validate_question(question)
        if not is_valid:
            logger.warning(f"Invalid question received: '{question}' - {validated_or_err}")
            return StructuredQuery(intent="UNKNOWN")

        clean_q = validated_or_err

        # Quick check for clear out-of-scope queries to avoid wasting LLM inference
        lower_q = clean_q.lower()
        for word in self.OUT_OF_SCOPE_WORDS:
            if word in lower_q:
                logger.info(f"Query flagged as out-of-scope by rule filter: '{clean_q}'")
                return StructuredQuery(intent="UNKNOWN")

        # Attempt parsing via Ollama if available
        health = self.client.check_health()
        if health.get("available"):
            logger.info(f"Dispatching query parsing to Ollama: '{clean_q}'")
            raw_response = self.client.generate(
                prompt=f"Parse this query: \"{clean_q}\"",
                system=QUERY_PARSER_SYSTEM_PROMPT,
                json_format=True,
            )
            if raw_response:
                parsed_json = safe_json_extract(raw_response)
                if parsed_json and "intent" in parsed_json:
                    try:
                        structured = StructuredQuery(**parsed_json)
                        logger.info(f"Ollama parsed query successfully into: {structured.model_dump()}")
                        return structured
                    except Exception as err:
                        logger.warning(f"Failed to validate Ollama JSON response: {err}")

        # Fallback to rule parser
        logger.info(f"Using rule-based parser fallback for query: '{clean_q}'")
        return self.parse_with_rules(clean_q)


query_parser = QueryParser()
