"""Input sanitization, question validation, and safe JSON extraction."""

import json
import re
from typing import Any, Dict, Optional, Tuple


def sanitize_text(text: Optional[str]) -> str:
    """Strip leading/trailing whitespace and collapse internal spaces."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).strip())


def validate_question(question: Optional[str], max_length: int = 300) -> Tuple[bool, str]:
    """Validate user natural language question."""
    if not question:
        return False, "Question cannot be empty."
    cleaned = sanitize_text(question)
    if len(cleaned) < 2:
        return False, "Question is too short."
    if len(cleaned) > max_length:
        return False, f"Question exceeds maximum length of {max_length} characters."
    return True, cleaned


def safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    """Safely extract a JSON object from text, handling markdown code fences."""
    if not text:
        return None
    cleaned = text.strip()

    # Match ```json { ... } ``` or ``` { ... } ```
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if code_block:
        candidate = code_block.group(1)
    else:
        # Match first balanced/outer curly braces
        curly_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        candidate = curly_match.group(0) if curly_match else cleaned

    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None
