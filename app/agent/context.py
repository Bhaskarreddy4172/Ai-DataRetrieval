"""ContextResolver resolving pronouns, references, and implicit dataset attributes."""

import re
from typing import Any, Dict, List, Optional, Tuple
from app.agent.memory import conversation_memory


class ContextResolver:
    """Resolves pronouns (she, he, they, there, the second one) using structured conversation memory."""

    PRONOUN_PATTERNS = [
        r"\bshe\b", r"\bhe\b", r"\bthey\b", r"\bher\b", r"\bhim\b", r"\bthem\b",
        r"\bthat employee\b", r"\bthat person\b", r"\bthe first one\b", r"\bthe second one\b",
        r"\bthere\b", r"\bsame department\b", r"\bsame city\b"
    ]

    def resolve(
        self,
        user_message: str,
        session_id: str = "default",
        available_columns: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Resolve pronouns and return contextualized question string and resolved metadata."""
        state = conversation_memory.get_state(session_id)
        if not state:
            return user_message, {}

        resolved_msg = user_message
        resolved_meta = {}

        last_entity = state.get("last_entity")
        last_location = state.get("last_location")
        last_question = state.get("last_question")

        # 1. Resolve personal pronouns (she / he / her / him / that person) -> last_entity
        if last_entity:
            for pat in [r"\bshe\b", r"\bhe\b", r"\bher\b", r"\bhim\b", r"\bthat employee\b", r"\bthat person\b"]:
                if re.search(pat, resolved_msg, flags=re.IGNORECASE):
                    resolved_msg = re.sub(pat, str(last_entity), resolved_msg, flags=re.IGNORECASE)
                    resolved_meta["resolved_entity"] = last_entity

        # 2. Resolve location references (there / in that city) -> last_location
        if last_location:
            for pat in [r"\bthere\b", r"\bin that city\b", r"\bin that state\b"]:
                if re.search(pat, resolved_msg, flags=re.IGNORECASE):
                    resolved_msg = re.sub(pat, f"in {last_location}", resolved_msg, flags=re.IGNORECASE)
                    resolved_meta["resolved_location"] = last_location

        # 3. Resolve follow-up ordinal queries ("what about the second one?")
        history = conversation_memory.get_history(session_id)
        if history and history[-1].results and len(history[-1].results) > 1:
            last_results = history[-1].results
            if re.search(r"\bthe second one\b", resolved_msg, flags=re.IGNORECASE) and len(last_results) >= 2:
                second_item = last_results[1]
                entity_val = next(iter(second_item.values()), None)
                if entity_val:
                    resolved_msg = re.sub(r"\bthe second one\b", str(entity_val), resolved_msg, flags=re.IGNORECASE)
                    resolved_meta["resolved_entity"] = entity_val

        return resolved_msg, resolved_meta


context_resolver = ContextResolver()

