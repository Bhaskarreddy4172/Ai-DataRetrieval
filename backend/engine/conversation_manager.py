"""Conversation Memory Engine (Deliverable 13).

Tracks multi-turn context, resolves pronouns (he/she/it/they), ordinal references
('second one', 'top person'), and merges follow-up questions seamlessly.
"""

from typing import Any, Dict, List, Optional, Tuple
from app.conversation.context import conversation_manager as core_conv_manager
from app.conversation.resolver import conversation_resolver
from app.query.schema import StructuredQuery


class ConversationMemoryEngine:
    """Universal Conversation Memory and Context Resolution Engine."""

    def add_turn(
        self,
        session_id: str,
        question: str,
        query: StructuredQuery,
        result_count: int,
        results: Optional[List[Dict[str, Any]]] = None,
        answer: Optional[str] = None
    ) -> None:
        """Record dialogue turn into session memory."""
        core_conv_manager.add_turn(
            session_id=session_id,
            question=question,
            query=query,
            result_count=result_count,
            results=results,
            answer=answer
        )

    def contextualize(
        self,
        question: str,
        session_id: str = "default",
        available_columns: Optional[List[str]] = None
    ) -> Tuple[str, Optional[StructuredQuery]]:
        """Resolve pronouns and merge follow-up intent from dialogue memory."""
        cols = available_columns or []
        return core_conv_manager.contextualize_and_resolve(question, session_id, cols)

    def get_history(self, session_id: str = "default") -> List[Dict[str, Any]]:
        """Retrieve conversation history turns for a session."""
        return core_conv_manager.get_history(session_id)

    def clear(self, session_id: Optional[str] = None) -> None:
        """Reset conversation memory for a session or globally."""
        if session_id:
            core_conv_manager.clear_session(session_id)
        else:
            core_conv_manager.clear_all()


conversation_memory_engine = ConversationMemoryEngine()

