"""Structured Conversation Memory tracking session entities, intents, and past tool results."""

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MemoryTurn(BaseModel):
    session_id: str
    question: str
    tool_name: str
    operation: str
    results: List[Dict[str, Any]]
    aggregation: Optional[Dict[str, Any]] = None
    answer: str
    timestamp: float = Field(default_factory=time.time)


class StructuredConversationMemory:
    """Manages structured session memory for context resolution and pronoun disambiguation."""

    def __init__(self):
        self._sessions: Dict[str, List[MemoryTurn]] = {}
        self._state: Dict[str, Dict[str, Any]] = {}

    def add_turn(
        self,
        session_id: str,
        question: str,
        tool_name: str,
        operation: str,
        results: List[Dict[str, Any]],
        aggregation: Optional[Dict[str, Any]],
        answer: str,
    ) -> None:
        turn = MemoryTurn(
            session_id=session_id,
            question=question,
            tool_name=tool_name,
            operation=operation,
            results=results,
            aggregation=aggregation,
            answer=answer,
        )

        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(turn)

        # Update structured state
        state = self._state.get(session_id, {})
        state["last_question"] = question
        state["last_tool"] = tool_name
        state["last_operation"] = operation
        state["last_answer"] = answer

        if results and len(results) > 0:
            first = results[0]
            # Track entities and locations if present
            for k, v in first.items():
                if any(w in k.lower() for w in ["name", "employee", "person"]):
                    state["last_entity"] = v
                elif any(w in k.lower() for w in ["city", "state", "location"]):
                    state["last_location"] = v

        self._state[session_id] = state

    def get_state(self, session_id: str) -> Dict[str, Any]:
        return self._state.get(session_id, {})

    def get_history(self, session_id: str) -> List[MemoryTurn]:
        return self._sessions.get(session_id, [])

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._state.pop(session_id, None)

    def clear_all(self) -> None:
        self._sessions.clear()
        self._state.clear()


conversation_memory = StructuredConversationMemory()

