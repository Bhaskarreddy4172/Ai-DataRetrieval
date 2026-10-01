"""Manages multi-turn conversation sessions, context resolution, and follow-up contextualization."""

import threading
from typing import Any, Dict, List, Optional, Tuple
from app.conversation.resolver import conversation_resolver
from app.query.schema import FilterCondition, StructuredQuery


class ConversationManager:
    """Tracks session dialogue state and merges contextual follow-up questions with pronoun resolution."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}

    def add_turn(
        self,
        session_id: str,
        question: str,
        query: StructuredQuery,
        result_count: int,
        results: Optional[List[Dict[str, Any]]] = None,
        answer: Optional[str] = None,
    ):
        import time
        from app.dataset.registry import parent_child_registry

        # Extract parent-child context entities from results or question
        curr_state = None
        curr_capital = None
        curr_child_ds = None
        curr_village = None

        if results and len(results) > 0:
            first_row = results[0]
            if isinstance(first_row, dict):
                curr_state = first_row.get("State") or first_row.get("state")
                curr_capital = first_row.get("Capital") or first_row.get("capital")
                curr_child_ds = first_row.get("Child_Dataset") or first_row.get("child_dataset")
                curr_village = first_row.get("Village") or first_row.get("village") or first_row.get("Village_ID")

        if not curr_child_ds and (curr_state or curr_capital or question):
            resolved = parent_child_registry.resolve_child_dataset(curr_state or curr_capital or question)
            if resolved:
                name, path = resolved
                curr_child_ds = path.name
                if not curr_state:
                    curr_state = name

        curr_comparison = None
        if results and len(results) > 0 and isinstance(results[0], dict) and "left_entity" in results[0]:
            c_rec = results[0]
            curr_comparison = {
                "left_entity": c_rec.get("left_entity"),
                "right_entity": c_rec.get("right_entity"),
                "attribute": c_rec.get("attribute"),
                "scope": c_rec.get("scope"),
                "higher_entity": c_rec.get("higher_entity"),
                "lower_entity": c_rec.get("lower_entity"),
                "left_value": c_rec.get("left_value"),
                "right_value": c_rec.get("right_value"),
                "left_parent": c_rec.get("left_parent"),
                "right_parent": c_rec.get("right_parent"),
            }

        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            
            # Inherit previous session state only if not a global query
            from app.query.fast_classifier import fast_query_classifier
            is_global = fast_query_classifier.is_global_query(question)

            if is_global:
                if results and len(results) > 0 and isinstance(results[0], dict) and (results[0].get("state") or results[0].get("State")):
                    final_state = results[0].get("state") or results[0].get("State")
                    final_capital = results[0].get("capital") or results[0].get("Capital")
                    final_child_ds = None
                else:
                    final_state = None
                    final_capital = None
                    final_child_ds = None
                final_village = None
                final_comparison = curr_comparison
            else:
                prev_turn = self._sessions[session_id][-1] if self._sessions[session_id] else {}
                final_state = curr_state or prev_turn.get("state")
                final_capital = curr_capital or prev_turn.get("capital")
                final_child_ds = curr_child_ds or prev_turn.get("child_dataset")
                final_village = curr_village or prev_turn.get("village")
                final_comparison = curr_comparison or prev_turn.get("comparison")

            self._sessions[session_id].append({
                "question": question,
                "query": query,
                "result_count": result_count,
                "results": (results or [])[:5],
                "answer": answer or "",
                "state": final_state,
                "capital": final_capital,
                "child_dataset": final_child_ds,
                "village": final_village,
                "comparison": final_comparison,
                "timestamp": time.time(),
            })
            # Retain up to 50 turns per session
            if len(self._sessions[session_id]) > 50:
                self._sessions[session_id] = self._sessions[session_id][-50:]

    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a specific session."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def clear_all(self) -> None:
        """Clear all conversation sessions across the application."""
        with self._lock:
            self._sessions.clear()

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Return high-level summary of the session history."""
        with self._lock:
            history = list(self._sessions.get(session_id, []))
        return {
            "session_id": session_id,
            "turns_count": len(history),
            "last_question": history[-1]["question"] if history else None,
        }

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def get_last_query(self, session_id: str) -> Optional[StructuredQuery]:
        with self._lock:
            history = self._sessions.get(session_id)
            if history:
                return history[-1]["query"]
            return None

    def get_last_comparison(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve most recent comparison context from session turns."""
        with self._lock:
            history = self._sessions.get(session_id)
            if history:
                for turn in reversed(history):
                    if turn.get("comparison"):
                        return turn["comparison"]
            return None

    def get_active_state(self, session_id: str) -> Optional[str]:
        with self._lock:
            history = self._sessions.get(session_id)
            if history:
                return history[-1].get("state")
            return None

    def get_active_capital(self, session_id: str) -> Optional[str]:
        with self._lock:
            history = self._sessions.get(session_id)
            if history:
                return history[-1].get("capital")
            return None

    def get_active_child_dataset(self, session_id: str) -> Optional[str]:
        with self._lock:
            history = self._sessions.get(session_id)
            if history:
                return history[-1].get("child_dataset")
            return None

    def get_last_village(self, session_id: str) -> Optional[str]:
        with self._lock:
            history = self._sessions.get(session_id)
            if history:
                return history[-1].get("village")
            return None

    def contextualize_followup(self, question: str, session_id: str, current_query: StructuredQuery) -> StructuredQuery:
        """Backward-compatible wrapper for follow-up merge."""
        _, resolved = self.contextualize_and_resolve(question, session_id, [], current_query)
        return resolved or current_query

    def contextualize_and_resolve(
        self,
        question: str,
        session_id: str,
        available_columns: List[str],
        current_query: Optional[StructuredQuery] = None
    ) -> Tuple[str, Optional[StructuredQuery]]:
        """Resolve pronouns ('it', 'they') and merge follow-up intent ('what about X?')."""
        with self._lock:
            history = self._sessions.get(session_id, [])

        if not history:
            return question, current_query

        # Global queries must completely ignore last_entity and previous session state
        from app.query.fast_classifier import fast_query_classifier
        if fast_query_classifier.is_global_query(question, session_id=session_id):
            return question, current_query

        # 1. Use ConversationResolver
        if available_columns:
            clarified_text, resolved_query = conversation_resolver.resolve(question, history, available_columns)
            if resolved_query:
                return clarified_text, resolved_query
        else:
            clarified_text = question

        # 2. Merge follow-up filters
        last_query = history[-1]["query"]
        q_lower = question.lower()
        is_followup = any(marker in q_lower for marker in [
            "what about", "how about", "and for", "and only", "only in", "what of", "and in"
        ])
        if is_followup and last_query and current_query:
            merged_conditions = list(last_query.conditions)
            for c in current_query.conditions:
                existing = [ec for ec in merged_conditions if ec.column.lower() == c.column.lower()]
                if existing:
                    existing[0].value = c.value
                    existing[0].operator = c.operator
                else:
                    merged_conditions.append(c)

            current_query.conditions = merged_conditions
            if current_query.operation == "FILTER" and last_query.operation in {"SUM", "AVERAGE", "COUNT", "MEDIAN"}:
                current_query.operation = last_query.operation
                current_query.target_column = last_query.target_column

        return clarified_text, current_query


conversation_manager = ConversationManager()
