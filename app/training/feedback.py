"""Active learning feedback manager and human correction queue (Sections 47, 48, 60)."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.utils.logger import logger

FEEDBACK_FILE = Path("logs/active_learning_feedback.json")


class ActiveLearningFeedbackManager:
    """Stores validated user corrections and difficult queries for continuous learning."""

    def __init__(self):
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not FEEDBACK_FILE.exists():
            FEEDBACK_FILE.write_text("[]", encoding="utf-8")

    def _load_all(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_all(self, items: List[Dict[str, Any]]) -> None:
        FEEDBACK_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")

    def record_feedback(
        self,
        question: str,
        is_correct: bool,
        system_operation: Optional[str] = None,
        system_answer: Optional[str] = None,
        user_correction: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        items = self._load_all()
        entry = {
            "id": f"FB_{len(items) + 1:04d}",
            "timestamp": time.time(),
            "question": question,
            "is_correct": is_correct,
            "system_operation": system_operation,
            "system_answer": system_answer,
            "user_correction": user_correction,
            "status": "reviewed" if user_correction else "pending_review"
        }
        items.append(entry)
        self._save_all(items)
        logger.info(f"Recorded active learning feedback: {entry['id']}")
        return entry

    def get_review_queue(self) -> List[Dict[str, Any]]:
        items = self._load_all()
        return [it for it in items if not it.get("is_correct", True)]

    def get_stats(self) -> Dict[str, Any]:
        items = self._load_all()
        total = len(items)
        positive = sum(1 for it in items if it.get("is_correct", False))
        negative = total - positive
        return {
            "total_feedback_count": total,
            "positive_count": positive,
            "negative_count": negative,
            "accuracy_ratio": round(positive / total, 2) if total > 0 else 1.0,
            "pending_review_count": len(self.get_review_queue())
        }


feedback_manager = ActiveLearningFeedbackManager()
