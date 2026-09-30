"""ResultVerifier checking factual consistency before answer delivery."""

from typing import Any, Dict, List, Optional, Tuple
from app.agent.firewall import hallucination_firewall


class ResultVerifier:
    """Verifies that natural language answers align 100% with tool results."""

    def verify_answer(
        self,
        answer: str,
        results: List[Dict[str, Any]],
        aggregation: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[str]]:
        return hallucination_firewall.verify(answer, results, aggregation)


result_verifier = ResultVerifier()

