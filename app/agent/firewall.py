"""Hallucination Firewall verifying factual answer alignment against tool execution results."""

import re
from typing import Any, Dict, List, Tuple
from app.utils.logger import logger


class HallucinationFirewall:
    """Strictly verifies that numeric and factual claims in assistant response match tool outputs."""

    @staticmethod
    def verify(
        answer: str,
        tool_results: List[Dict[str, Any]],
        aggregation: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[str]]:
        """Check numbers and facts in answer against tool execution output."""
        if not answer:
            return True, []

        discrepancies = []

        # Extract numeric tokens from answer (e.g., 1490000, 80, 2.5)
        numbers_in_answer = re.findall(r"\b\d+(?:\.\d+)?\b", answer.replace(",", ""))

        # Collect all valid numbers from tool results & aggregation
        valid_numbers = set()
        
        if aggregation and "value" in aggregation and aggregation["value"] is not None:
            try:
                val_str = str(aggregation["value"]).replace(",", "")
                valid_numbers.update(re.findall(r"\b\d+(?:\.\d+)?\b", val_str))
            except Exception:
                pass

        for row in tool_results:
            for v in row.values():
                if v is not None:
                    try:
                        val_str = str(v).replace(",", "")
                        valid_numbers.update(re.findall(r"\b\d+(?:\.\d+)?\b", val_str))
                    except Exception:
                        pass

        # Also add matched row count
        valid_numbers.add(str(len(tool_results)))

        # Verify each numeric claim
        for num_str in numbers_in_answer:
            # Ignore common small formatting digits or year ranges unless suspicious
            try:
                num_val = float(num_str)
                if num_val in [0, 1, 2, 3, 5, 10, 50, 100]:
                    continue
            except ValueError:
                pass

            if num_str not in valid_numbers:
                # Check if formatted float representation exists
                found_match = False
                for v_str in valid_numbers:
                    try:
                        if abs(float(num_str) - float(v_str)) < 0.01:
                            found_match = True
                            break
                    except ValueError:
                        pass

                if not found_match:
                    discrepancies.append(f"Number '{num_str}' in answer not found in tool results.")

        if discrepancies:
            logger.warning(f"Hallucination Firewall detected discrepancies: {discrepancies}")
            return False, discrepancies

        return True, []


hallucination_firewall = HallucinationFirewall()

