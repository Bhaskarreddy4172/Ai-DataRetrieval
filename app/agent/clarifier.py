"""AmbiguityClarifier detecting ambiguous entity matches and constructing clarification options."""

import re
from typing import Any, Dict, List, Optional
import pandas as pd


class AmbiguityClarifier:
    """Detects ambiguity when multiple entities match a query and constructs clarification choices."""

    def check_ambiguity(
        self,
        question: str,
        df: pd.DataFrame,
        entity_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if df.empty:
            return None

        search_terms = []
        if entity_name:
            search_terms.append(entity_name)

        # Also extract proper nouns or words with length > 2 from question
        words = re.findall(r"\b[A-Z][a-z]+\b|\b\w{3,}\b", question)
        for w in words:
            if w.lower() not in ["show", "what", "where", "salary", "who", "have", "with", "from", "many", "which"]:
                search_terms.append(w)

        text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

        for term in search_terms:
            t_lower = term.lower()
            matches = []
            for col in text_cols:
                col_str = df[col].astype(str).str.lower()
                matched_df = df[col_str.str.contains(t_lower, regex=False, na=False)]
                if len(matched_df) > 1:
                    matches = matched_df.to_dict(orient="records")
                    break

            if len(matches) > 1:
                options = []
                for idx, r in enumerate(matches[:5], 1):
                    label = ", ".join([f"{k}: {v}" for k, v in r.items() if not k.startswith("_")][:3])
                    options.append(f"{idx}. {label}")

                return {
                    "is_ambiguous": True,
                    "message": f"I found multiple matching records for '{term}'. Which one do you mean?",
                    "options": options,
                    "candidate_records": matches[:5],
                }

        return None


ambiguity_clarifier = AmbiguityClarifier()

