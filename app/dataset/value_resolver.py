"""Dataset-aware value resolver: abbreviation mapping, typo correction, and ambiguity detection."""

import difflib
import re
from typing import Any, Dict, List, Optional, Tuple
from app.dataset.indexer import dataset_indexer
from app.dataset.alias_resolver import entity_alias_resolver
from app.utils.fuzzy_match import LOCATION_ABBREVIATIONS, fuzzy_match_value
from app.utils.logger import logger


class DatasetValueResolver:
    """Resolves natural language entities and typos against actual dataset values."""

    def resolve_value_in_column(
        self,
        raw_term: str,
        column_name: str,
        available_values: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Resolve a candidate term to an exact value in a specific column with ambiguity detection."""
        if not raw_term:
            return {"resolved_value": None, "confidence": 0.0, "is_ambiguous": False, "candidates": []}

        term_clean = raw_term.strip().lower()

        # Numeric values should never be treated as text prefixes or substrings (e.g. '5' is not '85')
        if term_clean.replace(".", "", 1).isdigit():
            return {"resolved_value": None, "confidence": 0.0, "is_ambiguous": False, "candidates": []}

        vals = available_values or dataset_indexer.get_column_values(column_name)
        if not vals:
            return {"resolved_value": raw_term, "confidence": 0.5, "is_ambiguous": False, "candidates": []}

        val_map = {str(v).strip().lower(): v for v in vals if v is not None}

        # 0. Check Entity Alias & Shortcut Resolution (AP, TG, WB, HYD, BLR, HR, IT, etc.)
        alias_res = entity_alias_resolver.resolve_alias(term_clean, available_columns=[column_name])
        if alias_res and alias_res.get("canonical_name"):
            canon_lower = alias_res["canonical_name"].strip().lower()
            if canon_lower in val_map:
                return {
                    "resolved_value": val_map[canon_lower],
                    "confidence": alias_res.get("confidence", 0.98),
                    "is_ambiguous": False,
                    "candidates": [val_map[canon_lower]]
                }

        # 1. Expand known location abbreviations
        expanded = LOCATION_ABBREVIATIONS.get(term_clean, term_clean).lower()

        # 2. Check for Ambiguity among distinct entity names (e.g., "Pune" vs "Pune City" vs "Pune Rural")
        # Only check word prefix if word is at least 3 chars
        if len(expanded) >= 3:
            matches_prefix = [v for k, v in val_map.items() if (k == expanded or k.startswith(f"{expanded} ") or f" {expanded}" in k)]
            if len(matches_prefix) > 1:
                exact_match = val_map.get(expanded)
                other_variants = [v for v in matches_prefix if str(v).strip().lower() != expanded]
                if exact_match and len(other_variants) >= 1:
                    return {
                        "resolved_value": exact_match,
                        "confidence": 0.85,
                        "is_ambiguous": True,
                        "candidates": [str(v) for v in matches_prefix],
                        "clarification_prompt": f"Do you mean '{exact_match}' specifically, or should I include {', '.join(str(v) for v in other_variants)}?"
                    }
                elif not exact_match and len(matches_prefix) > 1:
                    return {
                        "resolved_value": None,
                        "confidence": 0.60,
                        "is_ambiguous": True,
                        "candidates": [str(v) for v in matches_prefix],
                        "clarification_prompt": f"Did you mean {', '.join(str(v) for v in matches_prefix)}?"
                    }

        # 3. Exact Case-Insensitive Match
        if expanded in val_map:
            return {
                "resolved_value": val_map[expanded],
                "confidence": 1.0,
                "is_ambiguous": False,
                "candidates": [val_map[expanded]]
            }

        # 4. Whole-word / substring containment for text >= 4 characters
        if len(expanded) >= 4:
            for k, original_v in val_map.items():
                if len(k) >= 4 and (expanded in k or k in expanded):
                    return {
                        "resolved_value": original_v,
                        "confidence": 0.92,
                        "is_ambiguous": False,
                        "candidates": [original_v]
                    }
                elif len(k) < 4 and re.search(r"\b" + re.escape(k) + r"\b", expanded):
                    return {
                        "resolved_value": original_v,
                        "confidence": 0.92,
                        "is_ambiguous": False,
                        "candidates": [original_v]
                    }

        # 5. Fuzzy / Levenshtein Typo Match (e.g. "hydrabad" -> "Hyderabad", "karnatka" -> "Karnataka")
        close_matches = difflib.get_close_matches(expanded, list(val_map.keys()), n=3, cutoff=0.72)
        if close_matches:
            best_match_key = close_matches[0]
            ratio = difflib.SequenceMatcher(None, expanded, best_match_key).ratio()
            best_val = val_map[best_match_key]

            # Check ambiguity among close matches
            if len(close_matches) > 1:
                r2 = difflib.SequenceMatcher(None, expanded, close_matches[1]).ratio()
                if abs(ratio - r2) < 0.05:
                    return {
                        "resolved_value": None,
                        "confidence": ratio,
                        "is_ambiguous": True,
                        "candidates": [val_map[k] for k in close_matches[:2]],
                        "clarification_prompt": f"Did you mean '{best_val}' or '{val_map[close_matches[1]]}'?"
                    }

            return {
                "resolved_value": best_val,
                "confidence": round(ratio, 2),
                "is_ambiguous": False,
                "candidates": [best_val]
            }

        # 6. Phonetic Sound-Alike Match (e.g. "Cikkim" -> "Sikkim", "Kalkata" -> "Kolkata", "Bengluru" -> "Bengaluru")
        from app.utils.phonetic import phonetic_match_score
        phonetic_candidates: List[Tuple[float, str]] = []
        for k, original_v in val_map.items():
            p_score = phonetic_match_score(expanded, k)
            if p_score >= 0.78:
                phonetic_candidates.append((p_score, original_v))

        if phonetic_candidates:
            phonetic_candidates.sort(key=lambda x: x[0], reverse=True)
            top_score, top_val = phonetic_candidates[0]

            # Check ambiguity among top phonetic candidates
            if len(phonetic_candidates) > 1 and abs(top_score - phonetic_candidates[1][0]) < 0.03:
                return {
                    "resolved_value": None,
                    "confidence": top_score,
                    "is_ambiguous": True,
                    "candidates": [top_val, phonetic_candidates[1][1]],
                    "clarification_prompt": f"Did you mean '{top_val}' or '{phonetic_candidates[1][1]}'?"
                }

            return {
                "resolved_value": top_val,
                "confidence": round(top_score, 2),
                "is_ambiguous": False,
                "candidates": [top_val]
            }

        return {
            "resolved_value": None,
            "confidence": 0.0,
            "is_ambiguous": False,
            "candidates": []
        }


dataset_value_resolver = DatasetValueResolver()

