"""Universal Semantic Relationship Parser: extracts conceptual entity-attribute relationships.

Handles:
- Possessives: X's Y, X’s Y, Xs Y, X Y (noun adjunct)
- Prepositional: Y of X, Y for X, Y belonging to X, Y associated with X
- Wh-Inversions: Which city is X's Y?, X has which Y?
- Reverse Lookups: Which state has Y as its X?, Y is the X of which Z?, Y belongs to which Z?
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.dataset.indexer import dataset_indexer
from app.dataset.semantic_mapper import semantic_column_mapper
from app.dataset.value_resolver import dataset_value_resolver
from app.dataset.alias_resolver import entity_alias_resolver
from app.utils.fuzzy_match import extract_best_match_from_text
from app.utils.phonetic import phonetic_match_score


# Common column synonyms for dataset attributes across domains
ATTRIBUTE_SYNONYMS: Dict[str, List[str]] = {
    "capital": ["capital", "capital city", "headquarter", "hq", "main city"],
    "state": ["state", "province", "region", "territory"],
    "salary": ["salary", "pay", "income", "earnings", "wage", "ctc", "compensation", "package", "remuneration"],
    "department": ["department", "dept", "division", "team", "unit"],
    "designation": ["designation", "role", "title", "job title", "position", "post"],
    "city": ["city", "location", "town", "place", "metro"],
    "price": ["price", "cost", "mrp", "rate", "fee", "amount", "charge"],
    "category": ["category", "type", "class", "segment", "genre"],
    "performance": ["performance", "performance score", "rating", "score", "perf"],
    "joining date": ["joining date", "date of joining", "doj", "joined date", "start date"],
    "language": ["official language", "language", "mother tongue", "spoken language"],
}


@dataclass
class SemanticRelationship:
    """Conceptual representation of an entity-attribute inquiry."""
    subject_val: str
    subject_col: str
    target_col: str
    relationship_type: str  # "HAS", "OF", "BELONGS_TO"
    direction: str          # "DIRECT" (State -> Capital) or "REVERSE" (Capital -> State)
    confidence: float
    raw_entity_text: str
    raw_attribute_text: str
    explanation: str


class SemanticRelationshipParser:
    """Parses natural language queries into deterministic entity-attribute relationships."""

    QUESTION_CLEAN_PREFIXES = [
        r"^what\s+is\s+(?:the\s+)?",
        r"^what\s+are\s+(?:the\s+)?",
        r"^what's\s+(?:the\s+)?",
        r"^whats\s+(?:the\s+)?",
        r"^which\s+(?:city|state|department|employee|product|person)\s+is\s+(?:the\s+)?",
        r"^tell\s+me\s+(?:the\s+)?(?:about\s+)?",
        r"^give\s+me\s+(?:the\s+)?",
        r"^show\s+me\s+(?:the\s+)?",
        r"^can\s+you\s+tell\s+me\s+(?:the\s+)?",
        r"^do\s+you\s+know\s+(?:the\s+)?",
        r"^find\s+(?:the\s+)?",
        r"^search\s+(?:for\s+)?(?:the\s+)?",
        r"^what\s+about\s+(?:the\s+)?",
        r"^how\s+about\s+(?:the\s+)?",
        r"^batao\s+(?:ki\s+)?(?:the\s+)?",
        r"^dikhao\s+(?:the\s+)?",
        r"^cheppu\s+(?:the\s+)?",
        r"^chudu\s+(?:the\s+)?",
    ]

    QUESTION_CLEAN_SUFFIXES = [
        r"\s+enti$",
        r"\s+kya\s+hai$",
        r"\s+kya$",
        r"\s+what$",
        r"\s+hai$",
        r"\s+hain$",
        r"\s+batao$",
        r"\s+dikhao$",
        r"\s+cheppu$",
        r"\s+chudu$",
        r"\s+soller$",
        r"\s+heli$",
        r"\s+yenu$",
        r"\s+enu$",
        r"\s+enna$",
    ]

    def parse(
        self,
        question: str,
        available_columns: List[str],
        schema_dict: Optional[Dict[str, Any]] = None,
        df: Optional[Any] = None
    ) -> Optional[SemanticRelationship]:
        """Attempt to extract entity-attribute relationship from natural language question."""
        if not question or not available_columns:
            return None

        clean_q = question.strip()
        # Normalize curly apostrophes and multiple spaces
        clean_q = clean_q.replace("’", "'").replace("`", "'")
        clean_q = re.sub(r"\s+", " ", clean_q)
        q_lower = clean_q.lower().rstrip("?!.,;")

        # 1. Try Reverse Lookup Patterns First (more specific)
        rev_rel = self._parse_reverse_lookup(q_lower, clean_q, available_columns, schema_dict)
        if rev_rel:
            return rev_rel

        # 2. Try Prepositional Patterns: "Y of X", "Y for X", "Y belonging to X"
        prep_rel = self._parse_prepositional(q_lower, clean_q, available_columns, schema_dict)
        if prep_rel:
            return prep_rel

        # 3. Try Inverted Verb Patterns: "Which [type] is X's Y?", "X has which Y?"
        inv_rel = self._parse_inverted(q_lower, clean_q, available_columns, schema_dict)
        if inv_rel:
            return inv_rel

        # 4. Try Possessive Patterns: "X's Y", "X's Y?", "Xs Y"
        poss_rel = self._parse_possessive(q_lower, clean_q, available_columns, schema_dict)
        if poss_rel:
            return poss_rel

        # 5. Try Noun-Adjunct / Direct Word-Order Patterns: "West Bengal capital", "Rahul Patel salary"
        noun_rel = self._parse_noun_adjunct(q_lower, clean_q, available_columns, schema_dict)
        if noun_rel:
            return noun_rel

        return None

    def _parse_reverse_lookup(
        self,
        q_lower: str,
        orig_q: str,
        available_columns: List[str],
        schema_dict: Optional[Dict[str, Any]]
    ) -> Optional[SemanticRelationship]:
        """Extract reverse lookup inquiries:

        - Which state has Kolkata as [its] capital?
        - Kolkata is the capital of which state?
        - Bangalore belongs to which state?
        - Which state does Bangalore belong to?
        - To which state does Bangalore belong?
        """
        # Pattern 1: Which [target_col] has [entity] as [its] [attr_col]
        # e.g. "Which state has Kolkata as its capital?"
        p1 = re.search(
            r"which\s+([a-zA-Z\s]+?)\s+has\s+([a-zA-Z0-9\s]+?)\s+as(?:\s+its)?\s+([a-zA-Z\s]+?)(?:\?|$)",
            q_lower
        )
        if p1:
            target_cand = p1.group(1).strip()
            entity_cand = p1.group(2).strip()
            attr_cand = p1.group(3).strip()
            resolved = self._resolve_reverse_entities(target_cand, entity_cand, attr_cand, available_columns, schema_dict)
            if resolved:
                return resolved

        # Pattern 2: [entity] is [the] [attr_col] of which [target_col]
        # e.g. "Kolkata is the capital of which state?"
        p2 = re.search(
            r"([a-zA-Z0-9\s]+?)\s+is(?:\s+the)?\s+([a-zA-Z\s]+?)\s+of\s+which\s+([a-zA-Z\s]+?)(?:\?|$)",
            q_lower
        )
        if p2:
            entity_cand = p2.group(1).strip()
            attr_cand = p2.group(2).strip()
            target_cand = p2.group(3).strip()
            resolved = self._resolve_reverse_entities(target_cand, entity_cand, attr_cand, available_columns, schema_dict)
            if resolved:
                return resolved

        # Pattern 3: Which [target_col] does [entity] belong to / To which [target_col] does [entity] belong
        # e.g. "Which state does Bangalore belong to?", "To which state does Bangalore belong?"
        p3 = re.search(
            r"(?:to\s+)?which\s+([a-zA-Z\s]+?)\s+does\s+([a-zA-Z0-9\s]+?)\s+(?:belong|belongs)(?:\s+to)?(?:\?|$)",
            q_lower
        )
        if p3:
            target_cand = p3.group(1).strip()
            entity_cand = p3.group(2).strip()
            resolved = self._resolve_reverse_entities(target_cand, entity_cand, None, available_columns, schema_dict)
            if resolved:
                return resolved

        # Pattern 4: [entity] belongs to which [target_col]
        # e.g. "Bangalore belongs to which state?"
        p4 = re.search(
            r"([a-zA-Z0-9\s]+?)\s+(?:belongs?\s+to|comes?\s+under|is\s+in)\s+which\s+([a-zA-Z\s]+?)(?:\?|$)",
            q_lower
        )
        if p4:
            entity_cand = p4.group(1).strip()
            target_cand = p4.group(2).strip()
            resolved = self._resolve_reverse_entities(target_cand, entity_cand, None, available_columns, schema_dict)
            if resolved:
                return resolved

        # Pattern 5: Which [target_col] has [entity]
        # e.g. "Which state has Bangalore?", "Which state has Kolkata?"
        p5 = re.search(
            r"which\s+([a-zA-Z\s]+?)\s+has\s+([a-zA-Z0-9\s]+?)(?:\?|$)",
            q_lower
        )
        if p5:
            target_cand = p5.group(1).strip()
            entity_cand = p5.group(2).strip()
            resolved = self._resolve_reverse_entities(target_cand, entity_cand, None, available_columns, schema_dict)
            if resolved:
                return resolved

        # Pattern 6: Which [target_col] is associated with [entity]
        # e.g. "Which state is associated with Kolkata?"
        p6 = re.search(
            r"which\s+([a-zA-Z\s]+?)\s+is\s+(?:associated\s+with|linked\s+to|connected\s+to)\s+([a-zA-Z0-9\s]+?)(?:\?|$)",
            q_lower
        )
        if p6:
            target_cand = p6.group(1).strip()
            entity_cand = p6.group(2).strip()
            resolved = self._resolve_reverse_entities(target_cand, entity_cand, None, available_columns, schema_dict)
            if resolved:
                return resolved

        # Pattern 7: Which [target_col] is [entity] in / located in
        # e.g. "Which state is Bangalore in?", "Which state is Kolkata located in?"
        p7 = re.search(
            r"which\s+([a-zA-Z\s]+?)\s+is\s+([a-zA-Z0-9\s]+?)\s+(?:in|located\s+in)(?:\?|$)",
            q_lower
        )
        if p7:
            target_cand = p7.group(1).strip()
            entity_cand = p7.group(2).strip()
            resolved = self._resolve_reverse_entities(target_cand, entity_cand, None, available_columns, schema_dict)
            if resolved:
                return resolved

        # Pattern 8: [entity] which [target_col] comes / [entity] is in which [target_col]
        # e.g. "hyd which state comes?", "Kolkata is in which state?"
        if not re.search(r"\bhas\s+(?:which|what)\b", q_lower):
            p8 = re.search(
                r"([a-zA-Z0-9\s]+?)\s+(?:is\s+in\s+which\s+([a-zA-Z\s]+?)|which\s+([a-zA-Z\s]+?)\s+(?:comes|is|belongs))(?:\?|$)",
                q_lower
            )
            if p8:
                entity_cand = p8.group(1).strip()
                target_cand = (p8.group(2) or p8.group(3) or "").strip()
                resolved = self._resolve_reverse_entities(target_cand, entity_cand, None, available_columns, schema_dict)
                if resolved:
                    return resolved

        return None

    def _resolve_reverse_entities(
        self,
        target_cand: str,
        entity_cand: str,
        attr_cand: Optional[str],
        available_columns: List[str],
        schema_dict: Optional[Dict[str, Any]]
    ) -> Optional[SemanticRelationship]:
        """Resolve entity, attr column, and target column for reverse lookup."""
        target_col = self._resolve_column_name(target_cand, available_columns, schema_dict)
        if not target_col:
            return None

        # Resolve entity candidate across dataset
        entity_res = self._resolve_entity_value(entity_cand, available_columns, preferred_col=None)
        if not entity_res:
            return None

        resolved_val, entity_col = entity_res

        # If attr_cand provided, verify or use it
        if attr_cand:
            matched_attr_col = self._resolve_column_name(attr_cand, available_columns, schema_dict)
            if matched_attr_col:
                entity_col = matched_attr_col

        if entity_col == target_col:
            return None

        return SemanticRelationship(
            subject_val=resolved_val,
            subject_col=entity_col,
            target_col=target_col,
            relationship_type="OF",
            direction="REVERSE",
            confidence=0.98,
            raw_entity_text=entity_cand,
            raw_attribute_text=target_cand,
            explanation=f"Reverse lookup {target_col} where {entity_col} is '{resolved_val}'."
        )

    def _parse_prepositional(
        self,
        q_lower: str,
        orig_q: str,
        available_columns: List[str],
        schema_dict: Optional[Dict[str, Any]]
    ) -> Optional[SemanticRelationship]:
        """Extract 'Y of X', 'Y for X', 'Y belonging to X', 'Y associated with X'."""
        prep_patterns = [
            r"(?:the\s+)?([a-zA-Z\s]+?)\s+of\s+([a-zA-Z0-9\s]+?)(?:\?|$)",
            r"(?:the\s+)?([a-zA-Z\s]+?)\s+for\s+([a-zA-Z0-9\s]+?)(?:\?|$)",
            r"(?:the\s+)?([a-zA-Z\s]+?)\s+belonging\s+to\s+([a-zA-Z0-9\s]+?)(?:\?|$)",
            r"(?:the\s+)?([a-zA-Z\s]+?)\s+associated\s+with\s+([a-zA-Z0-9\s]+?)(?:\?|$)",
        ]

        cleaned_q = self._strip_question_prefixes(q_lower)

        for pat in prep_patterns:
            m = re.search(pat, cleaned_q)
            if m:
                attr_cand = m.group(1).strip()
                entity_cand = m.group(2).strip()

                # Filter out obvious non-relationships e.g. "count of employees"
                if attr_cand in {"count", "total", "sum", "average", "avg", "list", "records", "number"}:
                    continue

                target_col = self._resolve_column_name(attr_cand, available_columns, schema_dict)
                if not target_col:
                    continue

                entity_res = self._resolve_entity_value(entity_cand, available_columns, exclude_col=target_col)
                if not entity_res:
                    continue

                resolved_val, entity_col = entity_res
                return SemanticRelationship(
                    subject_val=resolved_val,
                    subject_col=entity_col,
                    target_col=target_col,
                    relationship_type="HAS",
                    direction="DIRECT",
                    confidence=0.98,
                    raw_entity_text=entity_cand,
                    raw_attribute_text=attr_cand,
                    explanation=f"Lookup {target_col} for entity '{resolved_val}' ({entity_col})."
                )

        return None

    def _parse_inverted(
        self,
        q_lower: str,
        orig_q: str,
        available_columns: List[str],
        schema_dict: Optional[Dict[str, Any]]
    ) -> Optional[SemanticRelationship]:
        """Extract:

        - Which city is West Bengal's capital?
        - West Bengal has which capital?
        """
        # Pattern: Which [type] is [entity]'s [attr]
        m1 = re.search(
            r"which\s+[a-zA-Z\s]+?\s+is\s+([a-zA-Z0-9\s]+?)'s\s+([a-zA-Z\s]+?)(?:\?|$)",
            q_lower
        )
        if m1:
            entity_cand = m1.group(1).strip()
            attr_cand = m1.group(2).strip()
            return self._build_direct_relationship(entity_cand, attr_cand, available_columns, schema_dict)

        # Pattern: [entity] has which/what [attr]
        m2 = re.search(
            r"([a-zA-Z0-9\s]+?)\s+has\s+(?:which|what)\s+([a-zA-Z\s]+?)(?:\?|$)",
            q_lower
        )
        if m2:
            entity_cand = m2.group(1).strip()
            attr_cand = m2.group(2).strip()
            return self._build_direct_relationship(entity_cand, attr_cand, available_columns, schema_dict)

        return None

    def _parse_possessive(
        self,
        q_lower: str,
        orig_q: str,
        available_columns: List[str],
        schema_dict: Optional[Dict[str, Any]]
    ) -> Optional[SemanticRelationship]:
        """Extract:

        - West Bengal's capital
        - What is West Bengal's capital?
        - Rahul's salary
        - Rahul Patel's department
        - iPhone 13's price
        """
        cleaned_q = self._strip_question_prefixes(q_lower)

        # Pattern: [entity]'s [attr]
        m = re.search(r"([a-zA-Z0-9\s]+?)'s\s+([a-zA-Z\s]+?)(?:\?|$)", cleaned_q)
        if m:
            entity_cand = m.group(1).strip()
            attr_cand = m.group(2).strip()
            return self._build_direct_relationship(entity_cand, attr_cand, available_columns, schema_dict)

        # Pattern: [entity]s [attr] (missing apostrophe, e.g. "West Bengals capital")
        COMMON_PLURALS = {"employees", "customers", "products", "records", "people", "items", "users", "rows", "staff", "sales", "details", "data", "members"}
        words = cleaned_q.split()
        for i in range(len(words) - 1):
            w_lower = words[i].lower()
            if w_lower in COMMON_PLURALS:
                continue
            if words[i].endswith("s") and len(words[i]) > 3:
                cand_entity = " ".join(words[:i]) + (" " if i > 0 else "") + words[i][:-1]
                cand_attr = " ".join(words[i+1:])
                rel = self._build_direct_relationship(cand_entity, cand_attr, available_columns, schema_dict)
                if rel:
                    return rel

        return None

    def _parse_noun_adjunct(
        self,
        q_lower: str,
        orig_q: str,
        available_columns: List[str],
        schema_dict: Optional[Dict[str, Any]]
    ) -> Optional[SemanticRelationship]:
        """Extract noun-adjunct and direct word-order inquiries:

        - West Bengal capital
        - Telangana capital
        - Rahul Patel salary
        - iPhone 13 price
        """
        cleaned_q = self._strip_question_prefixes(q_lower)
        # Multi-filter / sentence guard: if query has filter connectors like ' in ', ' based in ', ' with ', ' and ',
        # it is a multi-condition query, not a simple noun-adjunct!
        if any(conn in cleaned_q for conn in [" in ", " based in ", " from ", " and ", " with ", " earning ", " having "]):
            return None

        words = cleaned_q.split()
        if len(words) < 2:
            return None

        # Try splitting into entity (prefix) and attribute (suffix)
        # Suffix can be 1, 2, or 3 words
        for split_point in range(len(words) - 1, 0, -1):
            entity_cand = " ".join(words[:split_point])
            attr_cand = " ".join(words[split_point:])

            # Verify attr_cand is an actual column or synonym
            target_col = self._resolve_column_name(attr_cand, available_columns, schema_dict)
            if not target_col:
                continue

            # Verify entity_cand matches a dataset value
            entity_res = self._resolve_entity_value(entity_cand, available_columns, exclude_col=target_col)
            if entity_res:
                resolved_val, entity_col = entity_res
                return SemanticRelationship(
                    subject_val=resolved_val,
                    subject_col=entity_col,
                    target_col=target_col,
                    relationship_type="HAS",
                    direction="DIRECT",
                    confidence=0.95,
                    raw_entity_text=entity_cand,
                    raw_attribute_text=attr_cand,
                    explanation=f"Lookup {target_col} for entity '{resolved_val}' ({entity_col})."
                )

        # Try reversed order: attribute (prefix) and entity (suffix)
        # e.g. "capital ts", "capital telangana", "salary rahul"
        for split_point in range(1, len(words)):
            attr_cand = " ".join(words[:split_point])
            entity_cand = " ".join(words[split_point:])

            target_col = self._resolve_column_name(attr_cand, available_columns, schema_dict)
            if not target_col:
                continue

            entity_res = self._resolve_entity_value(entity_cand, available_columns, exclude_col=target_col)
            if entity_res:
                resolved_val, entity_col = entity_res
                return SemanticRelationship(
                    subject_val=resolved_val,
                    subject_col=entity_col,
                    target_col=target_col,
                    relationship_type="HAS",
                    direction="DIRECT",
                    confidence=0.95,
                    raw_entity_text=entity_cand,
                    raw_attribute_text=attr_cand,
                    explanation=f"Lookup {target_col} for entity '{resolved_val}' ({entity_col})."
                )

        return None

    def _build_direct_relationship(
        self,
        entity_cand: str,
        attr_cand: str,
        available_columns: List[str],
        schema_dict: Optional[Dict[str, Any]]
    ) -> Optional[SemanticRelationship]:
        """Validate and construct a direct SemanticRelationship."""
        target_col = self._resolve_column_name(attr_cand, available_columns, schema_dict)
        if not target_col:
            return None

        entity_res = self._resolve_entity_value(entity_cand, available_columns, exclude_col=target_col)
        if not entity_res:
            return None

        resolved_val, entity_col = entity_res
        return SemanticRelationship(
            subject_val=resolved_val,
            subject_col=entity_col,
            target_col=target_col,
            relationship_type="HAS",
            direction="DIRECT",
            confidence=0.98,
            raw_entity_text=entity_cand,
            raw_attribute_text=attr_cand,
            explanation=f"Lookup {target_col} for entity '{resolved_val}' ({entity_col})."
        )

    def _strip_question_prefixes(self, text: str) -> str:
        """Remove boilerplate question prefixes and regional suffixes."""
        t = text.strip()
        for pat in self.QUESTION_CLEAN_PREFIXES:
            t = re.sub(pat, "", t, flags=re.IGNORECASE).strip()
        for pat in self.QUESTION_CLEAN_SUFFIXES:
            t = re.sub(pat, "", t, flags=re.IGNORECASE).strip()
        return t

    def _resolve_column_name(
        self,
        text: str,
        available_columns: List[str],
        schema_dict: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Resolve a candidate attribute token to an actual dataset column."""
        t_clean = text.strip().lower()

        # Reject comparison/filter phrases from being treated as column names
        if any(cmp_kw in t_clean for cmp_kw in ["over ", "under ", "more than", "less than", "greater than", "at least", "at most", "between", ">", "<"]):
            return None

        matched_col = None
        # 1. Exact match against column names
        for col in available_columns:
            if col.lower() == t_clean:
                matched_col = col
                break

        # 2. Check semantic column mapper
        if not matched_col:
            mapping = semantic_column_mapper.map_column(t_clean, available_columns, schema_dict)
            if mapping.get("column"):
                matched_col = mapping["column"]

        # 3. Check domain synonym dictionary
        if not matched_col:
            for canon_attr, synonyms in ATTRIBUTE_SYNONYMS.items():
                if t_clean in synonyms or any(s == t_clean for s in synonyms):
                    for col in available_columns:
                        col_l = col.lower()
                        if canon_attr in col_l or any(s in col_l for s in synonyms):
                            matched_col = col
                            break
                    if matched_col:
                        break

        # 4. Partial substring match
        if not matched_col:
            for col in available_columns:
                if t_clean in col.lower() or col.lower() in t_clean:
                    if len(t_clean) >= 3:
                        matched_col = col
                        break

        # 5. Dataset-aware spell checker for column typos (e.g. 'captial' -> 'capital', 'salry' -> 'salary')
        if not matched_col:
            try:
                from app.dataset.spell_checker import dataset_spell_checker
                matched_col = dataset_spell_checker.resolve_column(t_clean, available_columns)
            except Exception:
                pass

        if not matched_col:
            return None

        # Anti-confusion guard:
        # A candidate attribute token must NOT be an actual entity value in the dataset!
        # E.g. "Hyderabad" is a value in City, NOT an attribute name.
        vals = dataset_indexer.get_column_values(matched_col)
        for v in vals:
            if str(v).strip().lower() == t_clean:
                return None

        return matched_col

    def _resolve_entity_value(
        self,
        cand: str,
        available_columns: List[str],
        exclude_col: Optional[str] = None,
        preferred_col: Optional[str] = None
    ) -> Optional[Tuple[str, str]]:
        """Resolve entity string to (canonical_value, column_name) using exact, fuzzy, and phonetic search."""
        cand_clean = cand.strip().strip("'\"")
        if not cand_clean or len(cand_clean) < 2:
            return None

        # Filter search columns
        cols_to_search = [c for c in available_columns if c != exclude_col and not c.startswith("_internal_")]
        if preferred_col and preferred_col in cols_to_search:
            cols_to_search = [preferred_col] + [c for c in cols_to_search if c != preferred_col]

        # 0. Check Entity Alias & Shortcut Resolution (AP, TG, WB, HYD, BLR, HR, IT, etc.)
        alias_res = entity_alias_resolver.resolve_alias(cand_clean, available_columns=cols_to_search)
        if alias_res and alias_res.get("canonical_name"):
            canon_name = alias_res["canonical_name"]
            hints = alias_res.get("target_column_hints", [])
            # Try columns matching hints first
            sorted_cols = sorted(cols_to_search, key=lambda c: 0 if any(h.lower() in c.lower() for h in hints) else 1)
            for col in sorted_cols:
                res = dataset_value_resolver.resolve_value_in_column(canon_name, col)
                if res.get("resolved_value") and res.get("confidence", 0) >= 0.75:
                    return res["resolved_value"], col

        # 1. Exact match or value resolver in each column
        for col in cols_to_search:
            res = dataset_value_resolver.resolve_value_in_column(cand_clean, col)
            if res.get("resolved_value") and res.get("confidence", 0) >= 0.75:
                return res["resolved_value"], col

        # 2. Substring or token match in column values
        for col in cols_to_search:
            vals = dataset_indexer.get_column_values(col)
            found_token = extract_best_match_from_text(cand_clean, [str(v) for v in vals if v])
            if found_token:
                res = dataset_value_resolver.resolve_value_in_column(found_token, col)
                if res.get("resolved_value"):
                    return res["resolved_value"], col

        # 3. Phonetic matching for sound-alikes (e.g. "West Bengel", "Cikkim", "Kalkata")
        for col in cols_to_search:
            vals = dataset_indexer.get_column_values(col)
            for v in vals:
                if not v:
                    continue
                v_str = str(v)
                p_score = phonetic_match_score(cand_clean, v_str)
                if p_score >= 0.78:
                    return v_str, col

        # 4. Universal Dataset-Aware Spell Checker (14-stage typo resolution pipeline)
        try:
            from app.dataset.spell_checker import dataset_spell_checker
            spell_res = dataset_spell_checker.resolve_candidate(cand_clean, target_columns=cols_to_search)
            if spell_res and spell_res.confidence >= 0.70:
                return spell_res.canonical_value, spell_res.column
        except Exception:
            pass

        return None


semantic_relationship_parser = SemanticRelationshipParser()
