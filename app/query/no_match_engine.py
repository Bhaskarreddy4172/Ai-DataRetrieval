"""Universal No-Match & Unsupported Query Decision Engine.

Enforces: CLIENT DATA = ABSOLUTE SOURCE OF TRUTH.
Prevents any fallback hallucination from LLM / pretrained weights when client dataset
does not contain the requested entity, attribute, or value.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from app.dataset.indexer import dataset_indexer
from app.dataset.semantic_mapper import semantic_column_mapper
from app.dataset.value_resolver import dataset_value_resolver
from app.dataset.alias_resolver import entity_alias_resolver
from app.utils.fuzzy_match import extract_best_match_from_text
from app.utils.logger import logger


@dataclass
class NoMatchDiagnosis:
    """Diagnostic outcome of query inspection."""
    category: str  # VALID_MATCH, ENTITY_NOT_FOUND, COLUMN_NOT_FOUND, NO_MATCHING_ROWS, NULL_VALUE, AMBIGUOUS, UNSUPPORTED
    entity_text: Optional[str] = None
    attribute_text: Optional[str] = None
    target_column: Optional[str] = None
    resolved_entity: Optional[str] = None
    explanation: Optional[str] = None
    user_message: Optional[str] = None


class NoMatchEngine:
    """Evaluates user questions against dataset schema and indexed values to detect missing data/fields."""

    # Common unsupported attributes for tabular / geographic / organizational data
    KNOWN_UNSUPPORTED_ATTRIBUTES = [
        "population", "chief minister", "cm", "governor", "gdp", "area", "size", "weather",
        "temperature", "climate", "rainfall", "crime", "crime rate", "literacy", "literacy rate",
        "president", "prime minister", "pm", "mayor", "currency", "timezone", "time zone",
        "pin code", "pincode", "zip code", "zipcode", "std code", "calling code", "iso code",
        "official animal", "state animal", "state bird", "state flower", "state tree",
        "ruling party", "mla", "mp", "constituency", "assembly seats", "lok sabha seats",
        "covid cases", "tourist places", "tourist attractions", "airports", "airports count"
    ]

    QUESTION_CLEAN_PREFIXES = [
        r"^what\s+is\s+(?:the\s+)?",
        r"^what\s+are\s+(?:the\s+)?",
        r"^what's\s+(?:the\s+)?",
        r"^whats\s+(?:the\s+)?",
        r"^which\s+(?:city|state|department|employee|product|person|place)\s+is\s+(?:the\s+)?",
        r"^which\s+(?:city|state|department|employee|product|person|place)\s+has\s+",
        r"^tell\s+me\s+(?:the\s+)?(?:about\s+)?",
        r"^give\s+me\s+(?:the\s+)?",
        r"^show\s+me\s+(?:the\s+)?",
        r"^can\s+you\s+tell\s+me\s+(?:the\s+)?",
        r"^do\s+you\s+know\s+(?:the\s+)?",
        r"^find\s+(?:the\s+)?",
        r"^search\s+(?:for\s+)?(?:the\s+)?",
        r"^who\s+is\s+(?:the\s+)?",
        r"^who\s+are\s+(?:the\s+)?",
    ]

    def diagnose_query(
        self,
        question: str,
        available_columns: List[str],
        df: Optional[pd.DataFrame] = None,
        schema_dict: Optional[Dict[str, Any]] = None
    ) -> NoMatchDiagnosis:
        """Inspect question against schema and values to detect missing entity, column, or ambiguity."""
        clean_q = question.strip()
        clean_q = clean_q.replace("’", "'").replace("`", "'")
        clean_q = re.sub(r"\s+", " ", clean_q)
        q_lower = clean_q.lower().rstrip("?!.,;")

        cols_clean = [c for c in available_columns if not c.startswith("_internal_")]
        cols_lower = [c.lower() for c in cols_clean]

        # 1. Detect Bare Attribute / Ambiguous Query (e.g. "What is the capital?", "capital?", "What is the salary?")
        ambiguity_diag = self._check_ambiguity(q_lower, clean_q, cols_clean)
        if ambiguity_diag:
            return ambiguity_diag

        # 2. Extract Candidate Entity & Attribute pairs from query
        extracted = self._extract_entity_and_attribute(q_lower, clean_q)
        if extracted:
            entity_cand, attr_cand, pattern_type = extracted

            # Check if attribute exists in schema
            mapped_col = self._map_to_schema(attr_cand, cols_clean, schema_dict)

            # Check if entity exists in dataset
            resolved_entity, entity_col = self._resolve_entity_in_dataset(entity_cand, cols_clean)

            # Case A: Column is NOT in dataset schema, but Entity IS in dataset
            if not mapped_col and resolved_entity:
                msg = f'The uploaded dataset does not contain {attr_cand} information, so I cannot determine it from the provided data.'
                return NoMatchDiagnosis(
                    category="COLUMN_NOT_FOUND",
                    entity_text=entity_cand,
                    attribute_text=attr_cand,
                    resolved_entity=resolved_entity,
                    explanation=f"Entity '{resolved_entity}' found in column '{entity_col}', but attribute '{attr_cand}' does not exist in dataset columns {cols_clean}.",
                    user_message=msg
                )

            # Case B: Entity is NOT in dataset, but Attribute IS in dataset schema
            if mapped_col and not resolved_entity:
                # Confirm entity_cand is not a stopword or noise
                if len(entity_cand) >= 2 and entity_cand not in {"what", "which", "state", "city", "capital", "the", "a", "is"}:
                    # Format user message exactly per specification
                    entity_type_label = self._get_entity_type_label(cols_clean)
                    msg = f'No matching {entity_type_label} or entity was found in the uploaded dataset for "{entity_cand.title()}".'
                    return NoMatchDiagnosis(
                        category="ENTITY_NOT_FOUND",
                        entity_text=entity_cand.title(),
                        attribute_text=attr_cand,
                        target_column=mapped_col,
                        explanation=f"Attribute '{mapped_col}' exists in schema, but entity '{entity_cand}' was not found in dataset values.",
                        user_message=msg
                    )

            # Case C: Both Entity and Attribute exist -> Check for NULL value
            if mapped_col and resolved_entity and df is not None and not df.empty and entity_col in df.columns and mapped_col in df.columns:
                matched_rows = df[df[entity_col].astype(str).str.lower() == str(resolved_entity).lower()]
                if not matched_rows.empty:
                    val = matched_rows[mapped_col].iloc[0]
                    if pd.isna(val) or str(val).strip() == "" or str(val).lower() in {"nan", "none", "null", "n/a", "-"}:
                        msg = f"{resolved_entity} was found in the dataset, but its {mapped_col} value is not available."
                        return NoMatchDiagnosis(
                            category="NULL_VALUE",
                            entity_text=resolved_entity,
                            attribute_text=mapped_col,
                            target_column=mapped_col,
                            resolved_entity=resolved_entity,
                            explanation=f"Entity '{resolved_entity}' exists, but column '{mapped_col}' cell is NULL or blank.",
                            user_message=msg
                        )

            # Case D: Pattern was reverse lookup e.g. "Which state has XYZ as capital?"
            if pattern_type == "REVERSE_LOOKUP":
                # attr_cand was e.g. 'capital', entity_cand was 'XYZ'
                if mapped_col and not resolved_entity:
                    msg = f'No matching records were found in the uploaded dataset for "{entity_cand.title()}".'
                    return NoMatchDiagnosis(
                        category="NO_MATCHING_ROWS",
                        entity_text=entity_cand.title(),
                        attribute_text=attr_cand,
                        target_column=mapped_col,
                        explanation=f"Reverse search for '{mapped_col}' = '{entity_cand}' yielded 0 records.",
                        user_message=msg
                    )

        # 3. Check for standalone unsupported topics (e.g. "Who is the Chief Minister of Telangana?")
        for unsupp_attr in self.KNOWN_UNSUPPORTED_ATTRIBUTES:
            if re.search(r"\b" + re.escape(unsupp_attr) + r"\b", q_lower):
                # Check if this attribute is present in schema
                mapped_col = self._map_to_schema(unsupp_attr, cols_clean, schema_dict)
                if not mapped_col:
                    # Find any entity mentioned in query
                    ent_found, ent_col = self._find_any_entity_in_text(q_lower, cols_clean)
                    disp_attr = unsupp_attr.title() if len(unsupp_attr) > 3 else unsupp_attr.upper()
                    msg = f"The uploaded dataset does not contain {disp_attr} information, so I cannot determine it from the provided data."
                    return NoMatchDiagnosis(
                        category="COLUMN_NOT_FOUND",
                        attribute_text=disp_attr,
                        entity_text=ent_found,
                        explanation=f"Requested field '{disp_attr}' is not present in columns {cols_clean}.",
                        user_message=msg
                    )

        return NoMatchDiagnosis(category="VALID_MATCH")

    def _check_ambiguity(
        self,
        q_lower: str,
        orig_q: str,
        available_columns: List[str]
    ) -> Optional[NoMatchDiagnosis]:
        """Check if query is asking for an attribute without specifying any entity."""
        # Clean question prefixes
        stripped = q_lower
        for p in self.QUESTION_CLEAN_PREFIXES:
            stripped = re.sub(p, "", stripped).strip()

        stripped = stripped.rstrip("?!.,;").strip()

        # If stripped question is just an attribute word (e.g. "capital", "the capital", "salary", "state")
        for col in available_columns:
            col_l = col.lower()
            if stripped in {col_l, f"the {col_l}", f"a {col_l}", f"{col_l}s", f"all {col_l}s"}:
                # User asked "What is the capital?" or "capital?"
                entity_label = self._get_entity_type_label(available_columns)
                msg = f"Could you specify which {entity_label} you mean?"
                return NoMatchDiagnosis(
                    category="AMBIGUOUS",
                    attribute_text=col,
                    target_column=col,
                    explanation=f"Query only specified target column '{col}' without any entity or filter condition.",
                    user_message=msg
                )

        # Pattern: "What is the capital?" or "Which capital?"
        bare_m = re.match(r"^(?:what|which)\s+is\s+(?:the\s+)?([a-zA-Z\s]+)$", q_lower)
        if bare_m:
            cand = bare_m.group(1).strip()
            for col in available_columns:
                if cand in {col.lower(), f"{col.lower()}s"}:
                    entity_label = self._get_entity_type_label(available_columns)
                    return NoMatchDiagnosis(
                        category="AMBIGUOUS",
                        attribute_text=col,
                        target_column=col,
                        explanation=f"Query is asking for '{col}' without specifying an entity.",
                        user_message=f"Could you specify which {entity_label} you mean?"
                    )

        return None

    def _extract_entity_and_attribute(
        self,
        q_lower: str,
        orig_q: str
    ) -> Optional[Tuple[str, str, str]]:
        """Extract (entity_candidate, attribute_candidate, pattern_type) from text.
        
        Returns:
            Tuple of (entity, attribute, pattern_type)
        """
        # Pattern 1: Reverse lookup: "Which state has {val} as its {attr}?" / "Which state has {val} as capital?"
        rev_m = re.search(r"which\s+([a-zA-Z\s]+?)\s+has\s+([a-zA-Z0-9\s]+?)\s+as\s+(?:its\s+)?([a-zA-Z\s]+?)(?:\?|$)", q_lower)
        if rev_m:
            entity_type = rev_m.group(1).strip()
            entity_val = rev_m.group(2).strip()
            if not any(w in entity_val.split() for w in ["highest", "lowest", "max", "min", "top", "bottom", "average", "avg", "mean", "sum", "count", "more", "less", "least", "most"]):
                attr_cand = rev_m.group(3).strip()
                return entity_val, attr_cand, "REVERSE_LOOKUP"

        # Pattern 1b: "Which state has {val} capital?"
        rev_m2 = re.search(r"which\s+([a-zA-Z\s]+?)\s+has\s+([a-zA-Z0-9\s]+?)\s+(?:as\s+)?(capital|state|salary|department)(?:\?|$)", q_lower)
        if rev_m2:
            entity_val = rev_m2.group(2).strip()
            if not any(w in entity_val.split() for w in ["highest", "lowest", "max", "min", "top", "bottom", "average", "avg", "mean", "sum", "count", "more", "less", "least", "most", "the"]):
                attr_cand = rev_m2.group(3).strip()
                return entity_val, attr_cand, "REVERSE_LOOKUP"

        # Pattern 1c: "Which state's capital is {val}?" / "Which state capital is {val}?"
        rev_m3 = re.search(r"which\s+[a-zA-Z\s]+?'s\s+([a-zA-Z\s]+?)\s+is\s+([a-zA-Z0-9\s]+?)(?:\?|$)", q_lower)
        if rev_m3:
            attr_cand = rev_m3.group(1).strip()
            entity_val = rev_m3.group(2).strip()
            return entity_val, attr_cand, "REVERSE_LOOKUP"

        # Pattern 1d: "{val} is the {attr} of which {type}?" (e.g. "Kolkata is the capital of which state?")
        rev_m4 = re.search(r"([a-zA-Z0-9\s]+?)\s+is\s+(?:the\s+)?([a-zA-Z\s]+?)\s+of\s+which\s+([a-zA-Z\s]+?)(?:\?|$)", q_lower)
        if rev_m4:
            entity_val = rev_m4.group(1).strip()
            attr_cand = rev_m4.group(2).strip()
            return entity_val, attr_cand, "REVERSE_LOOKUP"

        # Pattern 2: "What is the {attr} of {entity}?" / "{attr} of {entity}"
        of_m = re.search(r"(?:what\s+is\s+(?:the\s+)?)?([a-zA-Z\s]+?)\s+of\s+([a-zA-Z0-9\s\-]+?)(?:\?|$)", q_lower)
        if of_m:
            attr_cand = of_m.group(1).strip()
            entity_cand = of_m.group(2).strip()
            for p in self.QUESTION_CLEAN_PREFIXES:
                attr_cand = re.sub(p, "", attr_cand).strip()
            if attr_cand and entity_cand and not any(entity_cand.startswith(w) for w in ["which", "what", "whom", "where"]):
                return entity_cand, attr_cand, "OF_PREPOSITION"

        # Pattern 3: Possessive: "{entity}'s {attr}"
        poss_m = re.search(r"([a-zA-Z0-9\s\-]+?)'s\s+([a-zA-Z\s]+?)(?:\?|$)", q_lower)
        if poss_m:
            entity_cand = poss_m.group(1).strip()
            attr_cand = poss_m.group(2).strip()
            for p in self.QUESTION_CLEAN_PREFIXES:
                entity_cand = re.sub(p, "", entity_cand).strip()
            if entity_cand and attr_cand:
                return entity_cand, attr_cand, "POSSESSIVE"

        # Pattern 4: "Who is the {attr} of {entity}?" (e.g. "Who is the Chief Minister of Telangana?")
        who_m = re.search(r"who\s+is\s+(?:the\s+)?([a-zA-Z\s]+?)\s+(?:of|for|in)\s+([a-zA-Z0-9\s\-]+?)(?:\?|$)", q_lower)
        if who_m:
            attr_cand = who_m.group(1).strip()
            entity_cand = who_m.group(2).strip()
            return entity_cand, attr_cand, "WHO_PREPOSITION"

        # Pattern 5: Noun adjunct: "{entity} {attr}" e.g. "Telangana population", "Wakanda capital"
        for unsupp in self.KNOWN_UNSUPPORTED_ATTRIBUTES:
            m = re.search(r"\b" + re.escape(unsupp) + r"\b", q_lower)
            if m:
                idx = m.start()
                before = q_lower[:idx].strip()
                for p in self.QUESTION_CLEAN_PREFIXES:
                    before = re.sub(p, "", before).strip()
                if before and len(before) >= 2:
                    return before, unsupp, "NOUN_ADJUNCT"

        return None

    def _map_to_schema(
        self,
        attr_text: str,
        available_columns: List[str],
        schema_dict: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Check if attribute text corresponds to any available dataset column."""
        if not attr_text:
            return None
        attr_l = attr_text.lower().strip()

        # Direct column name match
        for c in available_columns:
            if c.lower() == attr_l or attr_l in c.lower():
                return c

        # Semantic column mapper check
        mapping = semantic_column_mapper.map_column(attr_l, available_columns, schema_dict)
        if mapping.get("column"):
            return mapping["column"]

        # Dataset spell checker for column typos (e.g. 'captial' -> 'capital', 'salry' -> 'salary')
        try:
            from app.dataset.spell_checker import dataset_spell_checker
            spell_col = dataset_spell_checker.resolve_column(attr_l, available_columns)
            if spell_col:
                return spell_col
        except Exception:
            pass

        return None

    def _resolve_entity_in_dataset(
        self,
        entity_cand: str,
        available_columns: List[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve entity string against indexed dataset values across columns."""
        if not entity_cand or len(entity_cand) < 2:
            return None, None

        # 1. Check entity alias resolver (AP, TG, WB, etc.)
        alias_res = entity_alias_resolver.resolve_alias(entity_cand, available_columns=available_columns)
        if alias_res and alias_res.get("canonical_name"):
            canon = alias_res["canonical_name"]
            for col in available_columns:
                res = dataset_value_resolver.resolve_value_in_column(canon, col)
                if res.get("resolved_value") and res.get("confidence", 0) >= 0.75:
                    return res["resolved_value"], col

        # 2. Check exact and fuzzy match in each column
        for col in available_columns:
            res = dataset_value_resolver.resolve_value_in_column(entity_cand, col)
            if res.get("resolved_value") and res.get("confidence", 0) >= 0.75:
                return res["resolved_value"], col

        # 3. Substring match
        for col in available_columns:
            vals = dataset_indexer.get_column_values(col)
            found_tok = extract_best_match_from_text(entity_cand, [str(v) for v in vals if v])
            if found_tok:
                res = dataset_value_resolver.resolve_value_in_column(found_tok, col)
                if res.get("resolved_value"):
                    return res["resolved_value"], col

        # 4. Universal dataset-aware spell checker
        try:
            from app.dataset.spell_checker import dataset_spell_checker
            spell_res = dataset_spell_checker.resolve_candidate(entity_cand, target_columns=available_columns)
            if spell_res and spell_res.confidence >= 0.70:
                return spell_res.canonical_value, spell_res.column
        except Exception:
            pass

        return None, None

    def _find_any_entity_in_text(
        self,
        text: str,
        available_columns: List[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Find any dataset value mentioned anywhere in text."""
        for col in available_columns:
            vals = dataset_indexer.get_column_values(col)
            val_strs = [str(v) for v in vals if v and len(str(v)) > 2]
            found_token = extract_best_match_from_text(text, val_strs)
            if found_token:
                return found_token, col

        # Universal dataset-aware spell checker for tokens in text
        try:
            from app.dataset.spell_checker import dataset_spell_checker
            words = text.split()
            for w in words:
                if len(w) >= 3 and w.lower() not in {"what", "which", "state", "city", "capital", "the", "who"}:
                    s_res = dataset_spell_checker.resolve_candidate(w, target_columns=available_columns)
                    if s_res and s_res.confidence >= 0.75:
                        return s_res.canonical_value, s_res.column
        except Exception:
            pass

        return None, None

    def _get_entity_type_label(self, available_columns: List[str]) -> str:
        """Infer user-friendly label for primary entity column (e.g. 'state', 'employee', 'product')."""
        cols_l = [c.lower() for c in available_columns]
        for c in cols_l:
            if "state" in c:
                return "state"
            if "employee" in c or "name" in c or "person" in c:
                return "employee"
            if "product" in c or "item" in c:
                return "product"
            if "customer" in c or "client" in c:
                return "customer"
        return "state or entity"


no_match_engine = NoMatchEngine()

