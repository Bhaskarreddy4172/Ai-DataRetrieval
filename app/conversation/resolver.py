"""Context and pronoun resolver for conversational multi-turn dialogue."""

import re
from typing import Any, Dict, List, Optional, Tuple
from app.query.schema import FilterCondition, StructuredQuery
from app.utils.fuzzy_match import expand_abbreviations, fuzzy_match_column, fuzzy_match_value, LOCATION_ABBREVIATIONS
from app.utils.logger import logger


class ConversationResolver:
    """Resolves pronouns ('it', 'they', 'that person'), ordinal references, and follow-up aggregations."""

    PRONOUN_PATTERNS = [
        r"\b(?:that|this)\s+(?:person|guy|customer|employee|product|car|vehicle|state|item)\b",
        r"\b(?:their|his|her|its)\b",
        r"\b(?:they|them|he|she|it)\b",
        r"\bthe same (?:person|customer|employee|product|one)\b",
    ]

    FOLLOWUP_PATTERNS = [
        r"^what about\s+(.+)$",
        r"^how about\s+(.+)$",
        r"^and for\s+(.+)$",
        r"^and in\s+(.+)$",
        r"^what of\s+(.+)$",
        r"^now in\s+(.+)$",
    ]

    ATTRIBUTE_KEYWORDS = {
        "department": ["department", "dept", "team", "division", "unit"],
        "designation": ["designation", "role", "title", "position", "job"],
        "salary": ["salary", "pay", "package", "earns", "earn", "earning", "compensation", "ctc", "income", "paid"],
        "city": ["city", "location", "office", "work", "works", "live", "lives", "station", "stationed", "based"],
        "performance score": ["performance", "score", "rating", "review", "evaluation", "performer"],
        "joining date": ["join", "joined", "joining", "hire", "hired", "start", "started", "date"],
        "employee id": ["employee id", "emp id", "id", "code"],
        "employee name": ["name", "full name", "person"],
        "population": ["population", "people", "inhabitants", "residents", "pop"],
        "no_of_females": ["female population", "females", "female", "women", "girls", "no of females", "no_of_females"],
        "no_of_males": ["male population", "males", "male", "men", "boys", "no of males", "no_of_males"],
        "literacy_rate_percent": ["literacy", "literacy rate", "literacy rate percent", "education", "literate"],
        "households": ["households", "houses", "homes", "families"],
        "area_sq_km": ["area", "area sq km", "area in sq km", "size"]
    }

    def resolve(
        self,
        question: str,
        session_history: List[Dict[str, Any]],
        available_columns: List[str]
    ) -> Tuple[str, Optional[StructuredQuery]]:
        """Resolve pronouns and context from session history into a clarified question or modified query."""
        if not session_history:
            return question, None

        last_turn = session_history[-1]
        last_query: Optional[StructuredQuery] = last_turn.get("query")
        last_results: List[Dict[str, Any]] = last_turn.get("results", [])

        q_clean = question.strip()
        q_lower = q_clean.lower()

        # Guard: If query is already a complete self-contained semantic question (e.g. "Which state has Kolkata as its capital?"),
        # do not hijack intra-sentence words like 'its' or 'what about'
        from app.query.semantic_parser import semantic_relationship_parser
        if semantic_relationship_parser.parse(q_clean, available_columns):
            return question, None

        # 1. First, check for Pronoun references: "What about her department?", "Where does she work?", "What is their salary?"
        has_pronoun = any(re.search(pat, q_lower) for pat in self.PRONOUN_PATTERNS)
        if has_pronoun:
            id_col, id_val = self._find_active_entity(session_history, available_columns)
            if id_val:
                # If this is a boolean question like "Is she from Hyderabad?" or "Does he earn > 10L?",
                # replace pronoun and return clarified string for boolean pipeline
                if any(q_lower.startswith(w + " ") for w in ["is", "are", "does", "did", "was", "were", "has", "have", "can"]):
                    clarified = self._replace_pronoun(q_clean, id_val)
                    logger.info(f"Resolved boolean pronoun to '{id_val}' ({id_col}): '{clarified}'")
                    return clarified, None

                # Check if question is asking for a specific attribute of this entity
                target_col = self._match_attribute_column(q_lower, available_columns)
                if target_col:
                    attr_query = StructuredQuery(
                        operation="FILTER",
                        conditions=[FilterCondition(column=id_col or available_columns[0], operator="=", value=id_val)],
                        select_columns=[id_col or available_columns[0], target_col],
                        limit=1,
                        explanation=f"Retrieve {target_col} for {id_val}"
                    )
                    return f"what is {id_val}'s {target_col}?", attr_query

                # Rewrite question with explicit entity name
                clarified = self._replace_pronoun(q_clean, id_val)
                logger.info(f"Resolved pronoun to '{id_val}' ({id_col}): '{clarified}'")
                return clarified, None

        # 2. Check for Ordinal references: "what about the second guy?", "second one", "next one", "who is next?"
        ord_m = re.search(r"\b(?:what about\s+)?(?:the\s+)?(second|2nd|third|3rd|fourth|4th|fifth|5th|next)\s*(?:guy|person|one|employee|customer|item)?\b", q_lower)
        if ord_m and last_query and (last_query.sort_column or last_query.target_column or last_query.operation in {"TOP_N", "MAX", "MIN", "RANK"}):
            ord_word = ord_m.group(1).lower()
            ord_map = {"second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5, "5th": 5}
            if ord_word == "next":
                prev_rank = getattr(last_query, "rank_offset", None) or 1
                rank_val = prev_rank + 1
            else:
                rank_val = ord_map.get(ord_word, 2)

            sort_col = last_query.sort_column or last_query.target_column or (available_columns[0] if available_columns else None)
            sort_order = last_query.sort_order or ("ASC" if last_query.operation in {"MIN", "BOTTOM_N"} else "DESC")
            new_q = StructuredQuery(
                operation="RANK",
                target_column=sort_col,
                sort_column=sort_col,
                sort_order=sort_order,
                rank_offset=rank_val,
                limit=1,
                explanation=f"Rank #{rank_val} for {sort_col} following previous turn"
            )
            return f"Who is #{rank_val} by {sort_col}?", new_q

        if any(pat in q_lower for pat in ["who is next", "who is the next", "who comes next", "next one", "next person"]):
            if last_results and "_internal_row_id" in last_results[0]:
                next_id = int(last_results[0]["_internal_row_id"]) + 1
                new_q = StructuredQuery(
                    operation="FILTER",
                    conditions=[FilterCondition(column="_internal_row_id", operator="=", value=next_id)],
                    limit=1,
                    explanation=f"Sequential next row (id {next_id})"
                )
                return "Who is next?", new_q

        # 3. Check for Follow-up pattern: "What about [Entity/Location]?" (no pronoun)
        for pat in self.FOLLOWUP_PATTERNS:
            m = re.match(pat, q_lower)
            if m:
                target_phrase = m.group(1).strip().rstrip("?")
                return self._resolve_followup(target_phrase, last_query, available_columns, question)

        # 4. Check for Ordinal references: "the highest one", "the top one"
        if ("the highest one" in q_lower or "the top one" in q_lower) and last_results:
            id_col, id_val = self._find_active_entity(session_history, available_columns)
            if id_val:
                return f"details for {id_val}", None

        return question, None

    def _match_attribute_column(self, q_lower: str, available_columns: List[str]) -> Optional[str]:
        """Match query text to an available column using semantic mapper and attribute keywords."""
        for col in available_columns:
            col_l = col.lower()
            # Direct name match (e.g. 'department' in q_lower)
            if col_l in q_lower:
                return col

        # Use semantic column mapper dynamically for any dataset
        try:
            from app.dataset.semantic_mapper import semantic_column_mapper
            words = q_lower.split()
            for i in range(len(words)):
                chunk = " ".join(words[i:i+2])
                mapped = semantic_column_mapper.map_column(chunk, available_columns)
                if mapped.get("column"):
                    return mapped["column"]
        except Exception:
            pass

        # Fallback to synonyms match
        for canonical_key, keywords in self.ATTRIBUTE_KEYWORDS.items():
            for col in available_columns:
                col_l = col.lower()
                if col_l == canonical_key or canonical_key in col_l:
                    if any(re.search(r"\b" + re.escape(kw) + r"\b", q_lower) for kw in keywords):
                        return col
        return None

    def _find_active_entity(
        self, session_history: List[Dict[str, Any]], available_columns: List[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Find active entity identifier by searching backwards through session history."""
        for turn in reversed(session_history):
            for row in turn.get("results", []):
                if isinstance(row, dict):
                    id_col, id_val = self._extract_identifier(row, available_columns)
                    if id_val:
                        return id_col, id_val
            # Check conditions in turn query
            q = turn.get("query")
            if q and hasattr(q, "conditions") and q.conditions:
                for cond in q.conditions:
                    if cond.operator == "=" and isinstance(cond.value, str):
                        return cond.column, str(cond.value)
        return None, None

    def _resolve_followup(
        self,
        target_phrase: str,
        last_query: Optional[StructuredQuery],
        available_columns: List[str],
        original_question: str
    ) -> Tuple[str, Optional[StructuredQuery]]:
        """Resolve 'what about X?' using the operation and filters from the previous query."""
        if not last_query:
            return original_question, None

        # If target phrase itself is a complete semantic query (e.g. "West Bengal's capital", "Rahul's salary"),
        # do NOT treat it as a continuation entity substitution!
        from app.query.semantic_parser import semantic_relationship_parser
        if semantic_relationship_parser.parse(target_phrase, available_columns):
            return target_phrase, None

        # Expand abbreviations on target phrase (e.g. "hyd" -> "Hyderabad")
        expanded_target = expand_abbreviations(target_phrase)

        # Find categorical column matching the entity
        city_col = None
        for c in available_columns:
            if any(term in c.lower() for term in ["city", "state", "location", "department", "category"]):
                city_col = c
                break

        target_col = city_col or (last_query.conditions[0].column if last_query.conditions else available_columns[0])
        matched_val = expanded_target.title()
        new_cond = FilterCondition(column=target_col, operator="=", value=matched_val)

        # Build modified query inheriting previous query's operation
        inherited_op = last_query.operation if last_query.operation in {"COUNT", "SUM", "AVERAGE", "MIN", "MAX", "MEDIAN"} else "FILTER"
        inherited_target = last_query.target_column

        # Keep other non-conflicting conditions
        preserved_conds = [c for c in last_query.conditions if c.column.lower() != target_col.lower()]
        preserved_conds.append(new_cond)

        new_query = StructuredQuery(
            operation=inherited_op,
            target_column=inherited_target,
            conditions=preserved_conds,
            limit=last_query.limit,
            explanation=f"Follow-up query for '{matched_val}' inheriting operation '{inherited_op}'"
        )
        return f"{inherited_op} for {matched_val}", new_query

    def _extract_identifier(self, entity_dict: Dict[str, Any], available_columns: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """Extract primary identifier string from an entity dictionary."""
        # Check priority columns
        for key in ["Village", "village", "Village_ID", "Capital", "capital", "State", "state", "Employee Name", "Customer Name", "Product Name", "Product", "Name"]:
            if key in entity_dict and entity_dict[key]:
                return key, str(entity_dict[key])

        # Check first non-id text column
        for c in available_columns:
            if c in entity_dict and not c.lower().endswith("id") and isinstance(entity_dict[c], str) and c != "_internal_row_id":
                return c, str(entity_dict[c])

        return None, None

    def _replace_pronoun(self, text: str, entity_name: str) -> str:
        """Replace pronoun in text with explicit entity name."""
        t = text
        for pat in [r"\btheir\b", r"\bhis\b", r"\bher\b", r"\bits\b"]:
            t = re.sub(pat, f"{entity_name}'s", t, flags=re.IGNORECASE)
        for pat in [r"\bthey\b", r"\bthem\b", r"\bhe\b", r"\bshe\b", r"\bit\b", r"\bthat person\b", r"\bthis person\b"]:
            t = re.sub(pat, entity_name, t, flags=re.IGNORECASE)
        return t


conversation_resolver = ConversationResolver()
