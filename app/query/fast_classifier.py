"""Fast Deterministic Query Classifier: classifies operations, scopes, and metrics locally without LLM calls.

Provides sub-millisecond intent and scope resolution for deterministic execution:
- Intent: SUM, COUNT, AVERAGE, MEDIAN, MIN, MAX, TOP_N, BOTTOM_N, COMPARE, DIFFERENCE, LOOKUP, RANGE, VARIANCE, STD
- Scope: ALL_STATES, ALL_VILLAGES, SELECTED_STATES, SINGLE_STATE, SINGLE_VILLAGE, STANDALONE, GENERIC
- Metric: Population, No_of_Males, No_of_Females, Literacy_Rate_Percent, Area_Sq_Km, Households, etc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from app.utils.normalization import normalize_question
from app.utils.fuzzy_match import expand_abbreviations


@dataclass
class FastClassificationResult:
    """Output of FastQueryClassifier."""
    is_fast_path: bool = False
    intent: str = "LOOKUP"  # SUM, COUNT, AVERAGE, MEDIAN, MIN, MAX, TOP_N, BOTTOM_N, COMPARE, DIFFERENCE, LOOKUP, RANGE, VARIANCE, STD
    scope: str = "GENERIC"  # GLOBAL, FILTERED, ALL_STATES, ALL_VILLAGES, SELECTED_STATES, SINGLE_STATE, SINGLE_VILLAGE, STANDALONE, GENERIC
    scope_type: str = "GLOBAL"  # GLOBAL, CURRENT_ENTITY, SELECTED_ENTITIES, FILTERED, PARENT_CHILD, CHILD_DATASET, MULTI_ENTITY
    return_entity: str = "Record"  # "Village", "State", "Capital", "Record"
    filter_entity: Optional[str] = None  # "State", "Capital", etc.
    filter_value: Optional[Any] = None  # "Andhra Pradesh", ["Telangana", "Andhra Pradesh"]
    filters: List[Dict[str, Any]] = field(default_factory=list)
    metric: str = "Population"
    entities: List[str] = field(default_factory=list)
    secondary_metric: Optional[str] = None
    n_limit: Optional[int] = None
    order: Optional[str] = None  # ASC, DESC
    group_by: Optional[str] = None  # State, Capital, Village, etc.
    original_question: str = ""
    normalized_question: str = ""

    def to_plan_dict(self) -> Dict[str, Any]:
        """Convert to Section 2/36 standardized query plan dict."""
        return {
            "intent": self.intent,
            "entity_type": self.return_entity,
            "return_entity": self.return_entity,
            "metric": self.metric,
            "filters": self.filters,
            "scope": self.scope,
            "scope_type": self.scope_type,
            "order": self.order,
            "limit": self.n_limit or 1,
            "filter_entity": self.filter_entity,
            "filter_value": self.filter_value
        }


class FastQueryClassifier:
    """Lightweight rule-based and regex classifier for instant query dispatch."""

    METRIC_MAP: Dict[str, str] = {
        "population": "Population",
        "pop": "Population",
        "people": "Population",
        "ppl": "Population",
        "inhabitants": "Population",
        "residents": "Population",
        "persons": "Population",
        "entha mandi": "Population",
        "entha mandhi": "Population",
        "kitne log": "Population",
        "kitne log hai": "Population",
        "kitni population": "Population",
        "total ppl": "Population",
        "how much ppl": "Population",
        "janaabha": "Population",
        "janasankhya": "Population",
        "aabadi": "Population",
        "males": "No_of_Males",
        "male": "No_of_Males",
        "men": "No_of_Males",
        "purush": "No_of_Males",
        "purushulu": "No_of_Males",
        "male population": "No_of_Males",
        "females": "No_of_Females",
        "female": "No_of_Females",
        "women": "No_of_Females",
        "mahilalu": "No_of_Females",
        "aadavaallu": "No_of_Females",
        "stree": "No_of_Females",
        "female population": "No_of_Females",
        "literacy": "Literacy_Rate_Percent",
        "literacy rate": "Literacy_Rate_Percent",
        "literacy rate percent": "Literacy_Rate_Percent",
        "education": "Literacy_Rate_Percent",
        "chaduvu": "Literacy_Rate_Percent",
        "aksharyata": "Literacy_Rate_Percent",
        "area": "Area_Sq_Km",
        "area sq km": "Area_Sq_Km",
        "size": "Area_Sq_Km",
        "kshetraphal": "Area_Sq_Km",
        "households": "Households",
        "houses": "Households",
        "homes": "Households",
        "families": "Households",
        "illu": "Households",
        "intlu": "Households",
        "ghar": "Households",
        "villages": "VILLAGE_COUNT",
        "village count": "VILLAGE_COUNT",
        "number of villages": "VILLAGE_COUNT",
        "count of villages": "VILLAGE_COUNT",
    }

    ALL_STATES_PATTERNS = [
        r"\ball\s+states\b",
        r"\bevery\s+state\b",
        r"\bindia'?s?\s+total\b",
        r"\btotal\s+population\s+of\s+india\b",
        r"\bpopulation\s+of\s+india\b",
        r"\bentire\s+country\b",
        r"\bacross\s+all\s+states\b",
        r"\bin\s+all\s+states\b",
        r"\bper\s+state\b",
        r"\beach\s+state\b",
        r"\bsab\s+states\b",
        r"\bsabhi\s+states\b",
        r"\bsab\s+rajya\b",
        r"\bsabhi\s+rajya\b",
        r"\banni\s+states\b",
        r"\banni\s+rastralu\b",
        r"\banni\s+rashtralu\b",
    ]

    ALL_VILLAGES_PATTERNS = [
        r"\ball\s+villages\b",
        r"\bacross\s+all\s+villages\b",
        r"\bevery\s+village\b",
        r"\bin\s+all\s+villages\b",
    ]

    def is_global_query(self, question: str, session_id: Optional[str] = None) -> bool:
        """Deterministically determine if question is a GLOBAL query across all states/villages."""
        norm_q = normalize_question(question).lower()
        exp_q = expand_abbreviations(norm_q).lower()

        # 1. Obvious all-states / all-villages patterns
        if any(re.search(pat, norm_q) or re.search(pat, exp_q) for pat in self.ALL_STATES_PATTERNS + self.ALL_VILLAGES_PATTERNS):
            return True

        # Check if this is a follow-up comparison in an active session
        if session_id:
            try:
                from app.conversation.context import conversation_manager
                last_comp = conversation_manager.get_last_comparison(session_id)
                if last_comp and last_comp.get("left_entity") and last_comp.get("right_entity"):
                    comp_followup_pats = [
                        r"\bwhich\s+village\s+has\s+(?:more|fewer|less|higher|lower)\b",
                        r"\bwhich\s+one\s+has\s+(?:more|fewer|less|higher|lower)\b",
                        r"\bwhich\s+one\s+is\s+(?:more|less|higher|lower)\b",
                        r"\bhow\s+many\s+(?:more|fewer|less)\b",
                        r"\bhow\s+much\s+(?:more|fewer|less|higher|lower)\b",
                        r"\b(?:what\s+is\s+the\s+)?population\s+difference\b",
                        r"\bdifference\s+between\s+them\b",
                    ]
                    if any(re.search(p, norm_q) for p in comp_followup_pats):
                        return False
            except Exception:
                pass

        # 2. Open state extreme queries without named states (e.g. "which state has more population?", "who has the most people?")
        state_extreme_pat = r"\b(?:which\s+state|what\s+state|state\s+with|who\s+has|who\s+have|which\s+is\s+the\s+state|states\s+with)\b"
        if re.search(state_extreme_pat, norm_q) or re.search(r"^\s*who\s+(?:has|have|is)\b", norm_q):
            has_dir = any(re.search(r"\b" + re.escape(w) + r"\b", norm_q) for w in [
                "highest", "maximum", "max", "most", "largest", "more", "higher", "greater", "greatest", "top", "best",
                "lowest", "minimum", "min", "least", "smallest", "less", "fewer", "fewest", "lower", "bottom", "worst",
                "populated"
            ])
            if has_dir:
                states = self.extract_states(exp_q)
                if len(states) < 2:
                    return True

        # 3. Open village extreme queries (e.g. "which village has the highest population?")
        village_extreme_pat = r"\b(?:which\s+village|what\s+village|village\s+with)\b"
        if re.search(village_extreme_pat, norm_q):
            has_v_ext = any(re.search(r"\b" + re.escape(w) + r"\b", norm_q) for w in [
                "highest", "maximum", "max", "most", "largest",
                "lowest", "minimum", "min", "least", "smallest",
                "top", "bottom", "more", "less", "fewer", "higher", "lower"
            ])
            if has_v_ext:
                states = self.extract_states(exp_q)
                # GLOBAL ONLY IF NO STATE / LOCATION FILTERS ARE SPECIFIED!
                if len(states) == 0:
                    return True

        # 4. Global totals or averages without specific entity (e.g. "what is the total population?", "average population")
        if re.search(r"\b(?:total|sum\s+of|average|mean|median)\s+(?:population|people|villages|males|females)\b", norm_q):
            states = self.extract_states(exp_q)
            if not states:
                return True

        # 5. Global rankings (e.g. "rank all states", "top 5 states", "rank states by population")
        if re.search(r"\b(?:rank|ranking|top\s+\d+|bottom\s+\d+)\s+(?:all\s+)?(?:states|villages)\b", norm_q):
            return True

        return False

    def extract_metric(self, text: str, default: str = "Population") -> str:
        """Extract the target metric column from text."""
        t_low = text.lower()
        # Multi-word first
        for phrase, col in sorted(self.METRIC_MAP.items(), key=lambda x: len(x[0]), reverse=True):
            if re.search(r"\b" + re.escape(phrase) + r"\b", t_low):
                return col
        return default

    def extract_states(self, text: str) -> List[str]:
        """Extract all registered Indian state names and abbreviations from text in order."""
        from app.dataset.registry import parent_child_registry
        reg_states = parent_child_registry.get_registered_child_datasets()
        codes_map = parent_child_registry.get_state_codes_map()

        norm_text = normalize_question(text).lower()
        found: List[Tuple[str, int]] = []

        # 1. Full state names (multi-word first)
        for st in sorted(reg_states.keys(), key=len, reverse=True):
            pat = r"\b" + re.escape(st.lower()) + r"\b"
            for m in re.finditer(pat, norm_text):
                found.append((st, m.start()))

        # 2. State codes and abbreviations
        words = [(m.group().lower(), m.start()) for m in re.finditer(r"\b[a-zA-Z]{2,4}\b", norm_text)]
        english_stop_words = {"or", "in", "is", "to", "at", "an", "on", "it", "so", "by", "of", "if", "no", "do", "and", "all", "top", "sum", "lo", "ki", "ka", "ke", "me", "se", "ko", "bro", "plz", "pls"}
        for w, pos in words:
            if w in english_stop_words:
                # Disambiguate 'ka': Karnataka state code vs Hindi postposition
                if w == "ka":
                    is_karnataka = bool(re.search(r"\b(?:in|from|for|state|of)\s+ka\b", norm_text)) or bool(re.search(r"\bKA\b", text))
                    if not is_karnataka:
                        continue
                else:
                    continue
            if w in codes_map:
                canonical = codes_map[w]
                if canonical in reg_states and not any(f[0] == canonical for f in found):
                    found.append((canonical, pos))

        # 3. Multi-hop capital references (e.g. "whose capital is Hyderabad", "for capital Amaravati")
        if not found and re.search(r"\b(?:whose\s+capital\s+is|for\s+capital|capital\s+is|of\s+capital|capital\s+of|capital)\b", norm_text):
            main_df = parent_child_registry.get_main_dataframe()
            if main_df is not None and not main_df.empty:
                cap_col = next((c for c in main_df.columns if str(c).strip().lower() == "capital"), None)
                st_col = next((c for c in main_df.columns if str(c).strip().lower() == "state"), None)
                if cap_col and st_col:
                    for _, r in main_df.iterrows():
                        cap = str(r[cap_col]).strip().lower()
                        st = str(r[st_col]).strip()
                        if cap and re.search(r"\b" + re.escape(cap) + r"\b", norm_text):
                            if st in reg_states and not any(f[0] == st for f in found):
                                found.append((st, norm_text.find(cap)))
                                break

        found.sort(key=lambda x: x[1])
        res: List[str] = []
        seen: Set[str] = set()
        for st, _ in found:
            if st not in seen:
                seen.add(st)
                res.append(st)
        return res

    def extract_filters(self, question: str) -> List[Dict[str, Any]]:
        """Extract explicit filters such as State/Capital or numeric thresholds from query."""
        filters: List[Dict[str, Any]] = []
        norm_q = normalize_question(question)
        exp_q = expand_abbreviations(norm_q)
        states = self.extract_states(exp_q)
        if len(states) == 1:
            filters.append({"column": "State", "operator": "=", "value": states[0]})
        elif len(states) > 1:
            filters.append({"column": "State", "operator": "IN", "value": states})

        m_num = re.search(r"\b(population|literacy|males|females|households|area)\s*(>|<|>=|<=|=|==|above|over|greater\s+than|below|under|less\s+than)\s*([0-9]+(?:\.[0-9]+)?)\b", norm_q.lower())
        if m_num:
            metric_raw, op_raw, val_raw = m_num.groups()
            col = self.extract_metric(metric_raw)
            op = ">" if op_raw in ["above", "over", "greater than", ">"] else ("<" if op_raw in ["below", "under", "less than", "<"] else "=")
            filters.append({"column": col, "operator": op, "value": float(val_raw) if "." in val_raw else int(val_raw)})

        return filters

    def classify(self, question: str, session_id: Optional[str] = None) -> FastClassificationResult:
        """Deterministically classify question into FastClassificationResult."""
        norm_q = normalize_question(question)
        exp_q = expand_abbreviations(norm_q)
        q_lower = norm_q.lower()
        exp_lower = exp_q.lower()

        metric = self.extract_metric(q_lower)
        states = self.extract_states(exp_q)

        # 1. Check for Contextual Session References: "these 5 states", "these states"
        if not states and session_id:
            m_these = re.search(r"\b(?:these|those)\s+(?:(\d+)\s+)?states\b", q_lower)
            if m_these:
                from app.conversation.context import conversation_manager
                history = conversation_manager.get_history(session_id)
                session_states: List[str] = []
                for turn in reversed(history):
                    results = turn.get("results") or []
                    for r in results:
                        if isinstance(r, dict) and r.get("state") and r["state"] not in session_states:
                            session_states.append(r["state"])
                        elif isinstance(r, dict) and r.get("State") and r["State"] not in session_states:
                            session_states.append(r["State"])
                    if session_states:
                        break
                if session_states:
                    count_req = int(m_these.group(1)) if m_these.group(1) else len(session_states)
                    states = session_states[:count_req]

        # 1.6. Check Capital Lookup: "What is the capital of Telangana?", "Capitals of Telangana, AP and Karnataka"
        has_capital_word = any(w in q_lower for w in ["capital", "capitals", "rajadhani"])
        has_agg_word = any(w in q_lower for w in ["total", "sum", "average", "avg", "mean", "median", "population", "people", "villages", "count", "males", "females", "literacy", "households", "area"])
        if has_capital_word and not has_agg_word and states:
            return FastClassificationResult(
                is_fast_path=True,
                intent="CAPITAL_LOOKUP",
                scope="STATE_CAPITAL",
                metric="Capital",
                entities=states,
                original_question=question,
                normalized_question=norm_q
            )

        # 1.7. Distinct / Unique: "List distinct capitals", "distinct states", "unique capitals"
        if any(w in q_lower for w in ["distinct", "unique"]) and any(w in q_lower for w in ["capital", "capitals", "state", "states"]):
            target_col = "Capital" if "capital" in q_lower else "State"
            return FastClassificationResult(
                is_fast_path=True,
                intent="DISTINCT",
                scope="DISTINCT_VALUES",
                metric=target_col,
                entities=[],
                group_by=target_col,
                original_question=question,
                normalized_question=norm_q
            )

        # 1.8. Existence Checks: "Does Telangana exist in the dataset?", "Is there any village with population greater than 100000?", "Are there any villages in Telangana with population over 100000?"
        is_exist_word = bool(re.search(r"\b(?:exist|exists|existence|present\s+in|in\s+the\s+dataset)\b", q_lower))
        is_there_word = bool(re.search(r"^\s*(?:is\s+there|are\s+there|do\s+we\s+have)\b", q_lower))
        if is_exist_word or is_there_word:
            m_thresh = re.search(r"\b(?:greater\s+than|more\s+than|above|over)\s+([0-9]+)\b", q_lower)
            if m_thresh:
                thresh_val = int(m_thresh.group(1))
                return FastClassificationResult(
                    is_fast_path=True,
                    intent="EXISTS",
                    scope="CONDITION_EXISTENCE",
                    metric=metric or "Population",
                    entities=states,
                    n_limit=thresh_val,
                    original_question=question,
                    normalized_question=norm_q
                )
            elif states and is_exist_word:
                return FastClassificationResult(
                    is_fast_path=True,
                    intent="EXISTS",
                    scope="ENTITY_EXISTENCE",
                    metric="State",
                    entities=states,
                    original_question=question,
                    normalized_question=norm_q
                )

        # 2. Check ALL_STATES pattern
        is_all_states = any(re.search(pat, q_lower) or re.search(pat, exp_lower) for pat in self.ALL_STATES_PATTERNS)
        is_all_villages = any(re.search(pat, q_lower) or re.search(pat, exp_lower) for pat in self.ALL_VILLAGES_PATTERNS)

        # Global totals without state mention e.g. "what is the total population?", "give me the total population"
        if not states and not is_all_states and not is_all_villages:
            if re.search(r"\b(?:total|sum\s+of)\s+population\b", q_lower) or re.search(r"\bhow\s+many\s+people\b", q_lower):
                is_all_states = True

        # 3. Detect Top-N / Bottom-N states or villages
        m_top = re.search(r"\b(top|bottom)\s+(\d+)\s+(states|villages)\b", q_lower)
        if m_top:
            direction_word = m_top.group(1)
            n_val = int(m_top.group(2))
            target_entity = m_top.group(3)
            order = "DESC" if direction_word == "top" else "ASC"
            intent = "TOP_N" if direction_word == "top" else "BOTTOM_N"
            if target_entity == "villages" and states:
                return FastClassificationResult(
                    is_fast_path=True,
                    intent=intent,
                    scope="FILTERED",
                    scope_type="FILTERED",
                    return_entity="Village",
                    filter_entity="State",
                    filter_value=states[0] if len(states) == 1 else states,
                    filters=[{"column": "State", "operator": "=" if len(states) == 1 else "IN", "value": states[0] if len(states) == 1 else states}],
                    metric=metric,
                    entities=states,
                    n_limit=n_val,
                    order=order,
                    group_by=None,
                    original_question=question,
                    normalized_question=norm_q
                )
            return FastClassificationResult(
                is_fast_path=True,
                intent=intent,
                scope="ALL_STATES" if target_entity == "states" else "ALL_VILLAGES",
                scope_type="GLOBAL",
                return_entity="State" if target_entity == "states" else "Village",
                metric=metric,
                entities=[],
                n_limit=n_val,
                order=order,
                group_by="State" if target_entity == "states" else None,
                original_question=question,
                normalized_question=norm_q
            )

        # 4. Check State Rankings / Extremes: "which state has more population?", "which state has less population", "who has more people?", etc.
        m_state_ext = bool(re.search(r"\b(?:which\s+state|what\s+state|state\s+with|who\s+has|who\s+have|which\s+is\s+the\s+state|states\s+with)\b", q_lower))
        has_state_word = bool(re.search(r"\b(?:state|states)\b", q_lower))
        has_who_lead = bool(re.search(r"^\s*who\s+(?:has|have|is)\b", q_lower))
        is_village_target = bool(re.search(r"\b(?:which\s+village|what\s+village|village\s+with)\b", q_lower))

        ext_max_words = ["highest", "maximum", "max", "most", "largest", "more", "higher", "greater", "greatest", "top", "best"]
        ext_min_words = ["lowest", "minimum", "min", "least", "smallest", "less", "fewer", "fewest", "lower", "bottom", "worst"]

        # Check for Village Extremes first if village is targeted
        if is_village_target:
            if session_id:
                try:
                    from app.conversation.context import conversation_manager
                    last_comp = conversation_manager.get_last_comparison(session_id)
                    if last_comp and last_comp.get("left_entity") and last_comp.get("right_entity"):
                        if not is_all_villages and not is_all_states and not states:
                            return FastClassificationResult(
                                is_fast_path=True,
                                intent="COMPARE",
                                scope="CROSS_STATE_VILLAGE" if (last_comp.get("left_parent") != last_comp.get("right_parent")) else "SAME_STATE_VILLAGE",
                                metric=metric or last_comp.get("attribute") or "Population",
                                entities=[last_comp.get("left_entity"), last_comp.get("right_entity")],
                                original_question=question,
                                normalized_question=norm_q
                            )
                except Exception:
                    pass

            has_v_max = any(re.search(r"\b" + re.escape(w) + r"\b", q_lower) for w in ext_max_words)
            has_v_min = any(re.search(r"\b" + re.escape(w) + r"\b", q_lower) for w in ext_min_words)
            if has_v_max or has_v_min:
                is_max = has_v_max and not has_v_min
                order = "DESC" if is_max else "ASC"
                intent = "MAX" if is_max else "MIN"

                # Filtered village query when one or more states are specified
                if states:
                    return FastClassificationResult(
                        is_fast_path=True,
                        intent=intent,
                        scope="FILTERED",
                        scope_type="FILTERED",
                        return_entity="Village",
                        filter_entity="State",
                        filter_value=states[0] if len(states) == 1 else states,
                        filters=[{"column": "State", "operator": "=" if len(states) == 1 else "IN", "value": states[0] if len(states) == 1 else states}],
                        metric=metric or "Population",
                        entities=states,
                        n_limit=1,
                        order=order,
                        group_by=None,
                        original_question=question,
                        normalized_question=norm_q
                    )

                return FastClassificationResult(
                    is_fast_path=True,
                    intent=intent,
                    scope="ALL_VILLAGES",
                    scope_type="GLOBAL",
                    return_entity="Village",
                    filter_entity=None,
                    filter_value=None,
                    filters=[],
                    metric=metric or "Population",
                    entities=[],
                    n_limit=1,
                    order=order,
                    group_by=None,
                    original_question=question,
                    normalized_question=norm_q
                )

        if (m_state_ext or has_who_lead or (has_state_word and any(w in q_lower for w in ["most", "least", "highest", "lowest", "more", "less", "top", "bottom"]))):
            has_more_populated = bool(re.search(r"\b(?:more|most|highest)\s+populated\b", q_lower))
            has_less_populated = bool(re.search(r"\b(?:less|least|lowest)\s+populated\b", q_lower))
            has_max = has_more_populated or any(re.search(r"\b" + re.escape(w) + r"\b", q_lower) for w in ext_max_words)
            has_min = has_less_populated or any(re.search(r"\b" + re.escape(w) + r"\b", q_lower) for w in ext_min_words)

            if (has_max or has_min) and len(states) < 2:
                is_max = has_max and not has_min
                metric_resolved = metric
                if not metric_resolved:
                    if any(w in q_lower for w in ["village", "villages"]):
                        metric_resolved = "VILLAGE_COUNT"
                    elif any(w in q_lower for w in ["literacy", "literate", "education"]):
                        metric_resolved = "Literacy_Rate_Percent"
                    elif any(w in q_lower for w in ["male", "males", "men"]):
                        metric_resolved = "No_of_Males"
                    elif any(w in q_lower for w in ["female", "females", "women"]):
                        metric_resolved = "No_of_Females"
                    elif any(w in q_lower for w in ["household", "households", "houses"]):
                        metric_resolved = "Households"
                    elif any(w in q_lower for w in ["area", "size"]):
                        metric_resolved = "Area_Sq_Km"
                    else:
                        metric_resolved = "Population"

                return FastClassificationResult(
                    is_fast_path=True,
                    intent="MAX" if is_max else "MIN",
                    scope="STATE_RANKING",
                    metric=metric_resolved,
                    entities=[],
                    n_limit=1,
                    order="DESC" if is_max else "ASC",
                    group_by="State",
                    original_question=question,
                    normalized_question=norm_q
                )

        # 5. Check Average / Mean per state: "average population per state", "average population of all states"
        if any(w in q_lower for w in ["average", "avg", "mean"]) and (is_all_states or "per state" in q_lower):
            return FastClassificationResult(
                is_fast_path=True,
                intent="AVERAGE",
                scope="ALL_STATES",
                metric=metric,
                entities=[],
                group_by="State",
                original_question=question,
                normalized_question=norm_q
            )

        # 6. Check Total / Sum for ALL states
        if is_all_states:
            op = "SUM"
            if any(w in q_lower for w in ["average", "avg", "mean"]):
                op = "AVERAGE"
            elif any(w in q_lower for w in ["median"]):
                op = "MEDIAN"
            elif any(w in q_lower for w in ["range"]):
                op = "RANGE"
            elif any(w in q_lower for w in ["variance"]):
                op = "VARIANCE"
            elif any(w in q_lower for w in ["standard deviation", "std"]):
                op = "STANDARD_DEVIATION"
            elif any(w in q_lower for w in ["count", "how many states"]):
                op = "COUNT"

            # Check if per-state breakdown is requested: "total population of every state", "population of each state", "Rank all states"
            if any(w in q_lower for w in ["every state", "each state", "all states breakdown", "breakdown by state", "rank", "ranking", "sort", "order"]):
                return FastClassificationResult(
                    is_fast_path=True,
                    intent="GROUP_AGGREGATE",
                    scope="ALL_STATES_BREAKDOWN",
                    metric=metric,
                    entities=[],
                    group_by="State",
                    order="DESC",
                    original_question=question,
                    normalized_question=norm_q
                )

            return FastClassificationResult(
                is_fast_path=True,
                intent=op,
                scope="ALL_STATES",
                metric=metric,
                entities=[],
                original_question=question,
                normalized_question=norm_q
            )

        # 7. Check Total / Aggregations across ALL villages
        if is_all_villages:
            op = "SUM"
            if any(w in q_lower for w in ["average", "avg", "mean"]):
                op = "AVERAGE"
            elif any(w in q_lower for w in ["median"]):
                op = "MEDIAN"
            elif any(w in q_lower for w in ["count"]):
                op = "COUNT"
            elif any(w in q_lower for w in ["highest", "max", "maximum"]):
                op = "MAX"
            elif any(w in q_lower for w in ["lowest", "min", "minimum"]):
                op = "MIN"

            return FastClassificationResult(
                is_fast_path=True,
                intent=op,
                scope="ALL_VILLAGES",
                metric=metric,
                entities=[],
                original_question=question,
                normalized_question=norm_q
            )

        # 8. Check Selected States: 2 or more states with SUM/TOTAL/COUNT
        if len(states) >= 2:
            has_comp_word = (
                any(w in q_lower for w in ["compare", "vs", "versus", "difference", "differ", "between"])
                or bool(re.search(r"\b(?:more|fewer|less|higher|lower|greater)\b.*\bthan\b", q_lower))
                or bool(re.search(r"\bhow\s+many\s+(?:more|fewer|less)\b", q_lower))
                or bool(re.search(r"\bhow\s+much\s+(?:more|fewer|less|higher|lower)\b", q_lower))
                or bool(re.search(r"\bwhich\s+state\s+has\s+(?:more|fewer|less|higher|lower)\b", q_lower))
            )
            if has_comp_word:
                return FastClassificationResult(
                    is_fast_path=True,
                    intent="COMPARE",
                    scope="SELECTED_STATES",
                    metric=metric,
                    entities=states,
                    original_question=question,
                    normalized_question=norm_q
                )
            else:
                op = "SUM"
                if any(w in q_lower for w in ["average", "avg", "mean"]):
                    op = "AVERAGE"
                return FastClassificationResult(
                    is_fast_path=True,
                    intent=op,
                    scope="SELECTED_STATES",
                    metric=metric,
                    entities=states,
                    original_question=question,
                    normalized_question=norm_q
                )

        # 9. Single State Aggregation: "Total population of Telangana", "sum of population in AP"
        if len(states) == 1:
            state_target = states[0]
            # Check if query is actually a comparison inquiry (e.g. "Compare Telangana with Hyderabad", "Compare TG-001 and TG-002")
            has_comp_word = (
                any(w in q_lower for w in ["compare", "vs", "versus", "difference", "differ", "between"])
                or bool(re.search(r"\b(?:more|fewer|less|higher|lower|greater)\b.*\bthan\b", q_lower))
            )
            if has_comp_word:
                return FastClassificationResult(
                    is_fast_path=False,
                    intent="COMPARE",
                    scope="GENERIC",
                    metric=metric,
                    entities=states,
                    original_question=question,
                    normalized_question=norm_q
                )

            # Check if query asks for a single village in that state
            from app.dataset.registry import parent_child_registry
            if parent_child_registry.is_village_level_query(question):
                m_vid = re.search(r"\b([a-zA-Z]{2}-\d{3})\b", question)
                m_vname = re.search(r"\b([a-zA-Z]+_village_\d+)\b", q_lower)
                m_vnum = re.search(r"\bvillage\s*([0-9]+)\b", q_lower)
                if m_vid or m_vname or m_vnum:
                    return FastClassificationResult(
                        is_fast_path=True,
                        intent="LOOKUP",
                        scope="SINGLE_VILLAGE",
                        metric=metric,
                        entities=[state_target],
                        original_question=question,
                        normalized_question=norm_q
                    )

            # Single state sum or aggregate
            op = "SUM"
            if any(w in q_lower for w in ["average", "avg", "mean"]) or metric == "Literacy_Rate_Percent":
                op = "AVERAGE"
            elif any(w in q_lower for w in ["median"]):
                op = "MEDIAN"
            elif any(w in q_lower for w in ["count", "how many villages", "number of villages", "village count", "count of villages"]):
                op = "COUNT"
                if any(w in q_lower for w in ["village", "villages"]):
                    metric = "VILLAGE_COUNT"
            elif any(w in q_lower for w in ["highest", "max", "maximum", "most", "largest", "more", "higher"]):
                return FastClassificationResult(
                    is_fast_path=True,
                    intent="MAX",
                    scope="FILTERED",
                    scope_type="FILTERED",
                    return_entity="Village" if ("village" in q_lower or is_village_target) else "Record",
                    filter_entity="State",
                    filter_value=state_target,
                    filters=[{"column": "State", "operator": "=", "value": state_target}],
                    metric=metric or "Population",
                    entities=[state_target],
                    n_limit=1,
                    order="DESC",
                    original_question=question,
                    normalized_question=norm_q
                )
            elif any(w in q_lower for w in ["lowest", "min", "minimum", "least", "smallest", "less", "fewer", "lower"]):
                return FastClassificationResult(
                    is_fast_path=True,
                    intent="MIN",
                    scope="FILTERED",
                    scope_type="FILTERED",
                    return_entity="Village" if ("village" in q_lower or is_village_target) else "Record",
                    filter_entity="State",
                    filter_value=state_target,
                    filters=[{"column": "State", "operator": "=", "value": state_target}],
                    metric=metric or "Population",
                    entities=[state_target],
                    n_limit=1,
                    order="ASC",
                    original_question=question,
                    normalized_question=norm_q
                )

            return FastClassificationResult(
                is_fast_path=True,
                intent=op,
                scope="SINGLE_STATE",
                metric=metric,
                entities=[state_target],
                original_question=question,
                normalized_question=norm_q
            )

        # Default fallback
        return FastClassificationResult(
            is_fast_path=False,
            intent="LOOKUP",
            scope="GENERIC",
            metric=metric,
            entities=[],
            original_question=question,
            normalized_question=norm_q
        )


fast_query_classifier = FastQueryClassifier()

