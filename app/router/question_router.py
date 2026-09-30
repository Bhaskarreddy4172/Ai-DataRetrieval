"""Universal Question Router: classifies queries into DATASET, GENERAL_KNOWLEDGE, HYBRID, AMBIGUOUS, or UNSUPPORTED."""

import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from app.knowledge.general_knowledge import general_knowledge_engine, INDIAN_CITIES, INDIAN_STATES, WORLD_CAPITALS, TECHNICAL_DEFINITIONS
from app.boolean.assertion_builder import boolean_assertion_builder
from app.utils.fuzzy_match import expand_abbreviations, extract_best_match_from_text
from app.utils.normalization import normalize_question
from app.utils.logger import logger


class RoutingDecision(BaseModel):
    """Structured decision output from the Question Router."""
    question_type: str = Field(..., description="DATASET | GENERAL_KNOWLEDGE | HYBRID | AMBIGUOUS | UNSUPPORTED")
    normalized_question: str
    confidence: float = 1.0
    dataset_needed: bool = False
    general_knowledge_needed: bool = False
    dataset_subquery: Optional[str] = None
    general_subquery: Optional[str] = None
    target_entity: Optional[str] = None
    explanation: str = ""
    is_boolean: bool = False
    boolean_assertion: Optional[Dict[str, Any]] = None
    hierarchy_scope: Optional[str] = None
    child_dataset_target: Optional[str] = None


class QuestionRouter:
    """Universal router that inspects question semantics, dataset schema, and general knowledge."""

    def __init__(self):
        self.gk = general_knowledge_engine

    def route(
        self,
        question: str,
        available_columns: List[str],
        schema_info: Optional[Dict[str, Any]] = None,
        dataset_values: Optional[List[str]] = None,
        df: Optional[Any] = None
    ) -> RoutingDecision:
        """Classify question into DATASET, GENERAL_KNOWLEDGE, HYBRID, AMBIGUOUS, or UNSUPPORTED."""
        if not question or not question.strip():
            return RoutingDecision(
                question_type="UNSUPPORTED",
                normalized_question="",
                confidence=1.0,
                explanation="Question is empty."
            )

        # 1. Normalize question (contractions, typos, abbreviations, slang)
        norm_q = normalize_question(question)
        expanded_q = expand_abbreviations(norm_q)
        q_lower = expanded_q.lower()
        cols_lower = [c.lower() for c in available_columns]

        # 1.5. Check for BOOLEAN_CHECK assertion
        if boolean_assertion_builder.is_boolean_question(question) or boolean_assertion_builder.is_boolean_question(norm_q):
            assertion = boolean_assertion_builder.build_assertion(question, available_columns, schema_info, df)
            if not assertion:
                assertion = boolean_assertion_builder.build_assertion(norm_q, available_columns, schema_info, df)
            if assertion:
                return RoutingDecision(
                    question_type=assertion.source_target,
                    normalized_question=expanded_q,
                    confidence=1.0,
                    dataset_needed=(assertion.source_target in ["DATASET", "HYBRID"]),
                    general_knowledge_needed=(assertion.source_target in ["GENERAL_KNOWLEDGE", "HYBRID"]),
                    target_entity=assertion.subject,
                    explanation=assertion.explanation or "Boolean assertion check.",
                    is_boolean=True,
                    boolean_assertion=assertion.model_dump()
                )

        # 1.8. Check for Parent-Child Hierarchical dataset queries
        from app.dataset.registry import parent_child_registry
        from app.dataset.hierarchical_engine import hierarchical_query_engine
        if parent_child_registry.is_parent_child_active() and (hierarchical_query_engine.can_handle(expanded_q) or hierarchical_query_engine.can_handle(question)):
            target_q = expanded_q if hierarchical_query_engine.can_handle(expanded_q) else question
            h_scope, h_ent, h_path = hierarchical_query_engine.classify_scope(target_q)
            h_target = h_path.name if h_path else (h_ent or "all_child_datasets")
            return RoutingDecision(
                question_type="DATASET",
                normalized_question=expanded_q,
                confidence=0.98,
                dataset_needed=True,
                general_knowledge_needed=False,
                dataset_subquery=expanded_q,
                explanation=f"Hierarchical query routing to {h_scope} ({h_target}).",
                hierarchy_scope=h_scope,
                child_dataset_target=h_target
            )

        # 1.9. Check for Tabular Comparison queries on active dataset
        from app.query.comparison_engine import universal_comparison_engine
        if universal_comparison_engine.is_comparison_query(expanded_q) or universal_comparison_engine.is_comparison_query(question):
            comp_plan = universal_comparison_engine.parse_and_plan(expanded_q, df=df) or universal_comparison_engine.parse_and_plan(question, df=df)
            if comp_plan and comp_plan.scope == "GENERIC_DATASET":
                return RoutingDecision(
                    question_type="DATASET",
                    normalized_question=expanded_q,
                    confidence=0.99,
                    dataset_needed=True,
                    general_knowledge_needed=False,
                    dataset_subquery=expanded_q,
                    explanation=f"Generic comparison query between {comp_plan.entities[0].name} and {comp_plan.entities[1].name}."
                )

        # 2. Check for HYBRID question (combining dataset query + external general knowledge)
        hybrid_decision = self._check_hybrid(q_lower, expanded_q, available_columns, cols_lower)
        if hybrid_decision:
            return hybrid_decision

        # 3. Check for GENERAL KNOWLEDGE question
        gk_decision = self._check_general_knowledge(q_lower, expanded_q, available_columns, cols_lower, df=df)
        if gk_decision:
            return gk_decision

        # 4. Check for DATASET question
        dataset_decision = self._check_dataset(q_lower, expanded_q, available_columns, cols_lower, schema_info)
        if dataset_decision:
            return dataset_decision
        # 4.5. Check registered multi-dataset catalog
        from app.dataset.registry import parent_child_registry
        resolved_ds = parent_child_registry.resolve_dataset_for_query(expanded_q)
        if resolved_ds:
            ds_name, _, _ = resolved_ds
            return RoutingDecision(
                question_type="DATASET",
                normalized_question=expanded_q,
                confidence=0.98,
                dataset_needed=True,
                general_knowledge_needed=False,
                dataset_subquery=expanded_q,
                explanation=f"Query matched registered dataset '{ds_name}'."
            )

        # 5. Check for AMBIGUITY
        # If question is extremely short or vague without columns/entities
        words = [w for w in re.findall(r"\b\w+\b", q_lower) if len(w) > 2]
        if len(words) <= 1:
            return RoutingDecision(
                question_type="AMBIGUOUS",
                normalized_question=expanded_q,
                confidence=0.5,
                explanation="Your question is too short or ambiguous. Could you please provide more details?"
            )

        # 6. Default to DATASET query engine with safe fallback
        return RoutingDecision(
            question_type="DATASET",
            normalized_question=expanded_q,
            confidence=0.75,
            dataset_needed=True,
            general_knowledge_needed=False,
            dataset_subquery=expanded_q,
            explanation="Interpreted as dataset query."
        )

    def _check_hybrid(
        self,
        q_lower: str,
        orig_q: str,
        available_columns: List[str],
        cols_lower: List[str]
    ) -> Optional[RoutingDecision]:
        """Detect hybrid queries that require BOTH dataset data and general knowledge."""
        # Pattern 1: [Dataset question about people/records in a city] AND [which state is that city in / what state is X in]
        # e.g. "Who works in Bangalore and which state is Bangalore in?"
        # e.g. "Which employees are in Hyderabad and what state is Hyderabad in?"
        # e.g. "Who works in Bangalore and which state is that city in?"
        hybrid_connectors = [
            r"and\s+(?:which|what)\s+state\s+(?:is|does)",
            r"and\s+(?:what|which)\s+(?:is\s+the\s+)?capital",
            r"and\s+where\s+(?:is\s+that|is\s+it|does\s+it)",
            r"and\s+which\s+state\s+is\s+that\s+city\s+in",
            r"and\s+what\s+state\s+is\s+that\s+city\s+in",
        ]

        has_hybrid_connector = any(re.search(pat, q_lower) for pat in hybrid_connectors)
        if has_hybrid_connector:
            parts = re.split(r"\s+and\s+", orig_q, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                sub_data = parts[0].strip()
                sub_gen = parts[1].strip()

                # Find any city in sub_data to resolve pronouns like "that city", "it" in sub_gen
                matched_city = None
                for alias, canonical in self.gk._city_alias_map.items():
                    if re.search(r"\b" + re.escape(alias) + r"\b", sub_data.lower()):
                        matched_city = canonical
                        break

                if matched_city:
                    # Replace "that city", "that place", "the city", "it" in sub_gen with matched_city
                    sub_gen = re.sub(r"\b(?:that city|that place|the city|it)\b", matched_city, sub_gen, flags=re.IGNORECASE)
                    if not any(alias in sub_gen.lower() for alias in self.gk._city_alias_map):
                        sub_gen = f"{sub_gen} {matched_city}"

                return RoutingDecision(
                    question_type="HYBRID",
                    normalized_question=orig_q,
                    confidence=0.95,
                    dataset_needed=True,
                    general_knowledge_needed=True,
                    dataset_subquery=sub_data,
                    general_subquery=sub_gen,
                    target_entity=matched_city,
                    explanation="Combines dataset retrieval with general geographical knowledge."
                )

        return None

    def _check_general_knowledge(
        self,
        q_lower: str,
        orig_q: str,
        available_columns: List[str],
        cols_lower: List[str],
        df: Optional[Any] = None
    ) -> Optional[RoutingDecision]:
        """Detect questions that can and should be answered purely via general knowledge."""
        if df is None:
            try:
                from app.dataset.loader import dataset_loader
                df = dataset_loader.dataframe
            except Exception:
                pass
        # 1. Technical / Conceptual definitions
        for concept in TECHNICAL_DEFINITIONS.keys():
            # Check if query asks for definition
            pat = r"\b(?:what is|what does|meaning of|define|definition of|difference between)\s+(?:the\s+)?" + re.escape(concept) + r"\b"
            if re.search(pat, q_lower) or q_lower.strip(" ?") == concept:
                # Ensure concept is not an exact column name in dataset
                if concept not in cols_lower:
                    return RoutingDecision(
                        question_type="GENERAL_KNOWLEDGE",
                        normalized_question=orig_q,
                        confidence=1.0,
                        dataset_needed=False,
                        general_knowledge_needed=True,
                        general_subquery=orig_q,
                        target_entity=concept,
                        explanation=f"General definition query for '{concept}'."
                    )

        # 2. National / Global facts
        fact_patterns = [
            r"capital of india",
            r"president of india",
            r"prime minister of india",
            r"how many states.*in india",
            r"currency of india",
        ]
        if any(re.search(p, q_lower) for p in fact_patterns):
            return RoutingDecision(
                question_type="GENERAL_KNOWLEDGE",
                normalized_question=orig_q,
                confidence=1.0,
                dataset_needed=False,
                general_knowledge_needed=True,
                general_subquery=orig_q,
                explanation="General national factual query."
            )

        # 3. Capital of [State / Country] queries (direct, possessive, reverse)
        # If dataset HAS a column named 'State' and 'Capital' (like indian_states_capitals.csv),
        # then "capital of Telangana" or "West Bengal's capital" can be answered from dataset!
        # BUT if dataset does NOT have 'capital' or 'state' columns, it is GENERAL_KNOWLEDGE.
        has_capital_col = any("capital" in c for c in cols_lower)
        if not has_capital_col:
            cap_patterns = [
                r"capital of\s+([a-zA-Z\s]+)",
                r"([a-zA-Z\s]+?)'s\s+capital",
                r"which\s+city\s+is\s+([a-zA-Z\s]+?)'s\s+capital",
                r"([a-zA-Z\s]+?)\s+has\s+which\s+capital",
                r"which\s+state\s+has\s+([a-zA-Z\s]+?)\s+as(?:\s+its)?\s+capital",
                r"([a-zA-Z\s]+?)\s+is(?:\s+the)?\s+capital\s+of\s+which\s+state",
                r"([a-zA-Z\s]+?)\s+capital"
            ]
            for pat in cap_patterns:
                cap_match = re.search(pat, q_lower)
                if cap_match:
                    cand = cap_match.group(1).rstrip("?").strip()
                    cand = re.sub(r"^(?:what\s+is\s+|tell\s+me\s+|show\s+me\s+|what's\s+|whats\s+)", "", cand).strip()
                    if self.gk.find_state_match(cand) or any(c in cand for c in WORLD_CAPITALS) or self.gk.find_city_match(cand):
                        return RoutingDecision(
                            question_type="GENERAL_KNOWLEDGE",
                            normalized_question=orig_q,
                            confidence=1.0,
                            dataset_needed=False,
                            general_knowledge_needed=True,
                            general_subquery=orig_q,
                            target_entity=cand,
                            explanation=f"General geography query for capital related to '{cand}'."
                        )

        # 4. State of City queries (e.g. "Which state does banglore belongs to?", "banglore which state bro?")
        # If dataset does NOT have a 'state' column, this is 100% a General Knowledge query!
        # Even if the city name is a value in the 'City' column of the dataset.
        has_state_col = any("state" in c for c in cols_lower)

        city_state_patterns = [
            r"which state (?:does\s+)?([a-zA-Z\s]+?)(?:\s+belong|\s+belongs|\s+is|\s+comes|\s+located|\?|$)",
            r"([a-zA-Z\s]+?)\s+(?:is\s+)?(?:under|in|comes\s+under|belongs\s+to|comes\s+in)\s+which\s+state",
            r"([a-zA-Z\s]+?)\s+(?:which state|is in which state|comes under which state|belongs to which state)",
            r"where is\s+([a-zA-Z\s]+?)(?:\?|$)",
            r"where does\s+([a-zA-Z\s]+?)(?:\s+belong|\s+belongs|\s+come|\s+comes)(?:\s+to)?(?:\?|$)",
            r"what state is\s+([a-zA-Z\s]+?)(?:\s+in|\?|$)",
            r"which state is\s+([a-zA-Z\s]+?)(?:\s+in|\?|$)",
            r"([a-zA-Z\s]+?)'s\s+state(?:\?|$)",
            r"([a-zA-Z\s]+?)\s+state(?:\?|$)",
            r"([a-zA-Z\s]+?)\s+which state\b",
        ]

        for p in city_state_patterns:
            m = re.search(p, q_lower)
            if m:
                cand = m.group(1).strip()
                cand = re.sub(r"^(?:the\s+city\s+of|the\s+|does\s+|in\s+|where\s+does\s+)", "", cand).strip()
                cand = re.sub(r"(?:\s+bro|\s+city|\s+area|\s+town|\s+belong|\s+belongs|\s+to)$", "", cand).strip()

                # Check if candidate matches a known Indian city
                city_info = self.gk.get_city_info(cand)
                if city_info:
                    # Check if candidate city is in dataset
                    city_in_dataset = False
                    if df is not None and not df.empty:
                        for col in available_columns:
                            if col in df.columns:
                                col_vals = [str(v).lower() for v in df[col].dropna()]
                                if cand.lower() in col_vals or city_info["canonical_name"].lower() in col_vals:
                                    city_in_dataset = True
                                    break

                    # If dataset does not have 'State' column or city is not in dataset, route to GENERAL_KNOWLEDGE
                    if not has_state_col or not city_in_dataset:
                        return RoutingDecision(
                            question_type="GENERAL_KNOWLEDGE",
                            normalized_question=orig_q,
                            confidence=1.0,
                            dataset_needed=False,
                            general_knowledge_needed=True,
                            general_subquery=orig_q,
                            target_entity=city_info["canonical_name"],
                            explanation=f"General geography query for state of city '{city_info['canonical_name']}'."
                        )

        # Also search for any city mentioned in the whole question if 'state' or 'where' is in the query
        if any(w in q_lower for w in ["state", "where", "location", "region", "which part", "belong"]):
            for alias, canonical in self.gk._city_alias_map.items():
                if re.search(r"\b" + re.escape(alias) + r"\b", q_lower):
                    city_info = INDIAN_CITIES[canonical]
                    city_in_ds = False
                    if df is not None and not df.empty:
                        for col in available_columns:
                            if col in df.columns:
                                col_vals = [str(v).lower() for v in df[col].dropna()]
                                if alias.lower() in col_vals or canonical.lower() in col_vals:
                                    city_in_ds = True
                                    break
                    if not has_state_col or not city_in_ds:
                        return RoutingDecision(
                            question_type="GENERAL_KNOWLEDGE",
                            normalized_question=orig_q,
                            confidence=0.95,
                            dataset_needed=False,
                            general_knowledge_needed=True,
                            general_subquery=orig_q,
                            target_entity=canonical,
                            explanation=f"General geography query for '{canonical}'."
                        )

        # 5. Catch-all for "where is [City]" if not asking about people/records
        where_m = re.search(r"where is\s+([a-zA-Z\s]+?)(?:\?|$)", q_lower)
        if where_m:
            cand = where_m.group(1).strip()
            city_info = self.gk.get_city_info(cand)
            if city_info and not any(term in q_lower for term in ["employee", "customer", "person", "worker", "client", "service", "salary"]):
                return RoutingDecision(
                    question_type="GENERAL_KNOWLEDGE",
                    normalized_question=orig_q,
                    confidence=0.95,
                    dataset_needed=False,
                    general_knowledge_needed=True,
                    general_subquery=orig_q,
                    target_entity=city_info["canonical_name"],
                    explanation=f"General geography location query for '{city_info['canonical_name']}'."
                )

        return None

    def _check_dataset(
        self,
        q_lower: str,
        orig_q: str,
        available_columns: List[str],
        cols_lower: List[str],
        schema_info: Optional[Dict[str, Any]] = None
    ) -> Optional[RoutingDecision]:
        """Detect questions that directly target dataset entities, rows, or columns."""
        # 1. Check for standard dataset intents and entity queries
        dataset_intent_keywords = [
            "who earns", "who has highest", "who is getting", "who joined", "how many",
            "highest salary", "lowest salary", "average salary", "total salary",
            "best performance", "top earner", "service amount", "stock quantity",
            "who works in", "who is in", "employees in", "customers in",
            "compare", "same city", "same department", "duplicate", "distinct",
            "second highest", "third highest", "between"
        ]

        if any(kw in q_lower for kw in dataset_intent_keywords):
            return RoutingDecision(
                question_type="DATASET",
                normalized_question=orig_q,
                confidence=0.98,
                dataset_needed=True,
                general_knowledge_needed=False,
                dataset_subquery=orig_q,
                explanation="Directly references dataset entities, metrics, or aggregations."
            )

        # 2. Check if any column or column synonym appears in question
        for col in available_columns:
            if col.lower() in q_lower:
                return RoutingDecision(
                    question_type="DATASET",
                    normalized_question=orig_q,
                    confidence=0.95,
                    dataset_needed=True,
                    general_knowledge_needed=False,
                    dataset_subquery=orig_q,
                    explanation=f"Matches dataset column '{col}'."
                )

        return None


question_router = QuestionRouter()
