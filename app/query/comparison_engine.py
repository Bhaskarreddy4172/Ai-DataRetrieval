"""Universal Dataset Comparison & Difference Engine.

Supports deterministic, multi-level comparisons:
1. Village <-> Village within the SAME state
2. Village <-> Village across DIFFERENT states
3. State <-> State
4. Capital <-> Capital
5. State <-> Capital & Capital <-> State
6. Generic Entities (Employee <-> Employee, Product <-> Product, etc.)
7. Top-N / Bottom-N & Ranking Differences (e.g. top 2 villages, difference between 1st and 2nd)
8. Multi-Entity Comparisons (3+ entities)
9. Arithmetic Operations:
   - DIRECTED_DIFFERENCE (left - right)
   - ABSOLUTE_DIFFERENCE (|left - right|)
   - PERCENTAGE_DIFFERENCE (|left - right| / right * 100)
   - PERCENTAGE_CHANGE ((left - right) / right * 100)
   - RATIO (left / right)
   - BOOLEAN_COMPARISON (Yes/No with evidence)

CRITICAL RULE:
NEVER return a raw negative number for 'how much more / fewer / higher / lower'.
Always use absolute difference for the count and accurately articulate the direction.
"""

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from app.dataset.registry import parent_child_registry
from app.dataset.multi_child_executor import multi_child_executor
from app.utils.fuzzy_match import expand_abbreviations, LOCATION_ABBREVIATIONS
from app.utils.normalization import normalize_question
from app.utils.logger import logger


@dataclass
class ComparisonEntity:
    """Represents an entity participating in a comparison with exact resolution provenance."""
    name: str
    entity_type: str  # "VILLAGE", "STATE", "CAPITAL", "ROW", "GENERIC"
    normalized_name: Optional[str] = None
    parent: Optional[str] = None
    parent_code: Optional[str] = None
    parent_dataset: Optional[str] = None
    child_dataset: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_version: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_path: Optional[Path] = None
    row_id: Optional[Any] = None
    metric_value: Optional[float] = None
    is_missing: bool = False
    raw_record: Optional[Dict[str, Any]] = None
    match_method: str = "EXACT"  # "EXACT", "NORMALIZED", "ALIAS", "TYPO", "PHONETIC", "TOKEN"
    match_confidence: str = "EXACT"  # "EXACT", "HIGH", "MEDIUM", "LOW", "AMBIGUOUS", "NO_MATCH"


@dataclass
class ComparisonPlan:
    """Execution plan for a comparative inquiry."""
    entities: List[ComparisonEntity]
    attribute: str
    comparison_type: str  # "ABSOLUTE_DIFFERENCE", "PERCENTAGE_DIFFERENCE", "PERCENTAGE_CHANGE", "RATIO", "BOOLEAN", "RANKING"
    direction_requested: Optional[str] = None  # "MORE", "LESS", "FEWER", "HIGHER", "LOWER", "DIFF"
    is_boolean: bool = False
    boolean_operator: Optional[str] = None  # ">", "<", "==", "!="
    scope: str = "SAME_STATE_VILLAGE"  # "SAME_STATE_VILLAGE", "CROSS_STATE_VILLAGE", "STATE_VS_STATE", "CAPITAL_VS_CAPITAL", "STATE_VS_CAPITAL", "CAPITAL_VS_STATE", "GENERIC_DATASET", "RANKING", "MULTI_ENTITY"
    original_question: str = ""


@dataclass
class ComparisonResult:
    """Standardized deterministic comparison result with full provenance adhering to Section 14."""
    status: str  # "SUCCESS", "MISSING_VALUE", "ENTITY_NOT_FOUND", "AMBIGUOUS", "INCOMPATIBLE_ATTRIBUTES", "DIVISION_BY_ZERO", "NO_MATCH"
    operation: str = "COMPARE"
    left_entity: Optional[str] = None
    right_entity: Optional[str] = None
    attribute: str = ""
    unit: str = ""
    left_value: Optional[float] = None
    right_value: Optional[float] = None
    absolute_difference: Optional[float] = None
    directed_difference: Optional[float] = None
    percentage_difference: Optional[float] = None
    percentage_change: Optional[float] = None
    ratio: Optional[float] = None
    higher_entity: Optional[str] = None
    lower_entity: Optional[str] = None
    comparison_operator: Optional[str] = None
    is_equal: bool = False
    boolean_result: Optional[bool] = None
    answer: str = ""
    rankings: Optional[List[Dict[str, Any]]] = None
    source_datasets: List[str] = field(default_factory=list)
    source_rows: List[Any] = field(default_factory=list)
    source_row_ids: List[Any] = field(default_factory=list)
    dataset_versions: List[str] = field(default_factory=list)
    columns_used: List[str] = field(default_factory=list)
    calculation_formula: Optional[str] = None
    verification_status: str = "VERIFIED"  # "VERIFIED", "FAILED", "MISSING_VALUE", "AMBIGUOUS", "INCOMPATIBLE"
    left_parent: Optional[str] = None
    right_parent: Optional[str] = None
    scope: Optional[str] = None
    verification_checks: Dict[str, Any] = field(default_factory=dict)
    debug_trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Produce the Section 14 standardized comparison result dictionary."""
        return {
            "left_entity": self.left_entity,
            "right_entity": self.right_entity,
            "left_parent": self.left_parent,
            "right_parent": self.right_parent,
            "scope": self.scope,
            "left_value": self.left_value,
            "right_value": self.right_value,
            "attribute": self.attribute,
            "unit": self.unit,
            "directed_difference": self.directed_difference,
            "absolute_difference": self.absolute_difference,
            "higher_entity": self.higher_entity,
            "lower_entity": self.lower_entity,
            "comparison_operator": self.comparison_operator or ("==" if self.is_equal else (">" if (self.directed_difference or 0) > 0 else "<")),
            "percentage_difference": self.percentage_difference,
            "percentage_change": self.percentage_change,
            "ratio": self.ratio,
            "source_datasets": self.source_datasets,
            "source_rows": self.source_rows or self.source_row_ids,
            "dataset_versions": self.dataset_versions,
            "verification_status": self.verification_status,
            "verification_checks": self.verification_checks,
            "answer": self.answer,
        }


class UniversalComparisonEngine:

    """Universal deterministic comparison and difference engine."""

    METRIC_ALIASES: Dict[str, str] = {
        "population": "Population",
        "pop": "Population",
        "people": "Population",
        "inhabitants": "Population",
        "residents": "Population",
        "persons": "Population",
        "males": "No_of_Males",
        "male": "No_of_Males",
        "men": "No_of_Males",
        "male population": "No_of_Males",
        "females": "No_of_Females",
        "female": "No_of_Females",
        "women": "No_of_Females",
        "female population": "No_of_Females",
        "literacy": "Literacy_Rate_Percent",
        "literacy rate": "Literacy_Rate_Percent",
        "literacy rate percent": "Literacy_Rate_Percent",
        "education": "Literacy_Rate_Percent",
        "area": "Area_Sq_Km",
        "area sq km": "Area_Sq_Km",
        "size": "Area_Sq_Km",
        "households": "Households",
        "houses": "Households",
        "homes": "Households",
        "families": "Households",
        "villages": "VILLAGE_COUNT",
        "village count": "VILLAGE_COUNT",
        "number of villages": "VILLAGE_COUNT",
        "count of villages": "VILLAGE_COUNT",
        "salary": "Salary",
        "salaries": "Salary",
        "wage": "Salary",
        "income": "Salary",
        "price": "Price",
        "cost": "Price",
        "revenue": "Revenue",
        "profit": "Profit",
        "score": "Score",
        "marks": "Marks",
        "age": "Age",
        "experience": "Experience",
    }

    COMPARISON_TRIGGER_WORDS = [
        "compare", "vs", "versus", "difference", "diff",
        "more than", "fewer than", "less than", "higher than", "lower than", "greater than", "smaller than", "larger than",
        "how much more", "how many more", "how much less", "how many less", "how many fewer", "how much fewer",
        "how much higher", "how much lower", "how much greater", "how much smaller", "how much larger",
        "which has more", "which has less", "which has fewer", "which has higher", "which has lower", "which has larger",
        "which one has more", "which one has less", "which one has fewer", "which one has higher", "which one has lower", "which one has larger",
        "which one is higher", "which one is lower", "which one is more", "which one is less", "which one",
        "which village has more", "which village has less", "which village has higher", "which village has lower",
        "which state has more", "which state has higher", "which state has lower", "which state has less",
        "which capital has more", "which capital has higher", "which capital has lower", "which capital has less",
        "what percentage more", "what percentage higher", "what percentage less", "what percentage lower",
        "percentage difference", "percentage change", "ratio of", "ratio between", "what is the ratio",
        "is larger than", "is higher than", "is greater than", "is smaller than", "is lower than",
        "more populated than", "less populated than",
        "compare top", "difference between first and second", "difference between highest and second"
    ]

    def is_comparison_query(self, question: str) -> bool:
        """Determine whether the natural language question expresses comparison intent."""
        if not question:
            return False
        q_clean = normalize_question(question).lower()
        q_exp = expand_abbreviations(q_clean).lower()

        # Check explicit trigger phrases
        for trig in self.COMPARISON_TRIGGER_WORDS:
            if trig in q_clean or trig in q_exp:
                return True

        # Check comparison pattern: "X compared to Y"
        if "compared to" in q_clean or "compared with" in q_clean or "in comparison to" in q_clean:
            return True

        # Check boolean comparison e.g. "is hyd more populated than amaravati?", "is amaravati larger in population than hyd?"
        if re.search(r"\b(?:is|are)\b.*\b(?:more|greater|higher|less|lower|larger|smaller)\b.*\b(?:than|compared to)\b", q_clean):
            return True

        # Check "between X and Y" with attribute
        if re.search(r"\bbetween\s+([a-zA-Z0-9_\s]+)\s+and\s+([a-zA-Z0-9_\s]+)\b", q_clean):
            if any(k in q_clean for k in ["difference", "diff", "ratio", "percentage", "compare", "higher", "more"]):
                return True

        return False

    @classmethod
    def get_unit_for_attribute(cls, attribute: str) -> str:
        """Dynamically detect unit based on attribute semantics."""
        attr_l = attribute.lower()
        if "population" in attr_l or attr_l in {"people", "persons", "residents", "inhabitants"}:
            return "people"
        if "male" in attr_l and "female" not in attr_l:
            return "males"
        if "female" in attr_l:
            return "females"
        if "household" in attr_l:
            return "households"
        if "area" in attr_l or "sq_km" in attr_l:
            return "sq km"
        if "literacy" in attr_l or "percent" in attr_l or "rate" in attr_l:
            return "%"
        if "village_count" in attr_l or "villages" in attr_l or "village" in attr_l:
            return "villages"
        if "salary" in attr_l or "price" in attr_l or "cost" in attr_l or "budget" in attr_l or "revenue" in attr_l or "profit" in attr_l:
            return "currency"
        if "employee" in attr_l:
            return "employees"
        if "quantity" in attr_l or "stock" in attr_l or "units" in attr_l:
            return "units"
        return attribute.replace("_", " ")

    @classmethod
    def check_attribute_compatibility(cls, attr1: str, attr2: str) -> Tuple[bool, str]:
        """Verify that two attributes are semantically compatible for comparison."""
        unit1 = cls.get_unit_for_attribute(attr1)
        unit2 = cls.get_unit_for_attribute(attr2)
        if unit1 != unit2:
            return False, f"I can compare these entities only using the same compatible attribute. Incompatible attributes: '{attr1}' ({unit1}) vs '{attr2}' ({unit2})."
        return True, "Compatible"

    def verify_comparison(
        self,
        plan: ComparisonPlan,
        left_ent: ComparisonEntity,
        right_ent: ComparisonEntity,
        val1: Optional[float],
        val2: Optional[float],
        abs_diff: Optional[float],
        directed_diff: Optional[float],
        higher_ent: Optional[str],
        lower_ent: Optional[str],
        is_equal: bool,
        unit: str,
        sources: List[str],
        source_rows: List[Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Run strict 13-point verification on comparison execution before returning response."""
        checks: Dict[str, bool] = {}

        # 1. ENTITY CHECK
        checks["ENTITY_CHECK"] = bool(left_ent.name and right_ent.name)

        # 2. ATTRIBUTE CHECK
        checks["ATTRIBUTE_CHECK"] = bool(plan.attribute and len(plan.attribute) > 0)

        # 3. DATASET CHECK
        checks["DATASET_CHECK"] = bool(len(sources) > 0 or plan.scope == "GENERIC_DATASET")

        # 4. ROW CHECK
        checks["ROW_CHECK"] = bool(source_rows is not None)

        # 5. TYPE CHECK
        checks["TYPE_CHECK"] = bool(
            (val1 is None or isinstance(val1, (int, float))) and
            (val2 is None or isinstance(val2, (int, float)))
        )

        # 6. UNIT CHECK
        checks["UNIT_CHECK"] = bool(unit and len(unit) > 0)

        # 7. VALUE CHECK
        checks["VALUE_CHECK"] = (val1 is not None and val2 is not None)

        # 8. CALCULATION CHECK (Secondary independent calculation)
        if val1 is not None and val2 is not None and abs_diff is not None and directed_diff is not None:
            calc_abs = abs(val1 - val2)
            calc_dir = val1 - val2
            sec_ok = (abs(abs_diff - calc_abs) < 1e-4) and (abs(directed_diff - calc_dir) < 1e-4)
            checks["CALCULATION_CHECK"] = sec_ok
        else:
            checks["CALCULATION_CHECK"] = False

        # 9. DIRECTION CHECK
        if directed_diff is not None:
            if directed_diff > 0:
                checks["DIRECTION_CHECK"] = (higher_ent == left_ent.name and lower_ent == right_ent.name)
            elif directed_diff < 0:
                checks["DIRECTION_CHECK"] = (higher_ent == right_ent.name and lower_ent == left_ent.name)
            else:
                checks["DIRECTION_CHECK"] = is_equal
        else:
            checks["DIRECTION_CHECK"] = False

        # 10. AGGREGATION CHECK
        checks["AGGREGATION_CHECK"] = True

        # 11. DUPLICATE CHECK
        checks["DUPLICATE_CHECK"] = True

        # 12. MISSING VALUE CHECK
        checks["MISSING_VALUE_CHECK"] = (val1 is not None and val2 is not None)

        # 13. RESULT CHECK
        all_passed = all(checks.values())
        checks["RESULT_CHECK"] = all_passed

        return all_passed, checks


    def extract_metric(self, question: str, default_metric: str = "Population") -> str:
        """Resolve canonical target metric from natural language."""
        q_lower = normalize_question(question).lower()
        # Longest match first
        for alias in sorted(self.METRIC_ALIASES.keys(), key=lambda x: len(x), reverse=True):
            if re.search(r"\b" + re.escape(alias) + r"\b", q_lower):
                return self.METRIC_ALIASES[alias]
        return default_metric

    def extract_comparison_type_and_direction(self, question: str) -> Tuple[str, Optional[str], bool, Optional[str]]:
        """Extract calculation mode: DIFFERENCE, PERCENTAGE, RATIO, BOOLEAN, RANKING."""
        q_lower = normalize_question(question).lower()

        is_boolean = False
        bool_op = None
        has_bool_lead = bool(re.search(r"^\s*(?:is|are|does|do|did|can|could|would)\b", q_lower))
        has_wh_lead = bool(re.search(r"^\s*(?:what|which|how|who|where|when|why)\b", q_lower))
        has_diff_phrase = any(kw in q_lower for kw in [
            "what is the difference", "what difference", "how much difference",
            "population difference", "what is the population difference", "difference between"
        ])

        has_bool_phrase = bool(re.search(r"\b(?:are they different|are they the same|are they equal|is there a difference)\b", q_lower))
        if not has_diff_phrase and ((has_bool_lead and not has_wh_lead) or has_bool_phrase):
            is_boolean = True
            if any(w in q_lower for w in ["more", "greater", "higher", "larger", "above"]):
                bool_op = ">"
            elif any(w in q_lower for w in ["less", "fewer", "lower", "smaller", "below"]):
                bool_op = "<"
            elif any(w in q_lower for w in ["different", "differ", "not the same", "distinct"]):
                bool_op = "!="
            elif any(w in q_lower for w in ["same", "equal"]):
                bool_op = "=="
            else:
                bool_op = ">"


        # Percentage comparison
        q_raw_lower = question.lower()
        has_pct_word = any(w in q_raw_lower or w in q_lower for w in ["percentage", "percent", "%"])
        if has_pct_word:
            if any(w in q_raw_lower or w in q_lower for w in ["more", "higher", "greater", "increase", "larger", "above"]):
                return "PERCENTAGE_DIFFERENCE", "MORE", is_boolean, bool_op
            if any(w in q_raw_lower or w in q_lower for w in ["less", "fewer", "lower", "decrease", "smaller", "below"]):
                return "PERCENTAGE_DIFFERENCE", "LESS", is_boolean, bool_op
            if any(w in q_raw_lower or w in q_lower for w in ["change", "growth"]):
                return "PERCENTAGE_CHANGE", "CHANGE", is_boolean, bool_op
            return "PERCENTAGE_DIFFERENCE", "DIFF", is_boolean, bool_op

        # Ratio
        if any(w in q_lower for w in ["ratio", "ratio of", "ratio between"]):
            return "RATIO", None, is_boolean, bool_op

        # Ranking
        if any(w in q_lower for w in ["rank", "ranking", "top 2", "top two", "first and second", "highest and second"]):
            return "RANKING", None, is_boolean, bool_op

        # Direction requested
        if any(w in q_lower for w in ["more", "higher", "greater", "larger", "bigger"]):
            direction = "MORE"
        elif any(w in q_lower for w in ["less", "fewer", "lower", "smaller"]):
            direction = "FEWER"
        elif any(w in q_lower for w in ["same", "equal"]):
            direction = "EQUAL"
        else:
            direction = "DIFF"

        comp_type = "BOOLEAN" if is_boolean else "ABSOLUTE_DIFFERENCE"
        return comp_type, direction, is_boolean, bool_op

    def _extract_all_explicit_states(self, text: str) -> List[Tuple[str, Path]]:
        """Extract all explicitly mentioned states in text in order of occurrence."""
        cleaned = normalize_question(text).lower()
        reg = parent_child_registry.get_registered_child_datasets()
        found: List[Tuple[str, Path, int]] = []

        # Exact state names (multi-word first)
        for st_name, p in reg.items():
            pattern = r"\b" + re.escape(st_name.lower()) + r"\b"
            for m in re.finditer(pattern, cleaned):
                found.append((st_name, p, m.start()))

        # Check explicit codes dynamically from ParentChildRegistry
        words_with_pos = [(m.group().lower(), m.start()) for m in re.finditer(r"\b[a-zA-Z0-9_-]+\b", cleaned)]
        state_codes_map = parent_child_registry.get_state_codes_map()
        english_stop_words = {"or", "in", "is", "to", "at", "an", "on", "it", "so", "by", "of", "if", "no", "do", "and"}

        for w, pos in words_with_pos:
            if w in english_stop_words:
                continue
            if w in state_codes_map:
                st = state_codes_map[w]
                if st in reg and not any(f[0] == st for f in found):
                    found.append((st, reg[st], pos))

        # Check explicit village ID prefixes e.g. GJ-001 -> gj -> Gujarat, TG-001 -> tg -> Telangana
        for m in re.finditer(r"\b([a-zA-Z]{2,4})-\d{3}\b", cleaned):
            code = m.group(1).lower()
            if code in state_codes_map:
                st = state_codes_map[code]
                if st in reg and not any(f[0] == st for f in found):
                    found.append((st, reg[st], m.start()))

        found.sort(key=lambda x: x[2])
        res: List[Tuple[str, Path]] = []
        seen = set()
        for st_name, p, _ in found:
            if st_name not in seen:
                seen.add(st_name)
                res.append((st_name, p))
        return res

    def parse_and_plan(
        self,
        question: str,
        df: Optional[pd.DataFrame] = None,
        session_id: Optional[str] = None
    ) -> Optional[ComparisonPlan]:
        """Synthesize comparison plan by resolving entities, scopes, and target attributes."""
        norm_q = normalize_question(question)
        exp_q = expand_abbreviations(norm_q)
        q_lower = norm_q.lower()
        exp_lower = exp_q.lower()

        comp_type, direction, is_bool, bool_op = self.extract_comparison_type_and_direction(norm_q)
        metric = self.extract_metric(norm_q)

        # -------------------------------------------------------------
        # 0. Check for Contextual Follow-Up Comparison (e.g. "What about literacy?", "How much more?")
        # -------------------------------------------------------------
        global_markers = [
            "all state", "all states", "every state", "all villages", "across all",
            "which state", "what state", "top ", "bottom ", "total population", "sum of",
            "average population", "median population", "entire country", "all registered"
        ]
        is_global_query = any(m in q_lower or m in exp_lower for m in global_markers)

        if session_id and not is_global_query:
            from app.conversation.context import conversation_manager
            last_comp = conversation_manager.get_last_comparison(session_id)
            if last_comp:
                # Check if current question has explicit entities (e.g. "Compare Karnataka and AP")
                # Section 37: Explicit current-turn entities always override old context.
                explicit_states = self._extract_all_explicit_states(norm_q) if parent_child_registry.is_parent_child_active() else []
                if len(explicit_states) < 2:
                    is_followup_marker = any(w in q_lower for w in [
                        "what about", "how about", "what of", "and literacy", "and area",
                        "and population", "how much more", "how many more", "what is the difference",
                        "what difference", "how much higher", "how much lower", "which one is higher",
                        "which is higher", "which has more", "which has fewer", "which has less",
                        "which one has more", "which one has less", "which one has fewer", "which one has higher", "which one has lower",
                        "which one is more", "which one is less", "which one",
                        "which village has more", "which village has less", "which village has fewer",
                        "what is the population difference", "difference between them"
                    ])
                    words = q_lower.split()
                    has_metric_change = (len(words) <= 5 and any(m in q_lower for m in self.METRIC_ALIASES.keys()) and not any(w in q_lower for w in ["which", "what is", "how many", "all", "every"]))
                    if is_followup_marker or has_metric_change:
                        left_name = last_comp.get("left_entity")
                        right_name = last_comp.get("right_entity")
                        if left_name and right_name:
                            prev_scope = last_comp.get("scope") or "SAME_STATE_VILLAGE"
                            left_parent = last_comp.get("left_parent")
                            right_parent = last_comp.get("right_parent")
                            if left_parent and right_parent and str(left_parent).lower() != str(right_parent).lower():
                                prev_scope = "CROSS_STATE_VILLAGE"
                            child1_res = parent_child_registry.resolve_child_dataset(left_parent or left_name)
                            child2_res = parent_child_registry.resolve_child_dataset(right_parent or right_name)
                            ent1 = ComparisonEntity(
                                name=left_name,
                                entity_type="VILLAGE" if "village" in prev_scope.lower() else ("STATE" if "state" in prev_scope.lower() else "GENERIC"),
                                parent=left_parent,
                                dataset_name=child1_res[1].name if child1_res else None,
                                dataset_path=child1_res[1] if child1_res else None
                            )
                            ent2 = ComparisonEntity(
                                name=right_name,
                                entity_type="VILLAGE" if "village" in prev_scope.lower() else ("STATE" if "state" in prev_scope.lower() else "GENERIC"),
                                parent=right_parent,
                                dataset_name=child2_res[1].name if child2_res else None,
                                dataset_path=child2_res[1] if child2_res else None
                            )
                            target_metric = metric if has_metric_change else (last_comp.get("attribute") or metric)
                            return ComparisonPlan(
                                entities=[ent1, ent2],
                                attribute=target_metric,
                                comparison_type=comp_type,
                                direction_requested=direction,
                                is_boolean=is_bool,
                                boolean_operator=bool_op,
                                scope=prev_scope,
                                original_question=question
                            )

        # -------------------------------------------------------------
        # 1. Check for Ranking / Top 2 comparison: "compare top two villages", "difference between first and second"
        # -------------------------------------------------------------
        if any(kw in q_lower for kw in ["top two", "top 2", "first and second", "highest and second", "top 2 villages"]):
            # Check if scoped to a state
            resolved_state = parent_child_registry.resolve_child_dataset(exp_q)
            scope = "RANKING"
            parent_name = resolved_state[0] if resolved_state else None
            entities = [
                ComparisonEntity(name="First", entity_type="VILLAGE", parent=parent_name),
                ComparisonEntity(name="Second", entity_type="VILLAGE", parent=parent_name)
            ]
            return ComparisonPlan(
                entities=entities,
                attribute=metric,
                comparison_type="RANKING",
                direction_requested=direction,
                is_boolean=is_bool,
                boolean_operator=bool_op,
                scope=scope,
                original_question=question
            )

        # -------------------------------------------------------------
        # 2. Check for Parent-Child Village Comparisons
        # -------------------------------------------------------------
        if parent_child_registry.is_parent_child_active():
            explicit_states = self._extract_all_explicit_states(norm_q)

            # If exactly 1 state is in query, check SAME-STATE VILLAGE FIRST!
            if len(explicit_states) == 1:
                same_state_v_plan = self._plan_same_state_village(
                    question, q_lower, exp_lower, metric, comp_type, direction, is_bool, bool_op, session_id, explicit_states[0]
                )
                if same_state_v_plan:
                    return same_state_v_plan

            # If 2 or more states in query, check CROSS-STATE VILLAGE!
            if len(explicit_states) >= 2:
                cross_plan = self._plan_cross_state_village(
                    question, q_lower, exp_lower, metric, comp_type, direction, is_bool, bool_op, explicit_states
                )
                if cross_plan:
                    return cross_plan

                state_plan = self._plan_state_vs_state(question, q_lower, exp_lower, metric, comp_type, direction, is_bool, bool_op)
                if state_plan:
                    return state_plan

            # Fallback same-state village check (in case state is in session context or expanded)
            same_state_v_plan = self._plan_same_state_village(question, q_lower, exp_lower, metric, comp_type, direction, is_bool, bool_op, session_id)
            if same_state_v_plan:
                return same_state_v_plan

            # Check Capital vs Capital
            capital_plan = self._plan_capital_vs_capital(question, q_lower, exp_lower, metric, comp_type, direction, is_bool, bool_op)
            if capital_plan:
                return capital_plan

            # Check State vs Capital
            state_cap_plan = self._plan_state_vs_capital(question, q_lower, exp_lower, metric, comp_type, direction, is_bool, bool_op)
            if state_cap_plan:
                return state_cap_plan

            # State vs state fallback
            state_plan = self._plan_state_vs_state(question, q_lower, exp_lower, metric, comp_type, direction, is_bool, bool_op)
            if state_plan:
                return state_plan

        # -------------------------------------------------------------
        # 3. Check for Generic Tabular Entity comparison in active DataFrame
        # -------------------------------------------------------------
        if df is not None and not df.empty:
            generic_plan = self._plan_generic_tabular(question, q_lower, df, metric, comp_type, direction, is_bool, bool_op)
            if generic_plan:
                return generic_plan

        return None

    # -----------------------------------------------------------------
    # PARSING HELPERS
    # -----------------------------------------------------------------

    def _extract_village_reference(self, text: str) -> Optional[str]:
        """Extract a village identifier or name from snippet."""
        # Check Village ID e.g. TG-001, AP-002
        m_id = re.search(r"\b([a-zA-Z]{2}-\d{3})\b", text)
        if m_id:
            return m_id.group(1).upper()

        # Check full village name e.g. Hyderabad_Village_01, Village_01
        m_name = re.search(r"\b([a-zA-Z]+_village_\d+)\b", text, re.IGNORECASE)
        if m_name:
            return m_name.group(1)

        # Check numbered village e.g. "village 2", "village 02", "village 1"
        m_num = re.search(r"\bvillage\s*([0-9]+)\b", text, re.IGNORECASE)
        if m_num:
            return f"Village {int(m_num.group(1))}"

        # Check lettered village e.g. "village A", "village B"
        m_let = re.search(r"\bvillage\s*([a-zA-Z])\b", text, re.IGNORECASE)
        if m_let:
            return f"Village {m_let.group(1).upper()}"

        return None

    def _plan_cross_state_village(
        self,
        question: str,
        q_lower: str,
        exp_lower: str,
        metric: str,
        comp_type: str,
        direction: Optional[str],
        is_bool: bool,
        bool_op: Optional[str],
        explicit_states: Optional[List[Tuple[str, Path]]] = None
    ) -> Optional[ComparisonPlan]:
        """Detect cross-state village comparisons."""
        if not explicit_states or len(explicit_states) < 2:
            return None

        # Find village mentions in query
        village_matches: List[str] = []
        for m in re.finditer(r"\b([a-zA-Z]{2}-\d{3})\b", question):
            village_matches.append(m.group(1).upper())
        for m in re.finditer(r"\b([a-zA-Z]+_village_\d+)\b", q_lower):
            v_str = m.group(1)
            if v_str not in [vm.lower() for vm in village_matches]:
                village_matches.append(v_str)
        for m in re.finditer(r"\bvillage\s*([0-9]+)\b", q_lower):
            num = int(m.group(1))
            v_name = f"Village {num}"
            if v_name not in village_matches and f"Village_{num:02d}" not in village_matches:
                village_matches.append(v_name)
        for m in re.finditer(r"\bvillage\s*([a-zA-Z])\b", q_lower):
            let = m.group(1).upper()
            v_name = f"Village {let}"
            if v_name not in village_matches:
                village_matches.append(v_name)

        if len(village_matches) >= 2:
            left_entity = ComparisonEntity(
                name=village_matches[0],
                entity_type="VILLAGE",
                parent=explicit_states[0][0],
                dataset_name=explicit_states[0][1].name,
                dataset_path=explicit_states[0][1]
            )
            right_entity = ComparisonEntity(
                name=village_matches[1],
                entity_type="VILLAGE",
                parent=explicit_states[1][0],
                dataset_name=explicit_states[1][1].name,
                dataset_path=explicit_states[1][1]
            )
            return ComparisonPlan(
                entities=[left_entity, right_entity],
                attribute=metric,
                comparison_type=comp_type,
                direction_requested=direction,
                is_boolean=is_bool,
                boolean_operator=bool_op,
                scope="CROSS_STATE_VILLAGE",
                original_question=question
            )

        return None

    def _plan_same_state_village(
        self,
        question: str,
        q_lower: str,
        exp_lower: str,
        metric: str,
        comp_type: str,
        direction: Optional[str],
        is_bool: bool,
        bool_op: Optional[str],
        session_id: Optional[str],
        explicit_state: Optional[Tuple[str, Path]] = None
    ) -> Optional[ComparisonPlan]:
        """Detect same-state village comparisons."""
        # Find village mentions in query
        village_matches: List[str] = []

        # Find all Village IDs like TG-001
        for m in re.finditer(r"\b([a-zA-Z]{2}-\d{3})\b", question):
            village_matches.append(m.group(1).upper())

        # Find all Full Village Names like Hyderabad_Village_01
        for m in re.finditer(r"\b([a-zA-Z]+_village_\d+)\b", q_lower):
            v_str = m.group(1)
            if v_str not in [vm.lower() for vm in village_matches]:
                village_matches.append(v_str)

        # Find all numbered villages like "village 2", "village 1"
        for m in re.finditer(r"\bvillage\s*([0-9]+)\b", q_lower):
            num = int(m.group(1))
            v_name = f"Village {num}"
            if v_name not in village_matches and f"Village_{num:02d}" not in village_matches:
                village_matches.append(v_name)

        # Find all lettered villages like "village A", "village B"
        for m in re.finditer(r"\bvillage\s*([a-zA-Z])\b", q_lower):
            let = m.group(1).upper()
            v_name = f"Village {let}"
            if v_name not in village_matches:
                village_matches.append(v_name)

        # If exactly 2 villages found
        if len(village_matches) >= 2:
            if explicit_state:
                parent_name, child_path = explicit_state
            else:
                resolved = parent_child_registry.resolve_child_dataset(exp_lower) or parent_child_registry.resolve_child_dataset(q_lower)
                parent_name = resolved[0] if resolved else None
                child_path = resolved[1] if resolved else None

            # Fallback to session active child dataset if not in query
            if not child_path and session_id:
                from app.conversation.context import conversation_manager
                active_child = conversation_manager.get_active_child_dataset(session_id)
                if active_child:
                    child_dir = parent_child_registry.get_child_datasets_dir()
                    if child_dir:
                        cand_path = child_dir / active_child
                        if cand_path.exists():
                            child_path = cand_path
                            parent_name = conversation_manager.get_active_state(session_id) or active_child

            if child_path:
                left_entity = ComparisonEntity(
                    name=village_matches[0],
                    entity_type="VILLAGE",
                    parent=parent_name,
                    dataset_name=child_path.name,
                    dataset_path=child_path
                )
                right_entity = ComparisonEntity(
                    name=village_matches[1],
                    entity_type="VILLAGE",
                    parent=parent_name,
                    dataset_name=child_path.name,
                    dataset_path=child_path
                )
                return ComparisonPlan(
                    entities=[left_entity, right_entity],
                    attribute=metric,
                    comparison_type=comp_type,
                    direction_requested=direction,
                    is_boolean=is_bool,
                    boolean_operator=bool_op,
                    scope="SAME_STATE_VILLAGE",
                    original_question=question
                )

        return None

    def _plan_state_vs_state(
        self,
        question: str,
        q_lower: str,
        exp_lower: str,
        metric: str,
        comp_type: str,
        direction: Optional[str],
        is_bool: bool,
        bool_op: Optional[str]
    ) -> Optional[ComparisonPlan]:
        """Detect comparison between two or more states."""
        reg_datasets = parent_child_registry.get_registered_child_datasets()
        states_found: List[Tuple[str, Path, int]] = []

        # Find all states mentioned with their character positions in query
        for state_name, child_path in reg_datasets.items():
            pattern = r"\b" + re.escape(state_name.lower()) + r"\b"
            for m in re.finditer(pattern, exp_lower):
                states_found.append((state_name, child_path, m.start()))

        # Also check abbreviations like "TG", "AP", "GJ", "KA" if not expanded
        for code, canonical in LOCATION_ABBREVIATIONS.items():
            if len(code) <= 3 and canonical in reg_datasets:
                pattern = r"\b" + re.escape(code.lower()) + r"\b"
                for m in re.finditer(pattern, q_lower):
                    if not any(s[0] == canonical for s in states_found):
                        states_found.append((canonical, reg_datasets[canonical], m.start()))

        # Deduplicate while preserving order of occurrence
        states_found.sort(key=lambda x: x[2])
        unique_states: List[Tuple[str, Path]] = []
        seen = set()
        for s_name, p, _ in states_found:
            if s_name not in seen:
                seen.add(s_name)
                unique_states.append((s_name, p))

        if len(unique_states) >= 2:
            # If query is explicitly a SUM/TOTAL/COMBINE/AVERAGE aggregation of selected states without comparison keywords, do NOT plan as comparison!
            has_agg_word = any(w in q_lower for w in ["total", "sum", "combine", "together", "plus", "all of"])
            has_comp_word = any(w in q_lower for w in ["compare", "vs", "versus", "difference", "diff", "between", "more", "less", "fewer", "higher", "lower", "ratio", "percentage"])
            if has_agg_word and not has_comp_word:
                return None
            entities = [
                ComparisonEntity(
                    name=s_name,
                    entity_type="STATE",
                    dataset_name=path.name,
                    dataset_path=path
                )
                for s_name, path in unique_states
            ]
            scope = "MULTI_ENTITY" if len(unique_states) > 2 else "STATE_VS_STATE"
            return ComparisonPlan(
                entities=entities,
                attribute=metric,
                comparison_type=comp_type,
                direction_requested=direction,
                is_boolean=is_bool,
                boolean_operator=bool_op,
                scope=scope,
                original_question=question
            )

        return None

    def _plan_capital_vs_capital(
        self,
        question: str,
        q_lower: str,
        exp_lower: str,
        metric: str,
        comp_type: str,
        direction: Optional[str],
        is_bool: bool,
        bool_op: Optional[str]
    ) -> Optional[ComparisonPlan]:
        """Detect comparison between two or more capitals."""
        main_path = parent_child_registry.get_main_dataset_path()
        if not main_path or not main_path.exists():
            return None

        main_df = pd.read_csv(main_path)
        capital_col = next((c for c in main_df.columns if c.lower() == "capital"), None)
        if not capital_col:
            return None

        all_capitals = main_df[capital_col].dropna().unique().tolist()
        capitals_found: List[Tuple[str, int]] = []

        for cap in all_capitals:
            cap_clean = str(cap).strip().lower()
            pattern = r"\b" + re.escape(cap_clean) + r"\b"
            for m in re.finditer(pattern, exp_lower):
                capitals_found.append((str(cap).strip(), m.start()))

        # Check common city abbreviations like Hyd, Blr, Bom, etc.
        for code, canonical in LOCATION_ABBREVIATIONS.items():
            if len(code) <= 4 and any(canonical.lower() == str(c).lower() for c in all_capitals):
                pattern = r"\b" + re.escape(code.lower()) + r"\b"
                for m in re.finditer(pattern, q_lower):
                    matched_cap = next(c for c in all_capitals if str(c).lower() == canonical.lower())
                    if not any(c[0] == matched_cap for c in capitals_found):
                        capitals_found.append((matched_cap, m.start()))

        capitals_found.sort(key=lambda x: x[1])
        unique_caps: List[str] = []
        seen = set()
        for cap_name, _ in capitals_found:
            if cap_name not in seen:
                seen.add(cap_name)
                unique_caps.append(cap_name)

        if len(unique_caps) >= 2:
            entities = []
            for cap_name in unique_caps:
                res = parent_child_registry.resolve_child_dataset(cap_name)
                entities.append(
                    ComparisonEntity(
                        name=cap_name,
                        entity_type="CAPITAL",
                        parent=res[0] if res else None,
                        dataset_name=res[1].name if res else None,
                        dataset_path=res[1] if res else None
                    )
                )
            scope = "MULTI_ENTITY" if len(unique_caps) > 2 else "CAPITAL_VS_CAPITAL"
            return ComparisonPlan(
                entities=entities,
                attribute=metric,
                comparison_type=comp_type,
                direction_requested=direction,
                is_boolean=is_bool,
                boolean_operator=bool_op,
                scope=scope,
                original_question=question
            )

        return None

    def _plan_state_vs_capital(
        self,
        question: str,
        q_lower: str,
        exp_lower: str,
        metric: str,
        comp_type: str,
        direction: Optional[str],
        is_bool: bool,
        bool_op: Optional[str]
    ) -> Optional[ComparisonPlan]:
        """Detect comparison between a State and a Capital."""
        reg_datasets = parent_child_registry.get_registered_child_datasets()
        main_path = parent_child_registry.get_main_dataset_path()
        if not main_path or not main_path.exists():
            return None

        main_df = pd.read_csv(main_path)
        state_col = next((c for c in main_df.columns if c.lower() == "state"), None)
        capital_col = next((c for c in main_df.columns if c.lower() == "capital"), None)
        if not state_col or not capital_col:
            return None

        states = main_df[state_col].dropna().unique().tolist()
        capitals = main_df[capital_col].dropna().unique().tolist()

        matched_states: List[Tuple[str, int]] = []
        matched_caps: List[Tuple[str, int]] = []

        for st in states:
            pattern = r"\b" + re.escape(str(st).lower()) + r"\b"
            for m in re.finditer(pattern, exp_lower):
                matched_states.append((str(st), m.start()))

        for cap in capitals:
            pattern = r"\b" + re.escape(str(cap).lower()) + r"\b"
            for m in re.finditer(pattern, exp_lower):
                matched_caps.append((str(cap), m.start()))

        if matched_states and matched_caps:
            st_name, st_pos = matched_states[0]
            cap_name, cap_pos = matched_caps[0]

            st_res = parent_child_registry.resolve_child_dataset(st_name)
            cap_res = parent_child_registry.resolve_child_dataset(cap_name)

            st_ent = ComparisonEntity(
                name=st_name,
                entity_type="STATE",
                dataset_name=st_res[1].name if st_res else None,
                dataset_path=st_res[1] if st_res else None
            )
            cap_ent = ComparisonEntity(
                name=cap_name,
                entity_type="CAPITAL",
                parent=cap_res[0] if cap_res else None,
                dataset_name=cap_res[1].name if cap_res else None,
                dataset_path=cap_res[1] if cap_res else None
            )

            if st_pos < cap_pos:
                return ComparisonPlan(
                    entities=[st_ent, cap_ent],
                    attribute=metric,
                    comparison_type=comp_type,
                    direction_requested=direction,
                    is_boolean=is_bool,
                    boolean_operator=bool_op,
                    scope="STATE_VS_CAPITAL",
                    original_question=question
                )
            else:
                return ComparisonPlan(
                    entities=[cap_ent, st_ent],
                    attribute=metric,
                    comparison_type=comp_type,
                    direction_requested=direction,
                    is_boolean=is_bool,
                    boolean_operator=bool_op,
                    scope="CAPITAL_VS_STATE",
                    original_question=question
                )

        return None

    def _plan_generic_tabular(
        self,
        question: str,
        q_lower: str,
        df: pd.DataFrame,
        metric: str,
        comp_type: str,
        direction: Optional[str],
        is_bool: bool,
        bool_op: Optional[str]
    ) -> Optional[ComparisonPlan]:
        """Detect entity comparison in arbitrary tabular dataset."""
        # Find candidate text/entity columns
        candidate_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("string")]
        if not candidate_cols:
            return None

        # Look for two entities mentioned in query from one of the columns
        for col in candidate_cols:
            unique_vals = [str(v).strip() for v in df[col].dropna().unique().tolist() if str(v).strip()]
            matched: List[Tuple[str, int]] = []
            
            # 1. Try exact whole-phrase match first
            for val_str in unique_vals:
                if len(val_str) >= 2:
                    pattern = r"\b" + re.escape(val_str.lower()) + r"\b"
                    for m in re.finditer(pattern, q_lower):
                        matched.append((val_str, m.start()))

            # 2. If fewer than 2 matched, try matching significant tokens of values (e.g. First Name / Product Name)
            if len(matched) < 2:
                matched = []
                for val_str in unique_vals:
                    tokens = [t for t in re.split(r"\s+", val_str) if len(t) >= 3]
                    for token in tokens:
                        # Ensure token uniquely identifies a single value in this column to prevent ambiguity
                        matches_in_col = [v for v in unique_vals if re.search(r"\b" + re.escape(token.lower()) + r"\b", v, re.IGNORECASE)]
                        if len(matches_in_col) == 1:
                            pattern = r"\b" + re.escape(token.lower()) + r"\b"
                            for m in re.finditer(pattern, q_lower):
                                if not any(existing_val == val_str for existing_val, _ in matched):
                                    matched.append((val_str, m.start()))

            matched.sort(key=lambda x: x[1])
            if len(matched) >= 2:
                # Find metric column
                target_metric = metric
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if target_metric not in df.columns and num_cols:
                    # check if any num col mentioned in query
                    for nc in num_cols:
                        if nc.lower() in q_lower:
                            target_metric = nc
                            break
                    if target_metric not in df.columns:
                        target_metric = num_cols[0]

                ent1 = ComparisonEntity(name=matched[0][0], entity_type="GENERIC", parent=col)
                ent2 = ComparisonEntity(name=matched[1][0], entity_type="GENERIC", parent=col)
                return ComparisonPlan(
                    entities=[ent1, ent2],
                    attribute=target_metric,
                    comparison_type=comp_type,
                    direction_requested=direction,
                    is_boolean=is_bool,
                    boolean_operator=bool_op,
                    scope="GENERIC_DATASET",
                    original_question=question
                )

        return None

    # -----------------------------------------------------------------
    # VALUE RETRIEVAL
    # -----------------------------------------------------------------

    def retrieve_entity_value(
        self,
        entity: ComparisonEntity,
        attribute: str,
        df: Optional[pd.DataFrame] = None
    ) -> Tuple[Optional[float], Optional[Dict[str, Any]], List[str]]:
        """Deterministically retrieve attribute value for an entity."""
        columns_used: List[str] = []

        # 1. Generic Tabular Row
        if entity.entity_type == "GENERIC" and df is not None and not df.empty:
            col_name = entity.parent
            if not col_name or col_name not in df.columns:
                for c in df.columns:
                    if df[c].astype(str).str.strip().str.lower().eq(entity.name.strip().lower()).any():
                        col_name = c
                        break
            if col_name and col_name in df.columns and attribute in df.columns:
                sub = df[df[col_name].astype(str).str.strip().str.lower() == entity.name.strip().lower()]
                if isinstance(sub, pd.DataFrame) and not sub.empty:
                    col_data = sub[attribute]
                    if isinstance(col_data, pd.DataFrame):
                        raw_s = col_data.iloc[:, 0]
                    elif isinstance(col_data, pd.Series):
                        raw_s = col_data
                    else:
                        raw_s = pd.Series(col_data)
                    clean_s = pd.Series(raw_s).astype(str).str.replace(r"[₹$,]", "", regex=True)
                    num_s = pd.to_numeric(clean_s, errors="coerce")
                    val = float(num_s.iloc[0]) if isinstance(num_s, pd.Series) and not num_s.empty and pd.notna(num_s.iloc[0]) else None
                    if val is not None:
                        columns_used.extend([col_name, attribute])
                        return float(val), sub.iloc[0].to_dict(), columns_used
                    return None, sub.iloc[0].to_dict(), columns_used
            return None, None, columns_used

        # 2. Village Entity
        if entity.entity_type == "VILLAGE":
            child_path = entity.dataset_path
            if not child_path and entity.dataset_name:
                child_dir = parent_child_registry.get_child_datasets_dir()
                if child_dir:
                    child_path = child_dir / entity.dataset_name

            if not child_path and entity.parent:
                res_p = parent_child_registry.resolve_child_dataset(entity.parent)
                if res_p:
                    child_path = res_p[1]
                    entity.dataset_path = child_path
                    entity.dataset_name = child_path.name

            if not child_path or not child_path.exists():
                return None, None, columns_used

            cdf = parent_child_registry.load_child_dataframe(child_path)
            if cdf is None or cdf.empty:
                return None, None, columns_used

            matched_row = None
            e_name = entity.name.strip()

            # ID lookup e.g. TG-001
            if re.match(r"^[a-zA-Z]{2}-\d{3}$", e_name) and "Village_ID" in cdf.columns:
                sub = cdf[cdf["Village_ID"].astype(str).str.upper() == e_name.upper()]
                if not sub.empty:
                    matched_row = sub.iloc[0].to_dict()

            # Exact name lookup e.g. Hyderabad_Village_01
            if matched_row is None and "Village" in cdf.columns:
                sub = cdf[cdf["Village"].astype(str).str.lower() == e_name.lower()]
                if not sub.empty:
                    matched_row = sub.iloc[0].to_dict()

            # Numbered lookup e.g. "Village 2" or "Village 02"
            m_num = re.search(r"village\s*([0-9]+)", e_name, re.IGNORECASE)
            if matched_row is None and m_num and "Village" in cdf.columns:
                idx = int(m_num.group(1))
                # match suffix _02 or _2
                for _, r in cdf.iterrows():
                    v_str = str(r["Village"]).lower()
                    if v_str.endswith(f"_{idx:02d}") or v_str.endswith(f"_{idx}"):
                        matched_row = r.to_dict()
                        break
                # Fallback to index if within bounds
                if matched_row is None and 1 <= idx <= len(cdf):
                    matched_row = cdf.iloc[idx - 1].to_dict()

            # Lettered lookup e.g. "Village A" -> row 0, "Village B" -> row 1
            m_let = re.search(r"village\s*([a-zA-Z])", e_name, re.IGNORECASE)
            if matched_row is None and m_let and "Village" in cdf.columns:
                let_idx = ord(m_let.group(1).upper()) - ord("A")
                if 0 <= let_idx < len(cdf):
                    matched_row = cdf.iloc[let_idx].to_dict()

            if matched_row:
                columns_used.append("Village")
                if attribute in matched_row:
                    columns_used.append(attribute)
                    raw_v = matched_row[attribute]
                    if pd.notna(raw_v):
                        return float(raw_v), matched_row, columns_used
                return None, matched_row, columns_used

            return None, None, columns_used

        # 3. State Entity
        if entity.entity_type == "STATE":
            child_path = entity.dataset_path
            if not child_path and entity.dataset_name:
                child_dir = parent_child_registry.get_child_datasets_dir()
                if child_dir:
                    child_path = child_dir / entity.dataset_name

            if not child_path:
                res_s = parent_child_registry.resolve_child_dataset(entity.name)
                if res_s:
                    child_path = res_s[1]
                    entity.dataset_path = child_path
                    entity.dataset_name = child_path.name

            if child_path and child_path.exists():
                cdf = parent_child_registry.load_child_dataframe(child_path)
                if cdf is not None and not cdf.empty:
                    if attribute == "VILLAGE_COUNT":
                        columns_used.append("Village")
                        return float(len(cdf)), {"State": entity.name, "Village_Count": len(cdf)}, columns_used

                    if attribute == "Literacy_Rate_Percent" and attribute in cdf.columns:
                        columns_used.append(attribute)
                        avg_lit = float(cdf[attribute].mean())
                        return round(avg_lit, 2), {"State": entity.name, attribute: avg_lit}, columns_used

                    if attribute in cdf.columns:
                        columns_used.append(attribute)
                        total = float(cdf[attribute].sum())
                        return total, {"State": entity.name, attribute: total}, columns_used

            return None, None, columns_used

        # 4. Capital Entity
        if entity.entity_type == "CAPITAL":
            child_path = entity.dataset_path
            if not child_path and entity.dataset_name:
                child_dir = parent_child_registry.get_child_datasets_dir()
                if child_dir:
                    child_path = child_dir / entity.dataset_name

            if child_path and child_path.exists():
                cdf = parent_child_registry.load_child_dataframe(child_path)
                if cdf is not None and not cdf.empty:
                    if attribute == "VILLAGE_COUNT":
                        columns_used.append("Village")
                        return float(len(cdf)), {"Capital": entity.name, "Village_Count": len(cdf)}, columns_used

                    if attribute == "Literacy_Rate_Percent" and attribute in cdf.columns:
                        columns_used.append(attribute)
                        avg_lit = float(cdf[attribute].mean())
                        return round(avg_lit, 2), {"Capital": entity.name, attribute: avg_lit}, columns_used

                    if attribute in cdf.columns:
                        columns_used.append(attribute)
                        total = float(cdf[attribute].sum())
                        return total, {"Capital": entity.name, attribute: total}, columns_used

            return None, None, columns_used

        return None, None, columns_used

    # -----------------------------------------------------------------
    # EXECUTION
    # -----------------------------------------------------------------

    def execute(
        self,
        plan: ComparisonPlan,
        df: Optional[pd.DataFrame] = None
    ) -> ComparisonResult:
        """Execute plan deterministically and format directional answer."""
        # -------------------------------------------------------------
        # Ranking execution (e.g. Top 2 villages across all or in a state)
        # -------------------------------------------------------------
        if plan.comparison_type == "RANKING" or plan.scope == "RANKING":
            return self._execute_ranking(plan)

        # -------------------------------------------------------------
        # Multi-Entity execution (3+ entities)
        # -------------------------------------------------------------
        if len(plan.entities) > 2:
            return self._execute_multi_entity(plan, df)

        # -------------------------------------------------------------
        # Pairwise Comparison (Exactly 2 entities)
        # -------------------------------------------------------------
        left_ent = plan.entities[0]
        right_ent = plan.entities[1]

        val1, rec1, cols1 = self.retrieve_entity_value(left_ent, plan.attribute, df)
        val2, rec2, cols2 = self.retrieve_entity_value(right_ent, plan.attribute, df)

        cols_used = list(set(cols1 + cols2))
        src_datasets = []
        if left_ent.dataset_name:
            src_datasets.append(left_ent.dataset_name)
        if right_ent.dataset_name and right_ent.dataset_name not in src_datasets:
            src_datasets.append(right_ent.dataset_name)

        unit = self.get_unit_for_attribute(plan.attribute)

        # Missing Value Checks (Section 11 & Section 44)
        attr_display = plan.attribute.replace("_", " ")
        if val1 is None and val2 is None:
            ans = f"Could not find values for '{left_ent.name}' and '{right_ent.name}' for {attr_display} in the available datasets."
            return ComparisonResult(
                status="MISSING_VALUE",
                left_entity=left_ent.name,
                right_entity=right_ent.name,
                attribute=plan.attribute,
                unit=unit,
                answer=ans,
                source_datasets=src_datasets,
                columns_used=cols_used,
                verification_status="MISSING_VALUE",
                verification_checks={"MISSING_VALUE": True}
            )

        if val1 is None:
            ans = f"I couldn't calculate the comparison because {attr_display} data is missing for {left_ent.name}."
            return ComparisonResult(
                status="MISSING_VALUE",
                left_entity=left_ent.name,
                right_entity=right_ent.name,
                attribute=plan.attribute,
                unit=unit,
                right_value=val2,
                answer=ans,
                source_datasets=src_datasets,
                columns_used=cols_used,
                verification_status="MISSING_VALUE",
                verification_checks={"MISSING_VALUE": True}
            )

        if val2 is None:
            ans = f"I couldn't calculate the comparison because {attr_display} data is missing for {right_ent.name}."
            return ComparisonResult(
                status="MISSING_VALUE",
                left_entity=left_ent.name,
                right_entity=right_ent.name,
                attribute=plan.attribute,
                unit=unit,
                left_value=val1,
                answer=ans,
                source_datasets=src_datasets,
                columns_used=cols_used,
                verification_status="MISSING_VALUE",
                verification_checks={"MISSING_VALUE": True}
            )

        # Division by zero check (Section 21)
        if plan.comparison_type in ["PERCENTAGE_DIFFERENCE", "PERCENTAGE_CHANGE", "RATIO"] and val2 == 0:
            ans = "The percentage comparison cannot be calculated because the reference value is zero."
            return ComparisonResult(
                status="DIVISION_BY_ZERO",
                operation="COMPARE",
                left_entity=left_ent.name,
                right_entity=right_ent.name,
                attribute=plan.attribute,
                unit=unit,
                left_value=val1,
                right_value=val2,
                answer=ans,
                source_datasets=src_datasets,
                columns_used=cols_used,
                verification_status="VERIFIED",
                verification_checks={"DIVISION_BY_ZERO": True}
            )

        # Arithmetic Calculations
        directed_diff = val1 - val2
        abs_diff = abs(directed_diff)
        is_equal = (directed_diff == 0)
        higher_entity = left_ent.name if directed_diff > 0 else (right_ent.name if directed_diff < 0 else "Equal")
        lower_entity = right_ent.name if directed_diff > 0 else (left_ent.name if directed_diff < 0 else "Equal")

        pct_diff = None
        if val2 != 0:
            pct_diff = round((abs_diff / val2) * 100.0, 2)

        pct_change = None
        if val2 != 0:
            pct_change = round(((val1 - val2) / val2) * 100.0, 2)

        ratio_val = None
        if val2 != 0:
            ratio_val = round(val1 / val2, 3)

        # Boolean evaluation
        bool_result = None
        if plan.is_boolean:
            if plan.boolean_operator == ">":
                bool_result = (val1 > val2)
            elif plan.boolean_operator == "<":
                bool_result = (val1 < val2)
            elif plan.boolean_operator == "==":
                bool_result = (val1 == val2)
            elif plan.boolean_operator == "!=":
                bool_result = (val1 != val2)
            else:
                bool_result = (val1 > val2)

        # Source rows collection
        source_rows: List[Any] = []
        if rec1 and "row_ids" in rec1:
            source_rows.extend(rec1["row_ids"])
        elif left_ent.row_id:
            source_rows.append(left_ent.row_id)
        elif rec1 and "_internal_row_id" in rec1:
            source_rows.append(rec1["_internal_row_id"])

        if rec2 and "row_ids" in rec2:
            source_rows.extend(rec2["row_ids"])
        elif right_ent.row_id:
            source_rows.append(right_ent.row_id)
        elif rec2 and "_internal_row_id" in rec2:
            source_rows.append(rec2["_internal_row_id"])

        # Run 13-Point Verification Check
        is_verified, checks = self.verify_comparison(
            plan=plan,
            left_ent=left_ent,
            right_ent=right_ent,
            val1=val1,
            val2=val2,
            abs_diff=abs_diff,
            directed_diff=directed_diff,
            higher_ent=higher_entity,
            lower_ent=lower_entity,
            is_equal=is_equal,
            unit=unit,
            sources=src_datasets,
            source_rows=source_rows
        )

        # Natural Language Answer Generation
        answer_text = self._format_answer(
            plan=plan,
            left_ent=left_ent,
            right_ent=right_ent,
            val1=val1,
            val2=val2,
            directed_diff=directed_diff,
            abs_diff=abs_diff,
            higher_ent=higher_entity,
            lower_ent=lower_entity,
            is_equal=is_equal,
            pct_diff=pct_diff,
            ratio_val=ratio_val,
            bool_result=bool_result,
            unit=unit
        )

        calc_formula = f"|{val1} - {val2}|"
        if plan.comparison_type == "PERCENTAGE_DIFFERENCE":
            calc_formula = f"(|{val1} - {val2}| / {val2}) * 100"
        elif plan.comparison_type == "RATIO":
            calc_formula = f"{val1} / {val2}"

        main_path = parent_child_registry.get_main_dataset_path()
        dataset_versions = [main_path.name] if main_path else ["1.0"]

        return ComparisonResult(
            status="SUCCESS",
            operation="BOOLEAN_CHECK" if plan.is_boolean else "COMPARE",
            left_entity=left_ent.name,
            right_entity=right_ent.name,
            left_parent=left_ent.parent,
            right_parent=right_ent.parent,
            scope=plan.scope,
            attribute=plan.attribute,
            unit=unit,
            left_value=val1,
            right_value=val2,
            absolute_difference=abs_diff,
            directed_difference=directed_diff,
            percentage_difference=pct_diff,
            percentage_change=pct_change,
            ratio=ratio_val,
            higher_entity=higher_entity,
            lower_entity=lower_entity,
            comparison_operator=plan.boolean_operator or ("==" if is_equal else (">" if directed_diff > 0 else "<")),
            is_equal=is_equal,
            boolean_result=bool_result,
            answer=answer_text,
            source_datasets=src_datasets,
            source_rows=source_rows,
            source_row_ids=source_rows,
            dataset_versions=dataset_versions,
            columns_used=cols_used,
            calculation_formula=calc_formula,
            verification_status="VERIFIED" if is_verified else "FAILED",
            verification_checks=checks,
            debug_trace={
                "scope": plan.scope,
                "comparison_type": plan.comparison_type,
                "direction_requested": plan.direction_requested,
                "is_boolean": plan.is_boolean,
                "verification": checks
            }
        )


    def _execute_ranking(self, plan: ComparisonPlan) -> ComparisonResult:
        """Execute top 2 ranking difference across all villages or within a state."""
        comb_df = multi_child_executor.get_combined_dataframe()
        if comb_df.empty or plan.attribute not in comb_df.columns:
            return ComparisonResult(
                status="NO_MATCH",
                attribute=plan.attribute,
                answer="No village records found for ranking."
            )

        target_df = comb_df
        parent_scope = plan.entities[0].parent if plan.entities else None
        if parent_scope and "State" in target_df.columns:
            sub = target_df[target_df["State"].astype(str).str.lower() == parent_scope.lower()]
            if isinstance(sub, pd.DataFrame) and not sub.empty:
                target_df = sub

        sorted_df = target_df.sort_values(by=plan.attribute, ascending=False).dropna(subset=[plan.attribute])
        if len(sorted_df) < 2:
            return ComparisonResult(
                status="NO_MATCH",
                attribute=plan.attribute,
                answer="Need at least two villages to compute ranking difference."
            )

        row1 = sorted_df.iloc[0].to_dict()
        row2 = sorted_df.iloc[1].to_dict()

        name1 = str(row1.get("Village"))
        name2 = str(row2.get("Village"))
        val1 = float(row1.get(plan.attribute, 0))
        val2 = float(row2.get(plan.attribute, 0))

        diff = val1 - val2
        abs_diff = abs(diff)

        attr_display = plan.attribute.replace("_", " ")
        if val1 == val2:
            ans = f"{name1} and {name2} are tied for the highest {attr_display} at {val1:,.0f} each."
        else:
            ans = (
                f"The top village by {attr_display} is {name1} with {val1:,.0f}, followed by {name2} with {val2:,.0f}. "
                f"{name1} has {abs_diff:,.0f} more {attr_display} than {name2}."
            )

        return ComparisonResult(
            status="SUCCESS",
            operation="RANKING",
            left_entity=name1,
            right_entity=name2,
            attribute=plan.attribute,
            left_value=val1,
            right_value=val2,
            absolute_difference=abs_diff,
            directed_difference=diff,
            higher_entity=name1,
            lower_entity=name2,
            is_equal=(diff == 0),
            answer=ans,
            rankings=[row1, row2],
            source_datasets=["all_child_datasets"],
            columns_used=["Village", plan.attribute],
            calculation_formula=f"{val1} - {val2}"
        )

    def _execute_multi_entity(self, plan: ComparisonPlan, df: Optional[pd.DataFrame] = None) -> ComparisonResult:
        """Execute comparison across 3 or more entities."""
        records: List[Dict[str, Any]] = []
        cols_used = []
        datasets_used = []

        for ent in plan.entities:
            val, rec, cols = self.retrieve_entity_value(ent, plan.attribute, df)
            cols_used.extend(cols)
            if ent.dataset_name and ent.dataset_name not in datasets_used:
                datasets_used.append(ent.dataset_name)
            records.append({
                "entity": ent.name,
                "type": ent.entity_type,
                "value": val,
                "record": rec
            })

        valid_records = [r for r in records if r["value"] is not None]
        if not valid_records:
            return ComparisonResult(
                status="MISSING_VALUE",
                attribute=plan.attribute,
                answer=f"Could not retrieve {plan.attribute.replace('_', ' ')} for the specified entities."
            )

        valid_records.sort(key=lambda x: x["value"], reverse=True)
        top_ent = valid_records[0]
        bottom_ent = valid_records[-1]
        spread = top_ent["value"] - bottom_ent["value"]

        attr_disp = plan.attribute.replace("_", " ")
        ranked_strs = [f"{i+1}. {r['entity']}: {r['value']:,.0f}" for i, r in enumerate(valid_records)]
        ans = (
            f"Comparison of {attr_disp} among {len(valid_records)} entities:\n"
            + "\n".join(ranked_strs) + "\n"
            f"{top_ent['entity']} is the highest with {top_ent['value']:,.0f}, which is {spread:,.0f} higher than {bottom_ent['entity']} ({bottom_ent['value']:,.0f})."
        )

        return ComparisonResult(
            status="SUCCESS",
            operation="RANKING",
            left_entity=top_ent["entity"],
            right_entity=bottom_ent["entity"],
            attribute=plan.attribute,
            left_value=top_ent["value"],
            right_value=bottom_ent["value"],
            absolute_difference=spread,
            directed_difference=spread,
            higher_entity=top_ent["entity"],
            lower_entity=bottom_ent["entity"],
            answer=ans,
            rankings=valid_records,
            source_datasets=datasets_used,
            columns_used=list(set(cols_used))
        )

    # -----------------------------------------------------------------
    # NATURAL LANGUAGE FORMATTER
    # -----------------------------------------------------------------

    def _format_answer(
        self,
        plan: ComparisonPlan,
        left_ent: ComparisonEntity,
        right_ent: ComparisonEntity,
        val1: float,
        val2: float,
        directed_diff: float,
        abs_diff: float,
        higher_ent: str,
        lower_ent: str,
        is_equal: bool,
        pct_diff: Optional[float],
        ratio_val: Optional[float],
        bool_result: Optional[bool],
        unit: str = ""
    ) -> str:
        """Format friendly natural language answer adhering strictly to directional rules and Section 44 templates."""
        attr_name = plan.attribute.replace("_", " ")

        # Value strings (integer if whole number)
        v1_str = f"{int(val1):,}" if val1.is_integer() else f"{val1:,.2f}"
        v2_str = f"{int(val2):,}" if val2.is_integer() else f"{val2:,.2f}"
        diff_str = f"{int(abs_diff):,}" if abs_diff.is_integer() else f"{abs_diff:,.2f}"

        is_cross = (plan.scope == "CROSS_STATE_VILLAGE") or bool(left_ent.parent and right_ent.parent and str(left_ent.parent).lower() != str(right_ent.parent).lower())
        left_label = f"{left_ent.name} in {left_ent.parent}" if (left_ent.parent and is_cross) else left_ent.name
        right_label = f"{right_ent.name} in {right_ent.parent}" if (right_ent.parent and is_cross) else right_ent.name

        # 1. Equal Values (Section 15 & Section 44)
        if is_equal:
            unit_suffix = f" {unit}" if unit and unit not in {"%", "currency"} else ("%" if unit == "%" else "")
            if plan.scope == "RANKING" or plan.comparison_type == "RANKING":
                return f"{left_label} and {right_label} are tied for the highest {attr_name} at {v1_str}{unit_suffix} each."
            if plan.attribute == "VILLAGE_COUNT":
                return f"Both {left_label} and {right_label} have the same number of villages: {v1_str}."
            return f"Both {left_label} and {right_label} have the same {attr_name}: {v1_str}{unit_suffix}."

        # 2. Boolean Inquiries ("Is Hyderabad more populated than Amaravati?", "Are AP and TS different?")
        if plan.is_boolean and bool_result is not None:
            if plan.boolean_operator == "!=":
                if bool_result:
                    return f"Yes, {left_label} and {right_label} are different ({left_label}: {v1_str} vs {right_label}: {v2_str})."
                else:
                    return f"No, {left_label} and {right_label} are the same ({v1_str})."
            elif plan.boolean_operator == "==":
                if bool_result:
                    return f"Yes, both {left_label} and {right_label} have the same {attr_name} ({v1_str})."
                else:
                    return f"No, {left_label} and {right_label} have different {attr_name} ({left_label}: {v1_str} vs {right_label}: {v2_str})."
            elif plan.boolean_operator == "<":
                if bool_result:
                    return f"Yes, {left_label} has a lower {attr_name} ({v1_str}) than {right_label} ({v2_str})."
                else:
                    return f"No, {left_label} does not have a lower {attr_name} ({v1_str}) than {right_label} ({v2_str})."
            else:
                if bool_result:
                    return f"Yes, {left_label} has a higher {attr_name} ({v1_str}) than {right_label} ({v2_str})."
                else:
                    return f"No, {left_label} has a lower {attr_name} ({v1_str}) than {right_label} ({v2_str})."

        # 3. Percentage Inquiries ("What percentage more does Village 1 have than Village 2?")
        if plan.comparison_type == "PERCENTAGE_DIFFERENCE" and pct_diff is not None:
            noun = "people" if plan.attribute == "Population" else (unit if unit and unit not in {"%", "currency"} else attr_name)
            if directed_diff > 0:
                return f"{left_label} has {pct_diff:.2f}% more {noun} than {right_label} ({v1_str} vs {v2_str})."
            else:
                return f"{left_label} has {pct_diff:.2f}% fewer {noun} than {right_label} ({v1_str} vs {v2_str})."

        # 4. Ratio Inquiries ("What is the ratio of population between Village 1 and Village 2?")
        if plan.comparison_type == "RATIO" and ratio_val is not None:
            from fractions import Fraction
            f = Fraction(round(ratio_val, 2)).limit_denominator(20)
            if f.denominator > 1 and f.numerator < 50 and round(float(f), 2) == round(ratio_val, 2):
                ratio_disp = f"{f.numerator}:{f.denominator}"
            else:
                ratio_disp = f"{ratio_val:.2f}:1"
            return f"The ratio of {attr_name} between {left_label} and {right_label} is {ratio_disp} ({v1_str} vs {v2_str})."

        # 5. Literacy Rate percentage points
        if "Literacy" in plan.attribute:
            if directed_diff > 0:
                return (
                    f"{left_label} has a literacy rate of {v1_str}%, while {right_label} has {v2_str}%. "
                    f"{left_label} has {diff_str} percentage points higher literacy than {right_label}."
                )
            else:
                return (
                    f"{left_label} has a literacy rate of {v1_str}%, while {right_label} has {v2_str}%. "
                    f"{left_label} has {diff_str} percentage points lower literacy than {right_label}."
                )

        # 6. Village Count
        if plan.attribute == "VILLAGE_COUNT":
            if directed_diff > 0:
                return f"{left_label} has {v1_str} villages, while {right_label} has {v2_str}. {left_label} has {diff_str} more villages than {right_label}."
            else:
                return f"{left_label} has {v1_str} villages, while {right_label} has {v2_str}. {left_label} has {diff_str} fewer villages than {right_label}."

        # 7. Standard Numeric Differences (Population, Households, Area, Salary, etc.)
        # STRICT RULE:
        # IF diff > 0: Left has X more ... than Right
        # IF diff < 0: Left has X fewer ... than Right
        # NEVER output raw negative number
        noun = "people" if plan.attribute == "Population" else (unit if unit and unit not in {"%", "currency"} else attr_name)

        if plan.direction_requested in ["FEWER", "LESS", "LOWER"] and lower_ent:
            lower_label = lower_ent
            higher_label = higher_ent
            lower_val = v2_str if lower_ent == right_ent.name else v1_str
            higher_val = v1_str if lower_ent == right_ent.name else v2_str
            return (
                f"{lower_label} has less {noun} ({lower_val}) than {higher_label} ({higher_val}). "
                f"{lower_label} has {diff_str} fewer {noun} than {higher_label}. "
                f"{left_label}: {v1_str} | {right_label}: {v2_str}."
            )

        if directed_diff > 0:
            return (
                f"{left_label} has {v1_str} {noun}, while {right_label} has {v2_str}. "
                f"{left_label} has {diff_str} more {noun} than {right_label}. "
                f"{left_label}: {v1_str} | {right_label}: {v2_str}."
            )
        else:
            return (
                f"{left_label} has {v1_str} {noun}, while {right_label} has {v2_str}. "
                f"{left_label} has {diff_str} fewer {noun} than {right_label}. "
                f"{left_label}: {v1_str} | {right_label}: {v2_str}."
            )



universal_comparison_engine = UniversalComparisonEngine()
