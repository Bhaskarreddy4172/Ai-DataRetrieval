"""Assertion Builder: translates natural language questions into structured BooleanAssertions."""

import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from app.boolean.schema import BooleanAssertion
from app.knowledge.general_knowledge import general_knowledge_engine, INDIAN_CITIES, INDIAN_STATES, WORLD_CAPITALS
from app.dataset.alias_resolver import entity_alias_resolver
from app.utils.fuzzy_match import expand_abbreviations, LOCATION_ABBREVIATIONS
from app.utils.normalization import normalize_question, parse_numeric_value, extract_numeric_filter, extract_date_filter
from app.utils.logger import logger


class BooleanAssertionBuilder:
    """Detects and parses natural language questions into structured Boolean assertions."""

    BOOLEAN_START_WORDS = [
        "is", "are", "does", "did", "do", "was", "were", "can", "could", "has", "have", "will", "would", "should"
    ]

    CONFIRMATION_TAGS = [
        "right", "na", "correct", "true or false", "or not", "is it", "isn't it"
    ]

    RELATIONSHIP_VERBS = [
        "belong to", "belongs to", "belong", "belongs", "comes under", "come under",
        "under", "comes in", "come in", "part of", "located in", "located at",
        "working in", "work in", "works in", "lives in", "live in", "from", "in"
    ]

    COMMON_NON_EXISTENT_ATTRS = [
        "age", "years old", "year old", "marital status", "married", "single",
        "gender", "sex", "religion", "caste", "nationality", "height", "weight",
        "blood group", "father", "mother", "spouse", "children", "kids",
        "population", "chief minister", "cm", "governor", "gdp", "area", "size",
        "weather", "temperature", "climate", "rainfall", "crime", "crime rate",
        "literacy", "literacy rate", "president", "prime minister", "pm", "mayor",
        "currency", "timezone", "time zone", "pin code", "pincode"
    ]

    def is_boolean_question(self, question: str) -> bool:
        """Determines whether a question is fact-checking / boolean / relationship / comparison."""
        if not question or not question.strip():
            return False

        q_clean = question.strip().lower()
        q_norm = normalize_question(q_clean)
        words = q_norm.split()
        if not words:
            return False

        # 0. Wh-questions asking for specific attribute lookups (e.g. "Bangalore belongs to which state?", "where does Chennai belong to?") are NOT boolean checks
        if any(attr_q in q_clean for attr_q in ["which state", "what state", "which city", "what city", "which department", "what department", "where does ", "where is ", "where do ", "where are "]):
            if not any(tag in q_clean for tag in ["true or false", "right?", "correct?", "is it"]):
                return False

        if any(q_norm.startswith(wh) for wh in ["who ", "which ", "what ", "where ", "when ", "why ", "how "]):
            if not any(tag in q_clean for tag in ["true or false", "right?", "correct?", "is it"]):
                if not any(pat in q_clean for pat in [
                    "belongs to", "belong to", "part of", "comes under", "come under",
                    "same as", "different from", "greater than", "less than", "higher than", "lower than", "same city", "same department"
                ]):
                    return False

        first_word = words[0].rstrip("?:,.")
        # 1. Starts with auxiliary verb or verification phrase
        if first_word in self.BOOLEAN_START_WORDS or re.match(r"^(?:check if|verify if|tell me if|whether|confirm if)\b", q_norm):
            return True

        # 2. Contains confirmation tags (right?, correct?, true or false?)
        if any(tag in q_clean for tag in self.CONFIRMATION_TAGS):
            return True

        # 3. Contains clear relationship or comparison phrase
        rel_keywords = [
            "belongs to", "belong to", "belongs", "belong", "part of", "comes under", "come under", "is under", "located in", "falls under",
            "same as", "same thing", "identical", "equal to", "different from", "not same", "separate", "distinct",
            "earns more", "salary greater", "salary higher", "greater than", "higher than", "more than", "less than", "lower than",
            "join before", "joined before", "join after", "joined after", "older than", "same city", "same dept", "same department"
        ]
        if any(kw in q_norm for kw in rel_keywords):
            return True

        # 4. Short equality or relationship assertion (e.g. "ap ts same", "hyd belongs ts", "up belongs uk", "rahul priya same city")
        if len(words) <= 5 and any(kw in q_norm for kw in ["same", "different", "equal", "belongs", "belong", "under"]):
            return True

        return False

    def build_assertion(
        self,
        question: str,
        available_columns: List[str],
        schema_info: Optional[Dict[str, Any]] = None,
        df: Optional[pd.DataFrame] = None
    ) -> Optional[BooleanAssertion]:
        """Convert natural language question into a validated BooleanAssertion."""
        q_norm = normalize_question(question)
        q_exp = expand_abbreviations(q_norm)

        # Universal Dataset-Aware Spell Checker & Typo Resolution
        try:
            from app.dataset.spell_checker import dataset_spell_checker
            correction_meta = dataset_spell_checker.correct_full_question(q_exp)
            if correction_meta and correction_meta.get("corrected_question"):
                q_exp = correction_meta["corrected_question"]
        except Exception:
            pass

        q_lower = q_exp.lower()
        cols_lower = [c.lower() for c in available_columns]

        # -------------------------------------------------------------
        # 1. Check for Missing Field Guard (Section 14 & 19)
        # e.g. "Is Rahul 25 years old?", "Is Ananya married?"
        # -------------------------------------------------------------
        entity_name = self._find_entity_in_text(q_lower, df)
        for missing_attr in self.COMMON_NON_EXISTENT_ATTRS:
            if re.search(r"\b" + re.escape(missing_attr) + r"\b", q_lower):
                if not any(missing_attr in c for c in cols_lower):
                    subject = entity_name or "the requested person"
                    clean_attr = "age" if missing_attr in ["years old", "year old"] else missing_attr
                    return BooleanAssertion(
                        intent="BOOLEAN_CHECK",
                        assertion_type="MISSING_FIELD",
                        source_target="DATASET",
                        subject=subject,
                        attribute=clean_attr,
                        raw_question=question,
                        explanation=f"Attribute '{clean_attr}' is not present in the dataset columns."
                    )

        # -------------------------------------------------------------
        # 1.5 Hybrid Boolean (Dataset Entity + General Knowledge State)
        # e.g. "Does Rahul work in a city that belongs to Telangana?", "Does Ananya work in a city that belongs to Telangana?"
        # -------------------------------------------------------------
        if entity_name:
            hybrid_patterns = [
                r"(?:work|works|working|live|lives|living)\s+in\s+a\s+city\s+(?:that\s+)?(?:belongs\s+to|is\s+in|comes\s+under|under)\s+([a-zA-Z\s]+?)(?:\?|$)",
                r"(?:from|in)\s+a\s+city\s+(?:that\s+)?(?:belongs\s+to|is\s+in|under)\s+([a-zA-Z\s]+?)(?:\?|$)",
                r"city\s+(?:of|for)\s+([a-zA-Z\s]+?)\s+(?:belongs\s+to|is\s+in)\s+([a-zA-Z\s]+?)(?:\?|$)"
            ]
            for hp in hybrid_patterns:
                m = re.search(hp, q_lower)
                if m:
                    target_state_raw = m.group(1).strip()
                    matched_state = general_knowledge_engine.find_state_match(target_state_raw)
                    if matched_state:
                        return BooleanAssertion(
                            intent="BOOLEAN_CHECK",
                            assertion_type="MEMBERSHIP",
                            source_target="HYBRID",
                            subject=entity_name,
                            relationship="WORKS_IN_CITY_OF_STATE",
                            object=matched_state,
                            attribute="City",
                            raw_question=question,
                            explanation=f"Verify if {entity_name}'s city belongs to {matched_state}."
                        )

        # -------------------------------------------------------------
        # 2. Belonging / Membership / Location Relationship Detection
        # e.g. "is up belongs to uk", "does up belong to uk", "up belongs uk?", "does hyd belong to ts?", "is Rahul in Engineering?"
        # -------------------------------------------------------------
        mem_verbs = [
            r"belongs?\s+to", r"belong\s+to", r"belongs?", r"belong", r"part\s+of",
            r"comes?\s+under", r"come\s+under", r"is\s+under", r"located\s+in", r"falls?\s+under", r"comes?\s+in"
        ]
        for verb_pat in mem_verbs:
            m = re.search(r"^(?:is|are|does|did|do|check\s+if)?\s*([a-zA-Z0-9\s]+?)\s+" + verb_pat + r"\s+([a-zA-Z0-9\s]+?)\s*\??$", q_lower)
            if m:
                subj_raw = m.group(1).strip()
                obj_raw = m.group(2).strip()

                subj_clean = re.sub(r"^(?:is|are|does|did|do|check\s+if|verify\s+if)\s+", "", subj_raw, flags=re.IGNORECASE).strip()
                obj_clean = re.sub(r"\?$", "", obj_raw).strip()

                subj_res = entity_alias_resolver.resolve_alias(subj_clean)
                subj_canon = subj_res["canonical_name"] if subj_res and subj_res.get("canonical_name") else subj_clean.title()

                obj_res = entity_alias_resolver.resolve_alias(obj_clean)
                obj_canon = obj_res["canonical_name"] if obj_res and obj_res.get("canonical_name") else obj_clean.title()

                return BooleanAssertion(
                    intent="BOOLEAN_CHECK",
                    assertion_type="MEMBERSHIP",
                    source_target="DATASET",
                    subject=subj_canon,
                    relationship="BELONGS_TO",
                    object=obj_canon,
                    raw_question=question,
                    explanation=f"Verify if {subj_canon} belongs to {obj_canon}."
                )

        # -------------------------------------------------------------
        # 3. Same / Different / Equality Relationship Detection
        # e.g. "ap and ts are same?", "ap ts same?", "are ap and ts same?", "is ap equal to ts?", "are ap and ts different?"
        # -------------------------------------------------------------
        if any(kw in q_lower for kw in ["same", "different", "equal", "identical", "distinct"]):
            words = [w for w in re.findall(r"\b[A-Za-z0-9]+\b", q_norm) if w not in ["are", "is", "do", "does", "and", "vs", "to", "same", "different", "equal", "identical", "distinct", "check", "verify"]]
            if len(words) == 2:
                res1 = entity_alias_resolver.resolve_alias(words[0])
                res2 = entity_alias_resolver.resolve_alias(words[1])
                if res1 and res2 and res1.get("canonical_name") and res2.get("canonical_name"):
                    subj_canon = res1["canonical_name"]
                    obj_canon = res2["canonical_name"]
                    is_diff = any(kw in q_lower for kw in ["different", "distinct", "separate"])
                    return BooleanAssertion(
                        intent="BOOLEAN_CHECK",
                        assertion_type="EQUALITY",
                        source_target="DATASET",
                        subject=subj_canon,
                        object=obj_canon,
                        relationship="DIFFERENT_FROM" if is_diff else "SAME_AS",
                        raw_question=question,
                        explanation=f"Verify if {subj_canon} and {obj_canon} are {'different' if is_diff else 'the same'}."
                    )

        eq_patterns = [
            r"^(?:are|is|do|does|check\s+if)?\s*([a-zA-Z0-9\s]+?)\s+(?:equal\s+to|equals|same\s+as)\s+([a-zA-Z0-9\s]+?)\s*\??$",
            r"^(?:are|is|do|does|check\s+if)?\s*([a-zA-Z0-9\s]+?)\s+(?:and|vs|equal\s+to|equals)\s+([a-zA-Z0-9\s]+?)\s+(?:are\s+)?(same|different|equal|identical|distinct|separate)\s*\??$",
            r"^(?:are|is)?\s*([a-zA-Z0-9\s]+?)\s+([a-zA-Z0-9\s]+?)\s+(same|different|equal|identical|distinct|separate)\s*\??$",
            r"^(?:are|is)\s+([a-zA-Z0-9\s]+?)\s+and\s+([a-zA-Z0-9\s]+?)\s+(same|different|equal|identical|distinct|separate)\s*\??$"
        ]
        for eq_pat in eq_patterns:
            m = re.search(eq_pat, q_lower)
            if m:
                subj_raw = m.group(1).strip()
                obj_raw = m.group(2).strip()
                rel_type = m.group(3).strip().lower() if m.lastindex >= 3 else "same"

                subj_clean = re.sub(r"^(?:are|is|does|did|do|check\s+if|verify\s+if)\s+", "", subj_raw, flags=re.IGNORECASE).strip()
                obj_clean = obj_raw.strip()

                subj_res = entity_alias_resolver.resolve_alias(subj_clean)
                subj_canon = subj_res["canonical_name"] if subj_res and subj_res.get("canonical_name") else subj_clean.title()

                obj_res = entity_alias_resolver.resolve_alias(obj_clean)
                obj_canon = obj_res["canonical_name"] if obj_res and obj_res.get("canonical_name") else obj_clean.title()

                if "city" in q_lower or "location" in q_lower:
                    attr = "City"
                elif "dept" in q_lower or "department" in q_lower:
                    attr = "Department"
                else:
                    attr = None

                if attr:
                    return BooleanAssertion(
                        intent="BOOLEAN_CHECK",
                        assertion_type="CROSS_ROW",
                        source_target="DATASET",
                        subject=subj_canon,
                        secondary_subject=obj_canon,
                        attribute=attr,
                        comparison_op="=",
                        relationship="SAME_LOCATION" if attr == "City" else "SAME_DEPARTMENT",
                        raw_question=question,
                        explanation=f"Verify if {subj_canon} and {obj_canon} share the same {attr}."
                    )
                else:
                    is_diff = rel_type in ["different", "distinct", "separate"]
                    return BooleanAssertion(
                        intent="BOOLEAN_CHECK",
                        assertion_type="EQUALITY",
                        source_target="DATASET",
                        subject=subj_canon,
                        object=obj_canon,
                        relationship="DIFFERENT_FROM" if is_diff else "SAME_AS",
                        raw_question=question,
                        explanation=f"Verify if {subj_canon} and {obj_canon} are {'different' if is_diff else 'the same'}."
                    )

        # -------------------------------------------------------------
        # 4. Superlative Cross-Row Comparison Detection
        # e.g. "is the employee with the highest salary from the same city as the employee with the highest performance score?"
        # -------------------------------------------------------------
        if ("highest salary" in q_lower or "top earner" in q_lower) and ("highest performance" in q_lower or "highest rating" in q_lower or "top rated" in q_lower or "highest score" in q_lower) and ("same city" in q_lower or "same location" in q_lower or "same department" in q_lower):
            attr = "Department" if "department" in q_lower else "City"
            return BooleanAssertion(
                intent="BOOLEAN_CHECK",
                assertion_type="SUPERLATIVE_COMPARISON",
                source_target="DATASET",
                subject="MAX_SALARY",
                secondary_subject="MAX_PERFORMANCE",
                attribute=attr,
                relationship="SAME_LOCATION" if attr == "City" else "SAME_DEPARTMENT",
                raw_question=question,
                explanation=f"Verify if highest salary employee and highest performance employee share the same {attr}."
            )

        # -------------------------------------------------------------
        # 6. State-Capital / General-Capital inquiry
        # e.g. "Is XYZ the capital of Telangana?", "Is Hyderabad the capital of Telangana?"
        # -------------------------------------------------------------
        cap_q_m = re.search(r"^(?:is|are|check\s+if)\s+([A-Za-z0-9\s]+?)\s+(?:the\s+)?capital\s+of\s+([A-Za-z0-9\s]+?)(?:\?|$)", q_lower)
        if not cap_q_m:
            cap_q_m = re.search(r"^(?:is|are|check\s+if)\s+([A-Za-z0-9\s]+?)'s\s+capital\s+([A-Za-z0-9\s]+?)(?:\?|$)", q_lower)
            if cap_q_m:
                cand_state_raw = cap_q_m.group(1).strip()
                cand_city_raw = cap_q_m.group(2).strip()
            else:
                cand_state_raw, cand_city_raw = None, None
        else:
            cand_city_raw = cap_q_m.group(1).strip()
            cand_state_raw = cap_q_m.group(2).strip()

        if cand_state_raw and cand_city_raw:
            st_res = entity_alias_resolver.resolve_alias(cand_state_raw, expected_type="STATE")
            canonical_state = st_res["canonical_name"] if st_res and st_res.get("canonical_name") else cand_state_raw.title()
            
            c_res = entity_alias_resolver.resolve_alias(cand_city_raw, expected_type="CITY")
            canonical_city = c_res["canonical_name"] if c_res and c_res.get("canonical_name") else cand_city_raw.title()

            is_dataset = any("state" in c.lower() or "capital" in c.lower() for c in available_columns)
            return BooleanAssertion(
                intent="BOOLEAN_CHECK",
                assertion_type="EQUALITY",
                source_target="DATASET" if is_dataset else "GENERAL_KNOWLEDGE",
                subject=canonical_state,
                attribute="capital",
                relationship="CAPITAL_OF",
                object=canonical_city,
                raw_question=question,
                explanation=f"Verify if {canonical_city} is the capital of {canonical_state}."
            )

        # Check for State existence in dataset or nation: "Is Wakanda a state in India?" / "Is Wakanda a state?"
        state_exist_m = re.search(r"^(?:is|are|check\s+if)\s+([A-Za-z0-9\s]+?)\s+(?:a\s+)?state\b", q_lower)
        if state_exist_m:
            cand_state = state_exist_m.group(1).strip().title()
            is_dataset = any("state" in c.lower() for c in available_columns)
            return BooleanAssertion(
                intent="BOOLEAN_CHECK",
                assertion_type="EXISTENCE",
                source_target="DATASET" if is_dataset else "GENERAL_KNOWLEDGE",
                subject=cand_state,
                attribute="state",
                raw_question=question,
                explanation=f"Verify if '{cand_state}' is a state in the dataset."
            )

        # -------------------------------------------------------------
        # 7. Check for City-State Membership Boolean
        # e.g. "Is hyd belongs to Andra Pradesh?", "Is hyd in Telangana?"
        # -------------------------------------------------------------
        city_match, state_match = self._extract_city_and_state(q_lower)
        if city_match and state_match:
            is_dataset = any("state" in c.lower() or "capital" in c.lower() or "city" in c.lower() for c in available_columns)
            if "capital" in q_lower:
                return BooleanAssertion(
                    intent="BOOLEAN_CHECK",
                    assertion_type="EQUALITY" if is_dataset else "MEMBERSHIP",
                    source_target="DATASET" if is_dataset else "GENERAL_KNOWLEDGE",
                    subject=state_match if is_dataset else city_match,
                    attribute="capital" if is_dataset else None,
                    relationship="CAPITAL_OF",
                    object=city_match if is_dataset else state_match,
                    raw_question=question,
                    explanation=f"Verify if {city_match} is the capital of {state_match}."
                )

            return BooleanAssertion(
                intent="BOOLEAN_CHECK",
                assertion_type="MEMBERSHIP",
                source_target="DATASET" if is_dataset else "GENERAL_KNOWLEDGE",
                subject=city_match,
                relationship="BELONGS_TO_STATE",
                object=state_match,
                raw_question=question,
                explanation=f"Verify if {city_match} belongs to {state_match}."
            )

        # -------------------------------------------------------------
        # 4. Check for Dataset Ranking / Superlative Boolean
        # e.g. "Is the highest paid employee from Hyderabad?"
        # e.g. "Is Rahul the highest-paid employee?"
        # -------------------------------------------------------------
        if any(kw in q_lower for kw in ["highest paid", "highest salary", "top earner", "maximum salary", "highest pay"]):
            target_city = self._find_city_in_text(q_lower)
            if target_city and (not entity_name or entity_name.lower() == target_city.lower() or " from " in q_lower or " in " in q_lower):
                return BooleanAssertion(
                    intent="BOOLEAN_CHECK",
                    assertion_type="RANKING",
                    source_target="DATASET",
                    subject="MAX",
                    attribute="Salary",
                    relationship="City",
                    object=target_city,
                    raw_question=question,
                    explanation=f"Verify if highest paid employee is from {target_city}."
                )
            if entity_name:
                return BooleanAssertion(
                    intent="BOOLEAN_CHECK",
                    assertion_type="RANKING",
                    source_target="DATASET",
                    subject=entity_name,
                    attribute="Salary",
                    relationship="MAX",
                    raw_question=question,
                    explanation=f"Verify if {entity_name} is the highest paid employee."
                )

        # -------------------------------------------------------------
        # 5. Check for Cross-Row / Entity Comparison Boolean
        # e.g. "Does Rahul earn more than Ananya?"
        # e.g. "Do Rahul and Ananya work in the same department?"
        # -------------------------------------------------------------
        two_entities = self._find_two_entities_in_text(q_lower, df)
        if two_entities:
            e1, e2 = two_entities
            if any(kw in q_lower for kw in ["same department", "same dept"]):
                return BooleanAssertion(
                    intent="BOOLEAN_CHECK",
                    assertion_type="CROSS_ROW",
                    source_target="DATASET",
                    subject=e1,
                    secondary_subject=e2,
                    attribute="Department",
                    comparison_op="=",
                    raw_question=question,
                    explanation=f"Verify if {e1} and {e2} share the same Department."
                )
            if any(kw in q_lower for kw in ["same city", "same location"]):
                return BooleanAssertion(
                    intent="BOOLEAN_CHECK",
                    assertion_type="CROSS_ROW",
                    source_target="DATASET",
                    subject=e1,
                    secondary_subject=e2,
                    attribute="City",
                    comparison_op="=",
                    raw_question=question,
                    explanation=f"Verify if {e1} and {e2} share the same City."
                )
            if any(kw in q_lower for kw in [
                "earn more than", "earning more than", "higher salary", "salary above", "salary greater",
                "higher than", "greater than", "more than", "salary higher", "salary greater", "above"
            ]):
                return BooleanAssertion(
                    intent="BOOLEAN_CHECK",
                    assertion_type="COMPARISON",
                    source_target="DATASET",
                    subject=e1,
                    secondary_subject=e2,
                    attribute="Salary",
                    comparison_op=">",
                    raw_question=question,
                    explanation=f"Verify if {e1} earns more than {e2}."
                )
            if any(kw in q_lower for kw in [
                "earn less than", "earning less than", "lower salary", "salary below", "salary lower",
                "lower than", "less than", "below"
            ]):
                return BooleanAssertion(
                    intent="BOOLEAN_CHECK",
                    assertion_type="COMPARISON",
                    source_target="DATASET",
                    subject=e1,
                    secondary_subject=e2,
                    attribute="Salary",
                    comparison_op="<",
                    raw_question=question,
                    explanation=f"Verify if {e1} earns less than {e2}."
                )
            if any(kw in q_lower for kw in ["join before", "joined before", "before"]):
                return BooleanAssertion(
                    intent="BOOLEAN_CHECK",
                    assertion_type="COMPARISON",
                    source_target="DATASET",
                    subject=e1,
                    secondary_subject=e2,
                    attribute="Joining Date",
                    comparison_op="<",
                    raw_question=question,
                    explanation=f"Verify if {e1} joined before {e2}."
                )
            if any(kw in q_lower for kw in ["join after", "joined after", "after"]):
                return BooleanAssertion(
                    intent="BOOLEAN_CHECK",
                    assertion_type="COMPARISON",
                    source_target="DATASET",
                    subject=e1,
                    secondary_subject=e2,
                    attribute="Joining Date",
                    comparison_op=">",
                    raw_question=question,
                    explanation=f"Verify if {e1} joined after {e2}."
                )

        # -------------------------------------------------------------
        # 6. Check for Duplicates Boolean
        # e.g. "Are there two employees with the same salary?"
        # -------------------------------------------------------------
        if any(kw in q_lower for kw in ["same salary", "duplicate salary", "two employees with the same salary"]):
            return BooleanAssertion(
                intent="BOOLEAN_CHECK",
                assertion_type="DUPLICATE",
                source_target="DATASET",
                attribute="Salary",
                raw_question=question,
                explanation="Verify if duplicate salaries exist in dataset."
            )

        # -------------------------------------------------------------
        # 7. Check for Existence / Non-existence / Multi-condition Boolean
        # e.g. "Are there employees from Hyderabad?"
        # e.g. "Is there anyone from Hyderabad earning above 10L?"
        # e.g. "Is there nobody from Hyderabad?"
        # -------------------------------------------------------------
        is_non_existence = any(kw in q_lower for kw in ["nobody", "no one", "none", "not any", "zero employees", "nobody from", "nobody in"])
        is_existence = any(kw in q_lower for kw in [
            "are there employees", "are there any", "is there anyone", "is there somebody", "does anyone",
            "do we have employees", "do we have anyone", "anyone from", "anyone working", "is there nobody",
            "is there no one", "any employees from", "any employees in", "any employee in"
        ]) or is_non_existence or bool(re.search(r"\b(?:is|are)\s+there\b", q_lower))

        if is_existence:
            conditions = []
            target_city = self._find_city_in_text(q_lower)
            if target_city:
                conditions.append({"column": "City", "operator": "=", "value": target_city})

            target_dept = self._find_department_in_text(q_lower, df)
            if target_dept:
                conditions.append({"column": "Department", "operator": "=", "value": target_dept})

            num_filter = extract_numeric_filter(q_exp)
            if num_filter:
                target_col = self._resolve_column_by_keyword(q_lower, available_columns, default="Salary")
                conditions.append({
                    "column": target_col,
                    "operator": num_filter.get("operator", ">"),
                    "value": num_filter.get("value")
                })

            if conditions:
                return BooleanAssertion(
                    intent="BOOLEAN_CHECK",
                    assertion_type="NON_EXISTENCE" if is_non_existence else "EXISTENCE",
                    source_target="DATASET",
                    conditions=conditions,
                    raw_question=question,
                    explanation=f"Verify existence of records matching {conditions}."
                )

        # -------------------------------------------------------------
        # 8. Check for Single Entity Numeric / Date Condition
        # e.g. "Is salary of Rahul greater than 10L?"
        # e.g. "Is Rahul earning more than 10L?"
        # e.g. "Did Rahul join after 2024?"
        # -------------------------------------------------------------
        if entity_name:
            num_filter = extract_numeric_filter(q_exp)
            if num_filter:
                target_col = self._resolve_column_by_keyword(q_lower, available_columns, default="Salary")
                return BooleanAssertion(
                    intent="BOOLEAN_CHECK",
                    assertion_type="INEQUALITY",
                    source_target="DATASET",
                    subject=entity_name,
                    attribute=target_col,
                    comparison_op=num_filter.get("operator", ">"),
                    object=num_filter.get("value"),
                    raw_question=question,
                    explanation=f"Verify if {entity_name}'s {target_col} {num_filter.get('operator')} {num_filter.get('value')}."
                )

            date_filter = extract_date_filter(q_exp)
            if date_filter:
                target_col = self._resolve_column_by_keyword(q_lower, available_columns, default="Joining Date")
                return BooleanAssertion(
                    intent="BOOLEAN_CHECK",
                    assertion_type="INEQUALITY",
                    source_target="DATASET",
                    subject=entity_name,
                    attribute=target_col,
                    comparison_op=date_filter.get("operator", ">"),
                    object=date_filter.get("value"),
                    raw_question=question,
                    explanation=f"Verify if {entity_name}'s {target_col} {date_filter.get('operator')} {date_filter.get('value')}."
                )

            # Check for Entity Attribute Equality:
            # e.g. "Is Rahul working in Hyderabad?"
            # e.g. "Does Ananya work in Engineering?"
            target_city = self._find_city_in_text(q_lower)
            if target_city:
                return BooleanAssertion(
                    intent="BOOLEAN_CHECK",
                    assertion_type="EQUALITY",
                    source_target="DATASET",
                    subject=entity_name,
                    attribute="City",
                    relationship="WORKS_IN",
                    object=target_city,
                    raw_question=question,
                    explanation=f"Verify if {entity_name} works in {target_city}."
                )

            target_dept = self._find_department_in_text(q_lower, df)
            if target_dept:
                return BooleanAssertion(
                    intent="BOOLEAN_CHECK",
                    assertion_type="EQUALITY",
                    source_target="DATASET",
                    subject=entity_name,
                    attribute="Department",
                    relationship="WORKS_IN",
                    object=target_dept,
                    raw_question=question,
                    explanation=f"Verify if {entity_name} works in {target_dept}."
                )

        # -------------------------------------------------------------
        # 9. Fallback: Entity-less City or Department existence check
        # e.g. "Any employees in Hyderabad?"
        # -------------------------------------------------------------
        target_city = self._find_city_in_text(q_lower)
        if target_city and any(col.lower() == "city" for col in available_columns):
            return BooleanAssertion(
                intent="BOOLEAN_CHECK",
                assertion_type="EXISTENCE",
                source_target="DATASET",
                conditions=[{"column": "City", "operator": "=", "value": target_city}],
                raw_question=question,
                explanation=f"Verify existence of employees in {target_city}."
            )

        return None

    def _extract_city_and_state(self, q_lower: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract canonical city and state if both appear in question."""
        found_city = None
        for alias, canon in general_knowledge_engine._city_alias_map.items():
            if re.search(r"\b" + re.escape(alias) + r"\b", q_lower):
                found_city = canon
                break

        words = re.findall(r"\b[A-Za-z0-9]+\b", q_lower)
        if not found_city:
            for w in words:
                c_res = entity_alias_resolver.resolve_alias(w, expected_type="CITY")
                if c_res and c_res.get("canonical_name"):
                    found_city = c_res["canonical_name"]
                    break

        found_state = None
        for alias, canon in general_knowledge_engine._state_map.items():
            if re.search(r"\b" + re.escape(alias) + r"\b", q_lower):
                found_state = canon
                break

        if not found_state:
            for w in words:
                s_res = entity_alias_resolver.resolve_alias(w, expected_type="STATE")
                if s_res and s_res.get("canonical_name"):
                    found_state = s_res["canonical_name"]
                    break

        return found_city, found_state

    def _find_city_in_text(self, text: str) -> Optional[str]:
        """Match city in text using knowledge base and alias resolver."""
        for alias, canon in general_knowledge_engine._city_alias_map.items():
            if re.search(r"\b" + re.escape(alias) + r"\b", text):
                return canon

        for w in re.findall(r"\b[A-Za-z0-9]+\b", text):
            c_res = entity_alias_resolver.resolve_alias(w, expected_type="CITY")
            if c_res and c_res.get("canonical_name"):
                return c_res["canonical_name"]

        return None

    def _find_entity_in_text(self, text: str, df: Optional[pd.DataFrame] = None) -> Optional[str]:
        """Find employee or customer name or ID in question text."""
        # 1. Match ID pattern: EMP-1033, 1005, etc.
        id_m = re.search(r"\b(?:emp[-_]?(\d+)|employee\s+(\d+))\b", text, re.IGNORECASE)
        if id_m:
            num = id_m.group(1) or id_m.group(2)
            if df is not None and "Employee ID" in df.columns:
                for val in df["Employee ID"].dropna().unique():
                    if str(num) in str(val):
                        return str(val)
            return f"EMP-{num}"

        if df is not None:
            # Check name columns
            name_cols = [c for c in df.columns if any(k in c.lower() for k in ["name", "employee", "customer"])]
            for col in name_cols:
                unique_names = df[col].dropna().unique()
                for full_name in unique_names:
                    fn_str = str(full_name).strip()
                    # Check full name
                    if re.search(r"\b" + re.escape(fn_str.lower()) + r"\b", text):
                        return fn_str
                    # Check first name (if >= 3 letters and not common word)
                    parts = fn_str.split()
                    if parts:
                        first_name = parts[0]
                        if len(first_name) >= 3 and first_name.lower() not in ["the", "who", "all", "any", "are", "and"]:
                            if re.search(r"\b" + re.escape(first_name.lower()) + r"\b", text):
                                return fn_str

        # Check other entity columns in df (e.g. 'state', 'capital') only when no name columns exist
        if df is not None:
            name_cols = [c for c in df.columns if any(k in c.lower() for k in ["name", "employee", "customer"])]
            if not name_cols:
                other_cols = [c for c in df.columns if not c.startswith("_") and c not in name_cols]
                for col in other_cols:
                    for val in df[col].dropna().unique():
                        val_str = str(val).strip()
                        if len(val_str) >= 3 and val_str.lower() not in ["the", "who", "all", "any", "are", "and"]:
                            if re.search(r"\b" + re.escape(val_str.lower()) + r"\b", text):
                                return val_str

        # Static fallback entity names for standard samples
        common_names = ["Rahul", "Ananya", "Arjun", "Priya", "Tanvi", "Sanjay", "Sneha", "Kavita", "Vikram", "Ravi"]
        for cn in common_names:
            if re.search(r"\b" + re.escape(cn.lower()) + r"\b", text):
                return cn

        # Spell checker fallback for misspelled entities
        try:
            from app.dataset.spell_checker import dataset_spell_checker
            for word in text.split():
                if len(word) >= 3 and word.lower() not in ["the", "who", "all", "any", "are", "and", "does", "check", "is"]:
                    cand = dataset_spell_checker.resolve_candidate(word)
                    if cand and cand.confidence >= 0.75:
                        return cand.canonical_value
        except Exception:
            pass

        return None

    def _find_two_entities_in_text(self, text: str, df: Optional[pd.DataFrame] = None) -> Optional[Tuple[str, str]]:
        """Find two entity names in question for comparison, preserving question order."""
        found_with_pos: List[Tuple[int, str]] = []
        if df is not None:
            name_cols = [c for c in df.columns if any(k in c.lower() for k in ["name", "employee"])]
            for col in name_cols:
                for full_name in df[col].dropna().unique():
                    fn_str = str(full_name).strip()
                    parts = fn_str.split()
                    first_name = parts[0] if parts else fn_str
                    m = re.search(r"\b" + re.escape(first_name.lower()) + r"\b", text)
                    if m:
                        if not any(fn_str == item[1] or first_name == item[1].split()[0] for item in found_with_pos):
                            found_with_pos.append((m.start(), fn_str))

        common_names = ["Rahul", "Ananya", "Arjun", "Priya", "Tanvi", "Sanjay", "Sneha", "Kavita", "Vikram", "Ravi"]
        for cn in common_names:
            m = re.search(r"\b" + re.escape(cn.lower()) + r"\b", text)
            if m and not any(cn == item[1] or cn == item[1].split()[0] for item in found_with_pos):
                found_with_pos.append((m.start(), cn))

        if len(found_with_pos) >= 2:
            found_with_pos.sort(key=lambda x: x[0])
            return found_with_pos[0][1], found_with_pos[1][1]

        return None

    def _find_department_in_text(self, text: str, df: Optional[pd.DataFrame] = None) -> Optional[str]:
        """Find department name in question."""
        if df is not None and "Department" in df.columns:
            for dept in df["Department"].dropna().unique():
                if re.search(r"\b" + re.escape(str(dept).lower()) + r"\b", text):
                    return str(dept)

        dept_keywords = {
            "engineering": "Engineering",
            "eng": "Engineering",
            "sales": "Sales",
            "marketing": "Marketing",
            "finance": "Finance",
            "hr": "Human Resources",
            "human resources": "Human Resources",
            "support": "Customer Support",
            "operations": "Operations"
        }
        for kw, canonical in dept_keywords.items():
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                return canonical
        return None

    def _resolve_column_by_keyword(self, text: str, available_columns: List[str], default: str = "Salary") -> str:
        """Resolve metric column name from text keywords."""
        col_map = {c.lower(): c for c in available_columns}
        if any(w in text for w in ["salary", "earn", "paid", "pay", "money", "cash", "compensation"]):
            return col_map.get("salary", default)
        if any(w in text for w in ["performance", "score", "rating"]):
            return col_map.get("performance score", "Performance Score")
        if any(w in text for w in ["date", "joined", "joining"]):
            return col_map.get("joining date", "Joining Date")
        if any(w in text for w in ["amount", "bill", "cost"]):
            return col_map.get("service amount", default)
        return default


boolean_assertion_builder = BooleanAssertionBuilder()
