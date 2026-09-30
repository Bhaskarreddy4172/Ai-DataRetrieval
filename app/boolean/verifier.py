"""Deterministic Boolean & Fact-Check Verifier for Dataset, General Knowledge, and Hybrid queries."""

import re
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from app.boolean.schema import BooleanAssertion, BooleanResult
from app.knowledge.general_knowledge import general_knowledge_engine
from app.utils.logger import logger


class BooleanVerifier:
    """Evaluates BooleanAssertions deterministically with row-level evidence and zero hallucinations."""

    def verify(self, assertion: BooleanAssertion, df: pd.DataFrame) -> BooleanResult:
        """Route and execute deterministic verification based on source_target."""
        if assertion.source_target == "GENERAL_KNOWLEDGE":
            res = self._verify_general_knowledge(assertion)
        elif assertion.source_target == "HYBRID":
            res = self._verify_hybrid(assertion, df)
        else:
            res = self._verify_dataset(assertion, df)

        if res.evidence:
            res.evidence = self._sanitize_numpy(res.evidence)
        if isinstance(res.actual_value, (np.generic, np.ndarray)):
            res.actual_value = res.actual_value.item() if hasattr(res.actual_value, "item") else str(res.actual_value)
        if isinstance(res.expected_value, (np.generic, np.ndarray)):
            res.expected_value = res.expected_value.item() if hasattr(res.expected_value, "item") else str(res.expected_value)
        return res

    def _sanitize_numpy(self, val: Any) -> Any:
        if isinstance(val, dict):
            return {k: self._sanitize_numpy(v) for k, v in val.items()}
        elif isinstance(val, (list, tuple)):
            return [self._sanitize_numpy(v) for v in val]
        elif isinstance(val, np.integer):
            return int(val)
        elif isinstance(val, np.floating):
            return float(val)
        elif isinstance(val, (np.bool_, bool)):
            return bool(val)
        elif isinstance(val, np.ndarray):
            return val.tolist()
        return val

    # -------------------------------------------------------------------------
    # GENERAL KNOWLEDGE VERIFICATION
    # -------------------------------------------------------------------------
    def _verify_general_knowledge(self, assertion: BooleanAssertion) -> BooleanResult:
        """Evaluate general geographical / national knowledge assertions."""
        # 1. State Capital Assertion
        if assertion.relationship == "CAPITAL_OF":
            cap_res = general_knowledge_engine.verify_state_capital(assertion.subject or "", assertion.object or "")
            if not cap_res.get("verified"):
                return BooleanResult(
                    intent="BOOLEAN_CHECK",
                    result=None,
                    status="CANNOT_VERIFY",
                    answer=f"I can't reliably verify whether {assertion.subject} is the capital of {assertion.object}.",
                    source="GENERAL_KNOWLEDGE",
                    evidence=cap_res
                )

            is_cap = cap_res["is_capital"]
            actual_cap = cap_res["actual_capital"]
            target = cap_res["target_entity"]
            city = cap_res["city"]
            if is_cap:
                ans = f"Yes (True). {city} is the capital of {target}."
            else:
                ans = f"No (False). {actual_cap} is the capital of {target}, not {city}."

            return BooleanResult(
                intent="BOOLEAN_CHECK",
                result=is_cap,
                status="VERIFIED",
                answer=ans,
                source="GENERAL_KNOWLEDGE",
                confidence=1.0,
                evidence=cap_res,
                subject=city,
                expected_value=actual_cap
            )

        # 2. City-State Membership Assertion
        mem_res = general_knowledge_engine.verify_city_in_state(assertion.subject or "", assertion.object or "")
        if not mem_res.get("verified"):
            return BooleanResult(
                intent="BOOLEAN_CHECK",
                result=None,
                status="CANNOT_VERIFY",
                answer=f"I can't reliably verify whether {assertion.subject} is in {assertion.object}.",
                source="GENERAL_KNOWLEDGE",
                evidence=mem_res
            )

        is_member = mem_res["is_member"]
        disp_city = mem_res["city_display"]
        act_state = mem_res["actual_state"]
        tgt_state = mem_res["target_state"]

        if is_member:
            ans = f"Yes (True). {disp_city} is in {act_state}, India."
        else:
            ans = f"No (False). {disp_city} is in {act_state}, not {tgt_state}."

        return BooleanResult(
            intent="BOOLEAN_CHECK",
            result=is_member,
            status="VERIFIED",
            answer=ans,
            source="GENERAL_KNOWLEDGE",
            confidence=1.0,
            evidence=mem_res,
            subject=disp_city,
            actual_value=act_state,
            expected_value=tgt_state
        )

    # -------------------------------------------------------------------------
    # HYBRID VERIFICATION
    # -------------------------------------------------------------------------
    def _verify_hybrid(self, assertion: BooleanAssertion, df: pd.DataFrame) -> BooleanResult:
        """Evaluate assertions combining dataset row retrieval with general knowledge."""
        subject = assertion.subject or ""
        target_state = assertion.object or ""

        # Find row for subject
        row = self._find_row_for_entity(subject, df)
        if row is None:
            return BooleanResult(
                intent="BOOLEAN_CHECK",
                result=False,
                status="VERIFIED",
                answer=f"No (False). '{subject}' was not found in the dataset.",
                source="HYBRID",
                confidence=0.9
            )

        city_col = self._find_column(df, ["City", "Location", "Place"])
        if not city_col or city_col not in row or pd.isna(row[city_col]):
            return BooleanResult(
                intent="BOOLEAN_CHECK",
                result=None,
                status="CANNOT_VERIFY",
                answer=f"I can't verify that because the city for {subject} is not available in the dataset.",
                source="HYBRID"
            )

        actual_city = str(row[city_col]).strip()
        gk_res = general_knowledge_engine.verify_city_in_state(actual_city, target_state)
        if not gk_res.get("verified"):
            return BooleanResult(
                intent="BOOLEAN_CHECK",
                result=None,
                status="CANNOT_VERIFY",
                answer=f"I can't verify whether {actual_city} is in {target_state}.",
                source="HYBRID",
                evidence={"dataset_city": actual_city, "knowledge": gk_res}
            )

        is_member = gk_res["is_member"]
        act_state = gk_res["actual_state"]

        if is_member:
            ans = f"Yes (True). {subject} works in {actual_city}, which is in {target_state}."
        else:
            ans = f"No (False). {subject} works in {actual_city}, which is in {act_state}, not {target_state}."

        return BooleanResult(
            intent="BOOLEAN_CHECK",
            result=is_member,
            status="VERIFIED",
            answer=ans,
            source="HYBRID",
            confidence=1.0,
            evidence={
                "row_id": int(row.get("_internal_row_id", 0)),
                "entity": subject,
                "city": actual_city,
                "knowledge_evidence": gk_res
            },
            subject=subject,
            actual_value=actual_city,
            expected_value=target_state
        )

    # -------------------------------------------------------------------------
    # DATASET VERIFICATION
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # DATASET VERIFICATION
    # -------------------------------------------------------------------------
    def _verify_dataset(self, assertion: BooleanAssertion, df: pd.DataFrame) -> BooleanResult:
        """Evaluate dataset assertions across equality, comparisons, existence, ranking, etc."""
        if df.empty:
            return BooleanResult(
                intent="BOOLEAN_CHECK",
                result=None,
                status="UNKNOWN",
                answer="I can't determine that from the uploaded dataset.",
                source="DATASET"
            )

        # 1. MISSING FIELD GUARD
        if assertion.assertion_type == "MISSING_FIELD":
            attr = assertion.attribute or "information"
            return BooleanResult(
                intent="BOOLEAN_CHECK",
                result=None,
                status="UNKNOWN",
                answer=f"I can't verify that because '{attr}' is not present in the uploaded dataset. (UNKNOWN)",
                source="DATASET",
                confidence=1.0,
                evidence={"missing_column": attr}
            )

        # 2. DUPLICATES
        if assertion.assertion_type == "DUPLICATE":
            col = self._resolve_target_col(assertion.attribute, df, default="Salary")
            if not col or col not in df.columns:
                return BooleanResult(
                    intent="BOOLEAN_CHECK",
                    result=None,
                    status="UNKNOWN",
                    answer="I can't determine that from the uploaded dataset. (UNKNOWN)",
                    source="DATASET"
                )
            has_dups = bool(df[col].duplicated().any())
            if has_dups:
                ans = f"Yes (True). There are multiple records with the same {col} in the uploaded dataset."
            else:
                ans = f"No (False). All records have unique {col} values in the uploaded dataset."
            return BooleanResult(
                intent="BOOLEAN_CHECK",
                result=has_dups,
                status="VERIFIED",
                answer=ans,
                source="DATASET",
                confidence=1.0,
                evidence={"column": col, "has_duplicates": has_dups}
            )

        # 3. EQUALITY / SAME_AS / DIFFERENT_FROM (e.g. AP and TS are same?, are AP and TS different?)
        if assertion.assertion_type == "EQUALITY" and assertion.relationship in ["SAME_AS", "DIFFERENT_FROM"]:
            s1 = str(assertion.subject or "").strip()
            s2 = str(assertion.object or "").strip()

            row1 = self._find_row_for_entity(s1, df)
            row2 = self._find_row_for_entity(s2, df)

            if row1 is None or row2 is None:
                return BooleanResult(
                    intent="BOOLEAN_CHECK",
                    result=None,
                    status="UNKNOWN",
                    answer="I can't determine that from the uploaded dataset. (UNKNOWN)",
                    source="DATASET",
                    confidence=1.0,
                    evidence={"missing_entity": s1 if row1 is None else s2}
                )

            canon1 = self._get_canonical_entity_name(row1, s1)
            canon2 = self._get_canonical_entity_name(row2, s2)
            is_same = (canon1.lower() == canon2.lower())

            if assertion.relationship == "DIFFERENT_FROM":
                is_true = not is_same
                if is_true:
                    ans = f"Yes (True). {canon1} and {canon2} are different according to the uploaded dataset."
                else:
                    ans = f"No (False). {canon1} and {canon2} are the same according to the uploaded dataset."
            else:
                is_true = is_same
                if is_true:
                    ans = f"Yes (True). {canon1} and {canon2} are the same according to the uploaded dataset."
                else:
                    ans = f"No (False). {canon1} and {canon2} are different according to the uploaded dataset."

            return BooleanResult(
                intent="BOOLEAN_CHECK",
                result=is_true,
                status="VERIFIED",
                answer=ans,
                source="DATASET",
                confidence=1.0,
                evidence={
                    "row_ids": [int(row1.get("_internal_row_id", 0)), int(row2.get("_internal_row_id", 0))],
                    "entity1": canon1,
                    "entity2": canon2
                },
                subject=s1,
                expected_value=s2
            )

        # 4. MEMBERSHIP / BELONGS_TO / LOCATED_IN
        if assertion.assertion_type == "MEMBERSHIP" or assertion.relationship in ["BELONGS_TO", "BELONGS_TO_STATE"]:
            s1 = str(assertion.subject or "").strip()
            s2 = str(assertion.object or "").strip()

            rows1 = self._find_rows_for_entity(s1, df, raw_question=assertion.raw_question)
            rows2 = self._find_rows_for_entity(s2, df, raw_question=assertion.raw_question)

            if not rows1 and not rows2:
                # Try GK directly
                gk_check = general_knowledge_engine.verify_city_in_state(s1, s2)
                if gk_check.get("verified"):
                    is_m = gk_check["is_member"]
                    act_st = gk_check.get("actual_state", "")
                    disp_c = gk_check.get("city_display", s1)
                    if is_m:
                        ans = f"Yes (True). {disp_c} is in {act_st}, India."
                    else:
                        ans = f"No (False). {disp_c} is in {act_st}, not {s2}."
                    return BooleanResult(
                        intent="BOOLEAN_CHECK",
                        result=is_m,
                        status="VERIFIED",
                        answer=ans,
                        source="GENERAL_KNOWLEDGE",
                        confidence=1.0,
                        evidence=gk_check
                    )

                return BooleanResult(
                    intent="BOOLEAN_CHECK",
                    result=None,
                    status="UNKNOWN",
                    answer="I can't determine that from the uploaded dataset. (UNKNOWN)",
                    source="DATASET",
                    confidence=1.0,
                    evidence={"missing_subject": s1, "missing_object": s2}
                )

            if rows1:
                s2_clean = s2.lower()
                is_member = False
                matched_row = None

                for r in rows1:
                    row_vals = [str(v).lower() for v in r.values if pd.notna(v)]
                    if any(s2_clean == v or s2_clean in v for v in row_vals):
                        is_member = True
                        matched_row = r
                        break

                if is_member and matched_row is not None:
                    canon_s1 = self._get_canonical_entity_name(matched_row, s1)
                    ans = f"Yes (True). {canon_s1} belongs to {s2} according to the uploaded dataset."
                    return BooleanResult(
                        intent="BOOLEAN_CHECK",
                        result=True,
                        status="VERIFIED",
                        answer=ans,
                        source="DATASET",
                        confidence=1.0,
                        evidence={"row_id": int(matched_row.get("_internal_row_id", 0)), "subject": canon_s1, "object": s2}
                    )

                # Check if row1 has actual state column
                st_col = self._find_column(df, ["State", "State Name"])
                if st_col and st_col in rows1[0] and pd.notna(rows1[0][st_col]):
                    act_st = str(rows1[0][st_col]).strip()
                    if act_st.lower() != s2_clean:
                        ans = f"No (False). According to the uploaded dataset, {s1} is in {act_st}, not {s2}."
                        return BooleanResult(
                            intent="BOOLEAN_CHECK",
                            result=False,
                            status="VERIFIED",
                            answer=ans,
                            source="DATASET",
                            confidence=1.0,
                            evidence={"row_id": int(rows1[0].get("_internal_row_id", 0)), "subject": s1, "actual_state": act_st, "target_state": s2}
                        )

                # Check state-capital / city-state cross row if both in df
                if rows2 and not is_member:
                    r1 = rows1[0]
                    r2 = rows2[0]
                    st_col_cross = self._find_column(df, ["State", "State Name", "City", "Location"])
                    if st_col_cross and st_col_cross in r1 and st_col_cross in r2:
                        v1 = str(r1[st_col_cross]).strip()
                        v2 = str(r2[st_col_cross]).strip()
                        if v1.lower() != v2.lower():
                            ans = f"No (False). In the uploaded dataset, {v1} and {v2} are separate entries."
                            return BooleanResult(
                                intent="BOOLEAN_CHECK",
                                result=False,
                                status="VERIFIED",
                                answer=ans,
                                source="DATASET",
                                confidence=1.0,
                                evidence={"row_ids": [int(r1.get("_internal_row_id", 0)), int(r2.get("_internal_row_id", 0))], "entity1": v1, "entity2": v2}
                            )

                gk_check = general_knowledge_engine.verify_city_in_state(s1, s2)
                if gk_check.get("verified"):
                    if gk_check["is_member"]:
                        act_st = gk_check["actual_state"]
                        ans = f"Yes (True). {s1} is in {act_st}."
                        return BooleanResult(
                            intent="BOOLEAN_CHECK",
                            result=True,
                            status="VERIFIED",
                            answer=ans,
                            source="GENERAL_KNOWLEDGE",
                            confidence=1.0,
                            evidence={"row_id": int(rows1[0].get("_internal_row_id", 0)), "subject": s1, "object": s2, "gk": gk_check}
                        )
                    else:
                        act_st = gk_check["actual_state"]
                        ans = f"No (False). According to general knowledge, {s1} is in {act_st}, not {s2}."
                        return BooleanResult(
                            intent="BOOLEAN_CHECK",
                            result=False,
                            status="VERIFIED",
                            answer=ans,
                            source="GENERAL_KNOWLEDGE",
                            confidence=1.0,
                            evidence={"row_id": int(rows1[0].get("_internal_row_id", 0)), "subject": s1, "object": s2, "gk": gk_check}
                        )

                ans = f"No (False). According to the uploaded dataset, {s1} does not belong to {s2}."
                return BooleanResult(
                    intent="BOOLEAN_CHECK",
                    result=False,
                    status="VERIFIED",
                    answer=ans,
                    source="DATASET",
                    confidence=1.0,
                    evidence={"row_id": int(rows1[0].get("_internal_row_id", 0)), "subject": s1, "object": s2}
                )

            return BooleanResult(
                intent="BOOLEAN_CHECK",
                result=None,
                status="UNKNOWN",
                answer="I can't determine that from the uploaded dataset.",
                source="DATASET",
                confidence=1.0,
                evidence={"missing_subject": s1}
            )

        # 5. CROSS_ROW COMPARISON (e.g. Same department or city)
        if assertion.assertion_type == "CROSS_ROW":
            s1 = assertion.subject or ""
            s2 = assertion.secondary_subject or ""
            col = self._resolve_target_col(assertion.attribute, df, default="Department")

            row1 = self._find_row_for_entity(s1, df)
            row2 = self._find_row_for_entity(s2, df)
            if row1 is None or row2 is None:
                return BooleanResult(
                    intent="BOOLEAN_CHECK",
                    result=None,
                    status="UNKNOWN",
                    answer="I can't determine that from the uploaded dataset.",
                    source="DATASET",
                    confidence=1.0
                )

            v1 = str(row1.get(col, "")).strip()
            v2 = str(row2.get(col, "")).strip()
            is_same = (v1.lower() == v2.lower())
            if is_same:
                ans = f"Yes (True). Both {s1} and {s2} are in {v1} according to the uploaded dataset."
            else:
                ans = f"No (False). {s1} is in {v1}, while {s2} is in {v2} according to the uploaded dataset."

            return BooleanResult(
                intent="BOOLEAN_CHECK",
                result=is_same,
                status="VERIFIED",
                answer=ans,
                source="DATASET",
                confidence=1.0,
                evidence={
                    "row_ids": [int(row1.get("_internal_row_id", 0)), int(row2.get("_internal_row_id", 0))],
                    "columns": [col],
                    f"{s1}_{col}": v1,
                    f"{s2}_{col}": v2
                }
            )

        # 6. ENTITY-TO-ENTITY COMPARISON (e.g. Does Rahul earn more than Priya?)
        if assertion.assertion_type in ["COMPARISON", "INEQUALITY"] and assertion.secondary_subject:
            s1 = assertion.subject or ""
            s2 = assertion.secondary_subject or ""
            col = self._resolve_target_col(assertion.attribute, df, default="Salary")

            rows1 = self._find_rows_for_entity(s1, df, raw_question=assertion.raw_question)
            rows2 = self._find_rows_for_entity(s2, df, raw_question=assertion.raw_question)

            row1 = rows1[0] if rows1 else None
            row2 = rows2[0] if rows2 else None

            if row1 is None or row2 is None:
                return BooleanResult(
                    intent="BOOLEAN_CHECK",
                    result=None,
                    status="UNKNOWN",
                    answer="I can't determine that from the uploaded dataset.",
                    source="DATASET",
                    confidence=1.0
                )

            # Date comparison
            if any(k in col.lower() for k in ["date", "joined"]):
                d2 = str(row2.get(col, "")).split()[0]
                op = assertion.comparison_op or "<"
                op_word = "before" if op in ["<", "<="] else "after"

                matching_r1 = None
                for r in rows1:
                    d1_cand = str(r.get(col, "")).split()[0]
                    cond = (d1_cand < d2) if op in ["<", "<="] else (d1_cand > d2)
                    if cond:
                        matching_r1 = r
                        break

                active_r1 = matching_r1 if matching_r1 is not None else row1
                d1 = str(active_r1.get(col, "")).split()[0]
                is_valid = (d1 < d2) if op in ["<", "<="] else (d1 > d2)
                canon1 = self._get_canonical_entity_name(active_r1, s1)
                canon2 = self._get_canonical_entity_name(row2, s2)

                if is_valid:
                    ans = f"Yes (True). {canon1} ({d1}) joined {op_word} {canon2} ({d2}) according to the uploaded dataset."
                else:
                    ans = f"No (False). {canon1} ({d1}) did not join {op_word} {canon2} ({d2}) according to the uploaded dataset."

                return BooleanResult(
                    intent="BOOLEAN_CHECK",
                    result=is_valid,
                    status="VERIFIED",
                    answer=ans,
                    source="DATASET",
                    confidence=1.0,
                    evidence={"row_ids": [int(active_r1.get("_internal_row_id", 0)), int(row2.get("_internal_row_id", 0))], "joining_dates": [d1, d2]}
                )

            # Numeric comparison
            num1 = self._to_num(row1.get(col))
            num2 = self._to_num(row2.get(col))
            if num1 is None or num2 is None:
                return BooleanResult(
                    intent="BOOLEAN_CHECK",
                    result=None,
                    status="UNKNOWN",
                    answer="I can't determine that from the uploaded dataset.",
                    source="DATASET",
                    confidence=1.0
                )

            op = assertion.comparison_op or ">"
            is_true = (num1 > num2) if op in [">", ">="] else (num1 < num2)
            fmt1 = self._format_currency(num1, col)
            fmt2 = self._format_currency(num2, col)
            op_label = "greater than" if op in [">", ">="] else "less than"

            if is_true:
                ans = f"Yes (True). {s1}'s {col} ({fmt1}) is {op_label} {s2}'s ({fmt2}) according to the uploaded dataset."
            else:
                ans = f"No (False). {s1}'s {col} ({fmt1}) is not {op_label} {s2}'s ({fmt2}) according to the uploaded dataset."

            return BooleanResult(
                intent="BOOLEAN_CHECK",
                result=is_true,
                status="VERIFIED",
                answer=ans,
                source="DATASET",
                confidence=1.0,
                evidence={
                    "row_ids": [int(row1.get("_internal_row_id", 0)), int(row2.get("_internal_row_id", 0))],
                    "columns": [col],
                    f"{s1}_{col}": num1,
                    f"{s2}_{col}": num2
                }
            )

        # 7. SUPERLATIVE COMPARISON (e.g. Is highest salary employee from same city as highest rated employee?)
        if assertion.assertion_type == "SUPERLATIVE_COMPARISON":
            sal_col = self._resolve_target_col("Salary", df)
            perf_col = self._resolve_target_col("Performance Score", df)
            city_col = self._find_column(df, ["City", "Location"])
            name_col = self._find_column(df, ["Employee Name", "Customer Name", "Name"])

            if not sal_col or not perf_col or not city_col or not name_col:
                return BooleanResult(
                    intent="BOOLEAN_CHECK",
                    result=None,
                    status="UNKNOWN",
                    answer="I can't determine that from the uploaded dataset.",
                    source="DATASET"
                )

            num_sal = pd.to_numeric(df[sal_col].astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce")
            num_perf = pd.to_numeric(df[perf_col].astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce")

            top_sal_row = df.loc[num_sal.idxmax()]
            top_perf_row = df.loc[num_perf.idxmax()]

            c1 = str(top_sal_row[city_col]).strip()
            c2 = str(top_perf_row[city_col]).strip()
            is_same = (c1.lower() == c2.lower())

            n1 = str(top_sal_row[name_col]).strip()
            n2 = str(top_perf_row[name_col]).strip()

            if is_same:
                ans = f"Yes (True). Both the highest-paid employee ({n1}) and the highest-rated employee ({n2}) are from {c1}."
            else:
                ans = f"No (False). The highest-paid employee ({n1}) is from {c1}, while the highest-rated employee ({n2}) is from {c2}."

            return BooleanResult(
                intent="BOOLEAN_CHECK",
                result=is_same,
                status="VERIFIED",
                answer=ans,
                source="DATASET",
                confidence=1.0,
                evidence={"top_sal_employee": n1, "top_sal_city": c1, "top_perf_employee": n2, "top_perf_city": c2}
            )

        # 8. RANKING ASSERTION (e.g. Is highest paid employee from Hyderabad?)
        if assertion.assertion_type == "RANKING":
            col = self._resolve_target_col(assertion.attribute, df, default="Salary")
            num_series = pd.to_numeric(df[col].astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce")
            max_val = num_series.max()
            top_rows = df[num_series == max_val]

            name_col = self._find_column(df, ["Employee Name", "Customer Name", "Name"])
            top_names = top_rows[name_col].tolist() if name_col else []

            if assertion.subject and assertion.subject != "MAX":
                is_highest = any(assertion.subject.lower() in str(n).lower() for n in top_names)
                fmt_max = self._format_currency(max_val, col)
                top_str = ", ".join(str(n) for n in top_names[:2])
                if is_highest:
                    ans = f"Yes (True). {assertion.subject} is the highest-paid employee with a {col} of {fmt_max}."
                else:
                    ans = f"No (False). {top_str} is the highest-paid employee with a {col} of {fmt_max}, not {assertion.subject}."
                return BooleanResult(
                    intent="BOOLEAN_CHECK",
                    result=is_highest,
                    status="VERIFIED",
                    answer=ans,
                    source="DATASET",
                    confidence=1.0,
                    evidence={"top_earners": top_names, "max_salary": max_val}
                )

            if assertion.object and assertion.relationship == "City":
                target_city = assertion.object
                city_col = self._find_column(df, ["City", "Location"])
                top_cities = top_rows[city_col].tolist() if city_col else []
                is_from_city = any(str(target_city).lower() in str(c).lower() for c in top_cities)

                matching_rows = top_rows[top_rows[city_col].astype(str).str.lower() == str(target_city).lower()] if city_col else pd.DataFrame()
                if not matching_rows.empty and name_col:
                    top_person = matching_rows[name_col].iloc[0]
                else:
                    top_person = top_names[0] if top_names else "The top earner"

                actual_city = top_cities[0] if top_cities else "another city"
                if is_from_city:
                    ans = f"Yes (True). The highest-paid employee ({top_person}) is from {target_city}."
                else:
                    ans = f"No (False). The highest-paid employee ({top_person}) is from {actual_city}, not {target_city}."

                return BooleanResult(
                    intent="BOOLEAN_CHECK",
                    result=is_from_city,
                    status="VERIFIED",
                    answer=ans,
                    source="DATASET",
                    confidence=1.0,
                    evidence={"top_person": top_person, "actual_city": actual_city, "target_city": target_city}
                )

        # 9. EXISTENCE / NON_EXISTENCE / MULTI_CONDITION
        if assertion.assertion_type in ["EXISTENCE", "NON_EXISTENCE"]:
            if assertion.assertion_type == "EXISTENCE" and not assertion.conditions and assertion.subject:
                subj = assertion.subject
                row = self._find_row_for_entity(subj, df)
                if row is not None:
                    return BooleanResult(
                        intent="BOOLEAN_CHECK",
                        result=True,
                        status="VERIFIED",
                        answer=f'Yes (True). "{subj}" is present in the uploaded dataset.',
                        source="DATASET",
                        confidence=1.0,
                        evidence={"entity": subj}
                    )
                else:
                    return BooleanResult(
                        intent="BOOLEAN_CHECK",
                        result=None,
                        status="UNKNOWN",
                        answer=f'I can\'t determine or verify that because "{subj}" was not found in the uploaded dataset. (UNKNOWN)',
                        source="DATASET",
                        confidence=1.0,
                        evidence={"missing_entity": subj}
                    )

            filtered_df = df.copy()
            cond_descs = []
            for cond in assertion.conditions:
                c_col = cond["column"]
                c_op = cond["operator"]
                c_val = cond["value"]
                if c_col in filtered_df.columns:
                    if c_op == "=":
                        filtered_df = filtered_df[filtered_df[c_col].astype(str).str.lower() == str(c_val).lower()]
                        cond_descs.append(f"{c_col} = {c_val}")
                    elif c_op in [">", ">="]:
                        num_s = pd.to_numeric(filtered_df[c_col].astype(str).str.replace(r"[₹$,]", "", regex=True), errors="coerce")
                        filtered_df = filtered_df[num_s > float(c_val)] if c_op == ">" else filtered_df[num_s >= float(c_val)]
                        cond_descs.append(f"{c_col} {c_op} {self._format_currency(c_val, c_col)}")

            match_count = len(filtered_df)
            desc_str = ", ".join(cond_descs)
            if assertion.assertion_type == "NON_EXISTENCE":
                is_none = (match_count == 0)
                if is_none:
                    ans = f"Yes (True). There are no records matching {desc_str} in the uploaded dataset."
                else:
                    ans = f"No (False). There are {match_count} record(s) matching {desc_str} in the uploaded dataset."
                return BooleanResult(
                    intent="BOOLEAN_CHECK",
                    result=is_none,
                    status="VERIFIED",
                    answer=ans,
                    source="DATASET",
                    confidence=1.0,
                    evidence={"match_count": match_count, "conditions": assertion.conditions}
                )
            else:
                has_any = (match_count > 0)
                if has_any:
                    ans = f"Yes (True). There are {match_count} record(s) matching {desc_str} in the uploaded dataset."
                else:
                    ans = f"No (False). There are no records matching {desc_str} in the uploaded dataset."
                return BooleanResult(
                    intent="BOOLEAN_CHECK",
                    result=has_any,
                    status="VERIFIED",
                    answer=ans,
                    source="DATASET",
                    confidence=1.0,
                    evidence={"match_count": match_count, "conditions": assertion.conditions}
                )

        # 10. SINGLE ENTITY INEQUALITY (NUMERIC OR DATE)
        if assertion.assertion_type == "INEQUALITY":
            subj = assertion.subject or ""
            col = self._resolve_target_col(assertion.attribute, df, default="Salary")
            row = self._find_row_for_entity(subj, df)
            if row is None:
                return BooleanResult(
                    intent="BOOLEAN_CHECK",
                    result=None,
                    status="UNKNOWN",
                    answer="I can't determine that from the uploaded dataset.",
                    source="DATASET"
                )

            val = row.get(col)
            # Date comparison
            if any(k in col.lower() for k in ["date", "joined"]):
                act_str = str(val).split()[0]
                tgt_str = str(assertion.object).split()[0]
                op = assertion.comparison_op or ">"
                is_valid = (act_str > tgt_str) if op == ">" else (act_str < tgt_str)
                op_word = "after" if op == ">" else "before"
                if is_valid:
                    ans = f"Yes (True). {subj} joined on {act_str}, which is {op_word} {tgt_str} according to the uploaded dataset."
                else:
                    ans = f"No (False). {subj} joined on {act_str}, which is not {op_word} {tgt_str} according to the uploaded dataset."
                return BooleanResult(
                    intent="BOOLEAN_CHECK",
                    result=is_valid,
                    status="VERIFIED",
                    answer=ans,
                    source="DATASET",
                    confidence=1.0,
                    evidence={"row_id": int(row.get("_internal_row_id", 0)), "joining_date": act_str}
                )

            # Numeric comparison
            act_num = self._to_num(val)
            tgt_num = self._to_num(assertion.object)
            if act_num is None or tgt_num is None:
                return BooleanResult(
                    intent="BOOLEAN_CHECK",
                    result=None,
                    status="UNKNOWN",
                    answer="I can't determine that from the uploaded dataset.",
                    source="DATASET"
                )

            op = assertion.comparison_op or ">"
            is_valid = (act_num > tgt_num) if op == ">" else (act_num >= tgt_num if op == ">=" else act_num < tgt_num)
            op_label = "greater than" if op in [">", ">="] else "less than"
            fmt_act = self._format_currency(act_num, col)
            fmt_tgt = self._format_currency(tgt_num, col)

            if is_valid:
                ans = f"Yes (True). {subj}'s {col} is {fmt_act}, which is {op_label} {fmt_tgt} according to the uploaded dataset."
            else:
                ans = f"No (False). {subj}'s {col} is {fmt_act}, which is not {op_label} {fmt_tgt} according to the uploaded dataset."

            return BooleanResult(
                intent="BOOLEAN_CHECK",
                result=is_valid,
                status="VERIFIED",
                answer=ans,
                source="DATASET",
                confidence=1.0,
                evidence={"row_id": int(row.get("_internal_row_id", 0)), "actual_value": act_num, "target_value": tgt_num},
                subject=subj,
                actual_value=act_num,
                expected_value=tgt_num
            )

        # 11. ENTITY EQUALITY (e.g. Is Rahul working in Hyderabad?)
        if assertion.assertion_type == "EQUALITY":
            subj = assertion.subject or ""
            col = self._resolve_target_col(assertion.attribute, df, default="City")
            row = self._find_row_for_entity(subj, df)
            if row is None:
                return BooleanResult(
                    intent="BOOLEAN_CHECK",
                    result=None,
                    status="UNKNOWN",
                    answer=f"I can't determine or verify that because '{subj}' was not found in the uploaded dataset. (UNKNOWN)",
                    source="DATASET",
                    confidence=1.0,
                    evidence={"missing_entity": subj}
                )

            actual_val = str(row.get(col, "")).strip()
            target_val = str(assertion.object or "").strip()
            is_match = (actual_val.lower() == target_val.lower() or target_val.lower() in actual_val.lower())

            if assertion.relationship == "CAPITAL_OF":
                if is_match:
                    ans = f"Yes. True. The capital of {subj} is {actual_val} according to the uploaded dataset."
                else:
                    ans = f"No. False. The capital of {subj} is {actual_val}, not {target_val} according to the uploaded dataset."
            else:
                verb = "works in" if col in ["City", "Department"] else "is"
                if is_match:
                    ans = f"Yes. True. {subj} {verb} {actual_val} according to the uploaded dataset."
                else:
                    ans = f"No. False. {subj} {verb} {actual_val}, not {target_val} according to the uploaded dataset."

            return BooleanResult(
                intent="BOOLEAN_CHECK",
                result=is_match,
                status="VERIFIED",
                answer=ans,
                source="DATASET",
                confidence=1.0,
                evidence={
                    "row_ids": [int(row.get("_internal_row_id", 0))],
                    "columns": [col],
                    "actual_value": actual_val,
                    "target_value": target_val
                },
                subject=subj,
                attribute=col,
                actual_value=actual_val,
                expected_value=target_val
            )

        # 12. Fallback Unknown
        return BooleanResult(
            intent="BOOLEAN_CHECK",
            result=None,
            status="UNKNOWN",
            answer="I can't determine that from the uploaded dataset.",
            source="DATASET"
        )

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------
    def _get_canonical_entity_name(self, row: pd.Series, raw_entity: str) -> str:
        """Extract primary name or state or city from matching row."""
        for col in ["Employee Name", "Customer Name", "Name", "State", "State Name", "City", "Location"]:
            if col in row and pd.notna(row[col]):
                return str(row[col]).strip()
        for col in row.index:
            if not str(col).startswith("_") and pd.notna(row[col]):
                return str(row[col]).strip()
        return raw_entity.title()

    def _resolve_entity_alias_simple(self, text: str) -> str:
        """Resolve state/city abbreviation or alias."""
        try:
            from app.dataset.alias_resolver import entity_alias_resolver
            res = entity_alias_resolver.resolve_alias(text)
            if res and res.get("canonical_name"):
                return res["canonical_name"]
        except Exception:
            pass
        return text

    def _find_rows_for_entity(self, entity: str, df: pd.DataFrame, raw_question: Optional[str] = None) -> List[pd.Series]:
        """Find all matching rows for an entity name or ID in the DataFrame."""
        if not entity or df.empty:
            return []

        ent_clean = entity.strip().lower()

        # Check ID columns
        id_cols = [c for c in df.columns if any(k in c.lower() for k in ["id", "code"]) or "emp_id" in c.lower() or "emp id" in c.lower()]
        for c in id_cols:
            matches = df[df[c].astype(str).str.lower() == ent_clean]
            if not matches.empty:
                return [row for _, row in matches.iterrows()]
            matches = df[df[c].astype(str).str.lower().str.contains(ent_clean, regex=False)]
            if not matches.empty:
                return [row for _, row in matches.iterrows()]

        # Check Name columns
        name_cols = [c for c in df.columns if any(k in c.lower() for k in ["name", "employee", "customer", "student"]) and c not in id_cols]
        for c in name_cols:
            parts = [w for w in ent_clean.split() if len(w) >= 3 and w not in ["the", "who", "all", "any", "are", "and"]]
            first_name = parts[0] if parts else ent_clean

            token_matches = df[df[c].astype(str).str.lower().str.contains(r"\b" + re.escape(first_name) + r"\b", regex=True)]
            if not token_matches.empty:
                return [row for _, row in token_matches.iterrows()]

            exact_matches = df[df[c].astype(str).str.lower() == ent_clean]
            if not exact_matches.empty:
                return [row for _, row in exact_matches.iterrows()]
            contains_matches = df[df[c].astype(str).str.lower().str.contains(re.escape(first_name), regex=True)]
            if not contains_matches.empty:
                return [row for _, row in contains_matches.iterrows()]

        # Check all other text/object columns (e.g. 'state', 'capital', 'city', 'department')
        other_cols = [c for c in df.columns if not c.startswith("_") and c not in id_cols and c not in name_cols]
        for c in other_cols:
            matches = df[df[c].astype(str).str.lower() == ent_clean]
            if not matches.empty:
                return [row for _, row in matches.iterrows()]

        return []

    def _find_row_for_entity(self, entity: str, df: pd.DataFrame) -> Optional[pd.Series]:
        """Find the matching row for an entity name or ID in the DataFrame."""
        rows = self._find_rows_for_entity(entity, df)
        return rows[0] if rows else None

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find matching column name from candidates."""
        cols_lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        return None

    def _resolve_target_col(self, attr: Optional[str], df: pd.DataFrame, default: str = "Salary") -> str:
        """Resolve attribute string to exact DataFrame column."""
        if not attr:
            return default
        attr_lower = attr.lower()
        for c in df.columns:
            if c.lower() == attr_lower:
                return c
        for c in df.columns:
            if attr_lower in c.lower() or c.lower() in attr_lower:
                return c

        # Dataset spell checker fallback for column typos
        try:
            from app.dataset.spell_checker import dataset_spell_checker
            spell_col = dataset_spell_checker.resolve_column(attr_lower, list(df.columns))
            if spell_col:
                return spell_col
        except Exception:
            pass

        return default

    def _to_num(self, val: Any) -> Optional[float]:
        """Safely convert value to float."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        clean = re.sub(r"[₹$,]", "", str(val)).strip()
        try:
            return float(clean)
        except ValueError:
            return None

    def _format_currency(self, val: Any, col: str) -> str:
        """Format number as currency if salary/amount, else standard number."""
        num = self._to_num(val)
        if num is None:
            return str(val)
        if any(k in col.lower() for k in ["salary", "amount", "cost", "revenue", "price", "pay"]):
            return f"₹{num:,.2f}"
        if num.is_integer():
            return str(int(num))
        return f"{num:.2f}"


boolean_verifier = BooleanVerifier()

