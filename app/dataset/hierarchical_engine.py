"""Hierarchical Parent-Child Query Engine.
Intelligently handles:
1. MAIN_TO_CHILD navigation (State/Capital -> Child dataset query).
2. CHILD_ONLY direct village queries (Village Name/ID -> specific Child dataset).
3. MULTI_CHILD_COMPARISON (Cross-child aggregations, ranking, and comparisons across all state capitals).
4. Multi-turn follow-ups across parent and child scopes.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from app.dataset.registry import parent_child_registry
from app.dataset.multi_child_executor import multi_child_executor
from app.conversation.context import conversation_manager
from app.utils.fuzzy_match import expand_abbreviations, LOCATION_ABBREVIATIONS
from app.utils.normalization import normalize_question
from app.utils.logger import logger


class HierarchicalQueryEngine:
    """Engine executing hierarchical queries across main and child datasets."""

    METRIC_ALIASES: Dict[str, str] = {
        "population": "Population",
        "pop": "Population",
        "people": "Population",
        "inhabitants": "Population",
        "residents": "Population",
        "males": "No_of_Males",
        "male": "No_of_Males",
        "male population": "No_of_Males",
        "men": "No_of_Males",
        "females": "No_of_Females",
        "female": "No_of_Females",
        "female population": "No_of_Females",
        "women": "No_of_Females",
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
    }

    def can_handle(self, question: str, session_id: Optional[str] = None) -> bool:
        """Determine if question requires child dataset access or multi-child aggregation."""
        if not parent_child_registry.is_parent_child_active():
            return False

        q_lower = question.lower()
        norm_q = normalize_question(question)
        expanded_q = expand_abbreviations(norm_q)
        exp_lower = expanded_q.lower()

        # 0. Check Universal Global Aggregation Engine
        from app.query.global_aggregation_engine import universal_global_aggregation_engine
        if universal_global_aggregation_engine.can_handle(question, session_id) or universal_global_aggregation_engine.can_handle(expanded_q, session_id):
            return True

        # 0b. Check Universal Comparison Engine for hierarchical entities
        from app.query.comparison_engine import universal_comparison_engine
        if universal_comparison_engine.is_comparison_query(question) or universal_comparison_engine.is_comparison_query(expanded_q):
            plan = universal_comparison_engine.parse_and_plan(expanded_q, session_id=session_id) or universal_comparison_engine.parse_and_plan(question, session_id=session_id)
            if plan is not None and plan.scope != "GENERIC_DATASET":
                return True

        # 1. Check village keywords
        if parent_child_registry.is_village_level_query(question) or parent_child_registry.is_village_level_query(expanded_q):
            return True

        # 2. Check if question has a village ID (e.g. TG-001, KA-002)
        if re.search(r"\b[a-zA-Z]{2}-\d{3}\b", question):
            return True

        # 3. Check if question has a village name pattern
        if re.search(r"\b(?:[a-zA-Z]+_)?village_\d+\b", q_lower):
            return True

        # 4. Check if question refers to previous village in session
        if session_id:
            last_village = conversation_manager.get_last_village(session_id)
            if last_village and re.search(r"\b(?:it|its|that village|the village|this village)\b", q_lower):
                return True

        # 5. Cross-state comparison and all-states ranking keywords
        cross_keywords = [
            "across all villages", "across all states", "across all capitals",
            "which state has the highest village", "which state has the lowest village",
            "which capital has the highest village", "which capital has the lowest village",
            "which state has the village with", "which capital has the village with",
            "which village across", "highest village population", "lowest village population",
            "highest village", "lowest village", "highest total village population",
            "lowest total village population", "highest total population", "lowest total population",
            "which state has the highest total", "which state has the lowest total",
            "which state has the highest", "which state has the lowest",
            "which state has the most", "which state has the fewest", "which state has the least",
            "top 10 villages", "top 5 villages", "villages across all states"
        ]
        if any(kw in q_lower or kw in exp_lower for kw in cross_keywords):
            return True

        # Check two states comparison
        states_found = [s for s in parent_child_registry.get_registered_child_datasets().keys() if re.search(r"\b" + re.escape(s.lower()) + r"\b", q_lower)]
        if len(states_found) >= 2 and any(w in q_lower for w in ["compare", "vs", "versus"]):
            return True

        # 6. Child metric query with registered entity / code
        metric_col = self._extract_metric_column(q_lower) or self._extract_metric_column(exp_lower)
        if metric_col:
            # If cross-state or all-state ranking query
            if any(w in q_lower or w in exp_lower for w in ["which state", "which capital", "across all", "top", "bottom"]):
                return True

            resolved_child = parent_child_registry.resolve_child_dataset(question) or parent_child_registry.resolve_child_dataset(expanded_q)
            if resolved_child:
                return True

        return False

    def classify_scope(self, question: str, session_id: Optional[str] = None) -> Tuple[str, Optional[str], Optional[Path]]:
        """Classify query into COMPARISON, MULTI_CHILD_COMPARISON, MAIN_TO_CHILD, or CHILD_ONLY."""
        q_lower = question.lower()
        norm_q = normalize_question(question)
        exp_q = expand_abbreviations(norm_q)
        exp_lower = exp_q.lower()

        # 0. Check for comparative inquiry across entities first
        from app.query.comparison_engine import universal_comparison_engine
        comp_plan = universal_comparison_engine.parse_and_plan(question, session_id=session_id)
        if not comp_plan:
            comp_plan = universal_comparison_engine.parse_and_plan(exp_q, session_id=session_id)
        if comp_plan and comp_plan.scope != "GENERIC_DATASET":
            return "COMPARISON", None, None

        # 1. Check for explicit village code e.g. "TG-001" -> prefix "TG"
        m_code = re.search(r"\b([a-zA-Z]{2})-\d{3}\b", question)
        if m_code:
            code_prefix = m_code.group(1)
            resolved_child = parent_child_registry.resolve_child_dataset(code_prefix)
            if resolved_child:
                entity_name, child_path = resolved_child
                return "CHILD_ONLY", entity_name, child_path
            else:
                return "CHILD_ONLY", code_prefix, None

        # 2. Check for explicit village name e.g. "Hyderabad_Village_01", "Atlantis_Village_99"
        m_v = re.search(r"\b([a-zA-Z]+)_village_\d+\b", q_lower)
        if m_v:
            prefix = m_v.group(1)
            resolved_child = parent_child_registry.resolve_child_dataset(prefix)
            if resolved_child:
                entity_name, child_path = resolved_child
                return "CHILD_ONLY", entity_name, child_path
            else:
                return "CHILD_ONLY", prefix, None

        # 3. Check Universal Global Aggregation Engine
        from app.query.global_aggregation_engine import universal_global_aggregation_engine
        from app.query.fast_classifier import fast_query_classifier

        if universal_global_aggregation_engine.can_handle(question, session_id) or universal_global_aggregation_engine.can_handle(exp_q, session_id):
            return "GLOBAL_AGGREGATION", None, None

        if fast_query_classifier.is_global_query(question, session_id=session_id) or fast_query_classifier.is_global_query(exp_q, session_id=session_id):
            return "GLOBAL_AGGREGATION", None, None

        # 4. Check for State/Capital to Child navigation
        resolved_child = parent_child_registry.resolve_child_dataset(question) or parent_child_registry.resolve_child_dataset(exp_q)
        if resolved_child:
            entity_name, child_path = resolved_child
            return "MAIN_TO_CHILD", entity_name, child_path

        # 5. Check for active child dataset in session ONLY if genuine contextual reference
        if session_id and not fast_query_classifier.is_global_query(question):
            is_contextual = any(re.search(r"\b" + re.escape(p) + r"\b", q_lower) for p in [
                "it", "its", "they", "them", "this state", "that state", "the state",
                "this village", "that village", "the village", "this capital", "that capital",
                "here", "there"
            ]) or any(q_lower.startswith(w) for w in ["what about", "how about", "and for", "and in"])

            if is_contextual:
                active_child_fname = conversation_manager.get_active_child_dataset(session_id)
                active_state = conversation_manager.get_active_state(session_id)
                if active_child_fname:
                    child_dir = parent_child_registry.get_child_datasets_dir()
                    if child_dir:
                        child_path = child_dir / active_child_fname
                        if child_path.exists():
                            return "CHILD_ONLY", active_state or active_child_fname, child_path

        # Default fallback to MULTI_CHILD_COMPARISON if no single state/child identified
        return "MULTI_CHILD_COMPARISON", None, None

    def execute(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute hierarchical query deterministically and return result dictionary."""
        scope, entity_name, child_path = self.classify_scope(question, session_id)
        norm_q = normalize_question(question)
        q_lower = norm_q.lower()

        if scope == "COMPARISON":
            from app.query.comparison_engine import universal_comparison_engine
            plan = universal_comparison_engine.parse_and_plan(question, session_id=session_id)
            if not plan:
                plan = universal_comparison_engine.parse_and_plan(expand_abbreviations(norm_q), session_id=session_id)
            if plan:
                res = universal_comparison_engine.execute(plan)
                if plan.scope in ["STATE_VS_STATE", "CROSS_STATE_VILLAGE", "MULTI_ENTITY", "STATE_VS_CAPITAL", "CAPITAL_VS_STATE"]:
                    h_scope = "MULTI_CHILD_COMPARISON"
                elif plan.scope == "SAME_STATE_VILLAGE":
                    h_scope = "CHILD_ONLY"
                else:
                    h_scope = "MULTI_CHILD_COMPARISON" if len(res.source_datasets) > 1 else plan.scope

                res_dict: Dict[str, Any] = {
                    "operation": res.operation,
                    "scope": h_scope,
                    "comparison_scope": plan.scope,
                    "results": [{
                        "left_entity": res.left_entity,
                        "left_value": res.left_value,
                        "left_parent": plan.entities[0].parent if len(plan.entities) > 0 else None,
                        "right_entity": res.right_entity,
                        "right_value": res.right_value,
                        "right_parent": plan.entities[1].parent if len(plan.entities) > 1 else None,
                        "scope": plan.scope,
                        "absolute_difference": res.absolute_difference,
                        "directed_difference": res.directed_difference,
                        "higher_entity": res.higher_entity,
                        "lower_entity": res.lower_entity,
                        "attribute": res.attribute,
                        "unit": res.unit,
                        "is_equal": res.is_equal,
                        "boolean_result": res.boolean_result,
                    }] if res.left_entity else (res.rankings or []),
                    "result_count": 1 if res.left_entity else len(res.rankings or []),
                    "answer": res.answer,
                    "child_dataset": res.source_datasets[0] if res.source_datasets else "all_child_datasets",
                    "columns_used": res.columns_used,
                    "calculation_formula": res.calculation_formula,
                    "comparison_details": res.to_dict(),
                    "unit": res.unit,
                    "source_rows": res.source_rows,
                    "dataset_versions": res.dataset_versions,
                    "verification_status": res.verification_status,
                    "verification_checks": res.verification_checks,
                }
                return res_dict

        if scope == "GLOBAL_AGGREGATION":
            from app.query.global_aggregation_engine import universal_global_aggregation_engine
            return universal_global_aggregation_engine.execute(question, session_id)

        if scope == "MULTI_CHILD_COMPARISON":
            return self._execute_multi_child(question, q_lower)
        else:
            return self._execute_single_child(question, q_lower, scope, entity_name, child_path, session_id)

    def _execute_multi_child(self, question: str, q_lower: str) -> Dict[str, Any]:
        """Execute cross-state or all-India child dataset aggregation or comparison."""
        # Detect metric
        metric_col = self._extract_metric_column(q_lower) or "Population"
        is_min = any(w in q_lower for w in ["lowest", "minimum", "min", "smallest", "least", "bottom", "fewest"])

        # 0. Check Top-N / Bottom-N villages across all states
        m_top = re.search(r"\b(?:top|bottom)\s+(\d+)\s+villages\b", q_lower)
        if m_top:
            n_val = int(m_top.group(1))
            top_villages = multi_child_executor.get_top_n_villages(metric_col, n=n_val, is_max=(not is_min))
            ext_label = "Bottom" if is_min else "Top"
            ans = f"{ext_label} {len(top_villages)} villages by {metric_col.replace('_', ' ')} across all registered states:\n"
            ans += "\n".join([f"{v['rank']}. {v['village']} ({v['state']}): {v['value']:,.0f}" for v in top_villages])
            return {
                "operation": "TOP_N" if not is_min else "BOTTOM_N",
                "scope": "MULTI_CHILD_COMPARISON",
                "results": top_villages,
                "result_count": len(top_villages),
                "answer": ans,
                "child_dataset": "all_child_datasets",
                "columns_used": ["Village", "State", "Capital", metric_col]
            }

        # 1. Check which state has the most/fewest villages
        if "village" in q_lower and any(w in q_lower for w in ["most villages", "fewest villages", "least villages", "highest villages", "number of villages"]):
            res_v = multi_child_executor.rank_states_by_village_count(is_max=(not is_min))
            ext_type = "fewest" if is_min else "most"
            ans = f"The state with the {ext_type} recorded villages across all datasets is {res_v['top_state']} with {res_v['top_value']} villages."
            return {
                "operation": "RANK_VILLAGE_COUNT",
                "scope": "MULTI_CHILD_COMPARISON",
                "results": res_v["rankings"],
                "result_count": len(res_v["rankings"]),
                "answer": ans,
                "child_dataset": "all_child_datasets",
                "columns_used": ["State", "Village"]
            }

        # 2. Check which state has the highest/lowest total population or metric
        if any(w in q_lower for w in ["which state has the highest", "which state has the lowest", "highest total", "lowest total", "state with the highest", "state with the lowest", "which state has the max", "which state has the min", "which capital belongs to the state with", "capital of the state with"]):
            agg_f = "mean" if metric_col == "Literacy_Rate_Percent" else "sum"
            res_m = multi_child_executor.rank_states_by_metric(metric_col, agg_func=agg_f, is_max=(not is_min))
            ext_type = "lowest" if is_min else "highest"
            val_formatted = f"{res_m['top_value']:,.2f}%" if agg_f == "mean" else f"{res_m['top_value']:,.0f}"
            cap_info = ""
            if "capital" in q_lower:
                for meta in parent_child_registry.get_registered_child_datasets().values():
                    c_df = parent_child_registry.load_child_dataframe(meta)
                    if c_df is not None and not c_df.empty and "State" in c_df.columns and "Capital" in c_df.columns:
                        match_s = c_df[c_df["State"].astype(str).str.lower() == str(res_m['top_state']).lower()]
                        if not match_s.empty:
                            cap_val = str(match_s.iloc[0]["Capital"])
                            cap_info = f", and its capital is {cap_val}"
                            break
            ans = f"The state with the {ext_type} total {metric_col.replace('_', ' ')} across all registered states is {res_m['top_state']}{cap_info} with {val_formatted}."
            return {
                "operation": "RANK_STATE_METRIC",
                "scope": "MULTI_CHILD_COMPARISON",
                "results": res_m["rankings"],
                "result_count": len(res_m["rankings"]),
                "answer": ans,
                "child_dataset": "all_child_datasets",
                "columns_used": ["State", metric_col]
            }

        # Check comparison between two states
        states_found: List[str] = []
        for state_name in parent_child_registry.get_registered_child_datasets().keys():
            if re.search(r"\b" + re.escape(state_name.lower()) + r"\b", q_lower):
                states_found.append(state_name)

        if len(states_found) >= 2 and any(w in q_lower for w in ["compare", "vs", "versus", "difference", "higher", "more"]):
            comp_res = multi_child_executor.compare_states(states_found[0], states_found[1], metric_col)
            ans = (
                f"Comparison of total {metric_col} between {states_found[0]} and {states_found[1]}: "
                f"{states_found[0]} has {comp_res['value_a']:,.0f}, while {states_found[1]} has {comp_res['value_b']:,.0f}. "
                f"{comp_res['higher']} is higher by {abs(comp_res['difference']):,.0f}."
            )
            return {
                "operation": "COMPARE",
                "scope": "MULTI_CHILD_COMPARISON",
                "results": [comp_res],
                "result_count": 1,
                "answer": ans,
                "child_dataset": "all_child_datasets",
                "columns_used": ["State", metric_col]
            }

        # Check aggregation (SUM, AVG, COUNT) across all villages in India
        is_avg = any(w in q_lower for w in ["average", "avg", "mean"])
        is_sum = any(w in q_lower for w in ["total", "sum"])

        if is_avg:
            comb_df = multi_child_executor.get_combined_dataframe()
            val = float(comb_df[metric_col].mean()) if metric_col in comb_df.columns else 0.0
            ans = f"The average {metric_col} across all registered villages in India is {val:,.2f}."
            return {
                "operation": "AVERAGE",
                "scope": "MULTI_CHILD_COMPARISON",
                "results": [{"metric": metric_col, "average": val}],
                "result_count": 1,
                "answer": ans,
                "child_dataset": "all_child_datasets",
                "columns_used": [metric_col]
            }

        # Extremes (MAX or MIN individual village across all child datasets)
        extreme_row = multi_child_executor.find_extreme_village(metric_col, is_max=(not is_min))

        if extreme_row:
            extreme_type = "lowest" if is_min else "highest"
            val = extreme_row.get(metric_col)
            v_name = extreme_row.get("Village")
            st_name = extreme_row.get("State")
            cap_name = extreme_row.get("Capital")
            val_str = f"{val:,.1f}" if isinstance(val, float) else f"{val:,}"

            secondary_info = ""
            if any(w in q_lower for w in ["female", "females", "women"]):
                f_val = extreme_row.get("No_of_Females")
                if f_val is not None:
                    secondary_info = f", and has {f_val:,.0f} females"
            elif any(w in q_lower for w in ["male", "males", "men"]):
                m_val = extreme_row.get("No_of_Males")
                if m_val is not None:
                    secondary_info = f", and has {m_val:,.0f} males"
            elif any(w in q_lower for w in ["literacy", "literate"]):
                l_val = extreme_row.get("Literacy_Rate_Percent")
                if l_val is not None:
                    secondary_info = f", with a literacy rate of {l_val:.1f}%"
            elif any(w in q_lower for w in ["household", "households", "houses"]):
                h_val = extreme_row.get("Households")
                if h_val is not None:
                    secondary_info = f", and {h_val:,} households"

            ans = (
                f"The village with the {extreme_type} {metric_col} across all states is {v_name} "
                f"in {st_name} (Capital: {cap_name}) with a {metric_col} of {val_str}{secondary_info}."
            )
            return {
                "operation": "MIN" if is_min else "MAX",
                "scope": "MULTI_CHILD_COMPARISON",
                "results": [extreme_row],
                "result_count": 1,
                "answer": ans,
                "child_dataset": "all_child_datasets",
                "columns_used": ["Village", "State", "Capital", metric_col]
            }

        return {
            "operation": "NO_MATCH",
            "scope": "MULTI_CHILD_COMPARISON",
            "results": [],
            "result_count": 0,
            "answer": "No matching village records found across registered child datasets.",
            "child_dataset": "all_child_datasets",
            "columns_used": []
        }

    def _execute_single_child(
        self,
        question: str,
        q_lower: str,
        scope: str,
        entity_name: Optional[str],
        child_path: Optional[Path],
        session_id: Optional[str]
    ) -> Dict[str, Any]:
        """Execute query on a specific child dataset."""
        if not child_path or not child_path.exists():
            avail_states = ", ".join(parent_child_registry.get_available_states())
            return {
                "operation": "NO_MATCH",
                "scope": scope,
                "results": [],
                "result_count": 0,
                "answer": f"Data for '{entity_name}' is not available in the registered datasets. Available states are: {avail_states}.",
                "child_dataset": None,
                "columns_used": []
            }

        df = parent_child_registry.load_child_dataframe(child_path)
        if df is None or df.empty:
            return {
                "operation": "NO_MATCH",
                "scope": scope,
                "results": [],
                "result_count": 0,
                "answer": f"Child dataset for '{entity_name}' contains no records.",
                "child_dataset": child_path.name,
                "columns_used": []
            }

        metric_col = self._extract_metric_column(q_lower)

        # 1. Check for specific Village Mention or ID (e.g. Hyderabad_Village_01 or TG-001)
        village_target = None
        id_target = None

        m_vid = re.search(r"\b([a-zA-Z]{2}-\d{3})\b", question)
        if m_vid:
            id_target = m_vid.group(1).upper()

        m_vname = re.search(r"\b([a-zA-Z]+_village_\d+)\b", q_lower)
        if m_vname:
            village_target = m_vname.group(1)

        m_num = re.search(r"\bvillage\s*([0-9]+)\b", q_lower)
        if not village_target and m_num:
            village_target = f"village {int(m_num.group(1))}"

        # Pronoun check for previous village
        if not village_target and not id_target and session_id:
            last_v = conversation_manager.get_last_village(session_id)
            if last_v and re.search(r"\b(?:it|its|that village|the village|this village)\b", q_lower):
                village_target = last_v.lower()

        # If specific village row requested
        if village_target or id_target:
            matched_row = None
            if id_target and "Village_ID" in df.columns:
                sub_df = df[df["Village_ID"].astype(str).str.upper() == id_target]
                if not sub_df.empty:
                    matched_row = sub_df.iloc[0].to_dict()
            elif village_target and "Village" in df.columns:
                sub_df = df[df["Village"].astype(str).str.lower() == village_target]
                if not sub_df.empty:
                    matched_row = sub_df.iloc[0].to_dict()
                else:
                    # Numbered check e.g. "village 1" or "village 01"
                    m_idx = re.search(r"village\s*([0-9]+)", village_target, re.IGNORECASE)
                    if m_idx:
                        idx = int(m_idx.group(1))
                        for _, r in df.iterrows():
                            v_str = str(r["Village"]).lower()
                            if v_str.endswith(f"_{idx:02d}") or v_str.endswith(f"_{idx}") or f"village_{idx:02d}" in v_str or f"village_{idx}" in v_str:
                                matched_row = r.to_dict()
                                break
                        if matched_row is None and 1 <= idx <= len(df):
                            matched_row = df.iloc[idx - 1].to_dict()
                    else:
                        # Try fuzzy token match
                        for _, r in df.iterrows():
                            if village_target in str(r["Village"]).lower():
                                matched_row = r.to_dict()
                                break

            if matched_row:
                v_name = matched_row.get("Village")
                if metric_col and metric_col in matched_row:
                    m_val = matched_row[metric_col]
                    val_str = f"{m_val:,.1f}" if isinstance(m_val, float) else f"{m_val:,}"
                    ans = f"In {matched_row.get('State', entity_name)}, {v_name} has a {metric_col.replace('_', ' ')} of {val_str}."
                    cols_used = ["Village", metric_col]
                else:
                    st_label = matched_row.get("State") or entity_name
                    cap_label = f" (Capital: {matched_row.get('Capital')})" if matched_row.get('Capital') else ""
                    st_suffix = f" in {st_label}{cap_label}" if st_label else ""
                    ans = (
                        f"Record for {v_name} ({matched_row.get('Village_ID')}){st_suffix}: "
                        f"Population: {matched_row.get('Population'):,}, Males: {matched_row.get('No_of_Males'):,}, "
                        f"Females: {matched_row.get('No_of_Females'):,}, Literacy Rate: {matched_row.get('Literacy_Rate_Percent')}%, "
                        f"Households: {matched_row.get('Households'):,}, Area: {matched_row.get('Area_Sq_Km')} sq km."
                    )
                    cols_used = list(matched_row.keys())

                return {
                    "operation": "LOOKUP",
                    "scope": scope,
                    "results": [matched_row],
                    "result_count": 1,
                    "answer": ans,
                    "child_dataset": child_path.name,
                    "village": v_name,
                    "state": matched_row.get("State"),
                    "capital": matched_row.get("Capital"),
                    "columns_used": cols_used
                }
            else:
                return {
                    "operation": "NO_MATCH",
                    "scope": scope,
                    "results": [],
                    "result_count": 0,
                    "answer": f"No matching village was found for '{id_target or village_target}' in the dataset.",
                    "child_dataset": child_path.name if child_path else None,
                    "columns_used": []
                }

        # 2. Aggregations on the Child Dataset
        # Count of villages: "How many villages are in Bihar?"
        if any(w in q_lower for w in ["how many", "count of", "number of"]) and any(w in q_lower for w in ["village", "villages"]):
            count_val = len(df)
            ans = f"There are {count_val} villages recorded in the dataset for {entity_name}."
            return {
                "operation": "COUNT",
                "scope": scope,
                "results": [{"count": count_val, "entity": entity_name}],
                "result_count": count_val,
                "answer": ans,
                "child_dataset": child_path.name,
                "columns_used": ["Village"]
            }

        # Extreme within this child dataset: "Highest population village in Telangana"
        is_min = any(w in q_lower for w in ["lowest", "minimum", "min", "smallest", "least"])
        is_extreme = is_min or any(w in q_lower for w in ["highest", "maximum", "max", "largest", "biggest", "top"])

        if is_extreme and metric_col and metric_col in df.columns:
            sorted_df = df.sort_values(by=metric_col, ascending=is_min).dropna(subset=[metric_col])
            if not sorted_df.empty:
                top_row = sorted_df.iloc[0].to_dict()
                v_name = top_row.get("Village")
                m_val = top_row.get(metric_col)
                ext_str = "lowest" if is_min else "highest"
                val_str = f"{m_val:,.1f}" if isinstance(m_val, float) else f"{m_val:,}"
                ans = f"In {entity_name}, the village with the {ext_str} {metric_col.replace('_', ' ')} is {v_name} with {val_str}."
                return {
                    "operation": "MIN" if is_min else "MAX",
                    "scope": scope,
                    "results": [top_row],
                    "result_count": 1,
                    "answer": ans,
                    "child_dataset": child_path.name,
                    "village": v_name,
                    "state": top_row.get("State"),
                    "capital": top_row.get("Capital"),
                    "columns_used": ["Village", metric_col]
                }

        # Sum or Average within child dataset: "Total village population of Telangana" or "what is the population gj"
        is_avg = any(w in q_lower for w in ["average", "avg", "mean"]) or metric_col == "Literacy_Rate_Percent"
        if is_avg and metric_col and metric_col in df.columns:
            avg_val = float(df[metric_col].mean())
            pct_sign = "%" if ("Percent" in metric_col or "Rate" in metric_col) else ""
            ans = f"The average {metric_col.replace('_', ' ')} across recorded villages in {entity_name} is {avg_val:,.2f}{pct_sign}."
            return {
                "operation": "AVERAGE",
                "scope": scope,
                "results": [{"metric": metric_col, "average": avg_val, "value": avg_val, "entity": entity_name, "State": entity_name, "Child_Dataset": child_path.name}],
                "aggregation": {"column": metric_col, "operation": "AVERAGE", "value": avg_val},
                "result_count": 1,
                "answer": ans,
                "child_dataset": child_path.name,
                "columns_used": [metric_col],
                "calculation_formula": f"AVG({metric_col})"
            }

        if metric_col and metric_col in df.columns:
            sum_val = float(df[metric_col].sum())
            ans = f"The total {metric_col.replace('_', ' ')} across all recorded villages in {entity_name} is {sum_val:,.0f}."
            return {
                "operation": "SUM",
                "scope": scope,
                "results": [{"metric": metric_col, "sum": sum_val, "total": sum_val, "value": sum_val, "entity": entity_name, "State": entity_name, "Child_Dataset": child_path.name}],
                "aggregation": {"column": metric_col, "operation": "SUM", "value": sum_val},
                "result_count": 1,
                "answer": ans,
                "child_dataset": child_path.name,
                "columns_used": [metric_col],
                "calculation_formula": f"SUM({metric_col})"
            }

        if metric_col and metric_col not in df.columns:
            avail_cols = ", ".join([c for c in df.columns if not c.startswith("_")])
            return {
                "operation": "NO_MATCH",
                "scope": scope,
                "results": [],
                "result_count": 0,
                "answer": f"The attribute '{metric_col.replace('_', ' ')}' is not present in the {entity_name} dataset. Available attributes are: {avail_cols}.",
                "child_dataset": child_path.name,
                "columns_used": []
            }

        # Check if question asked for an unknown attribute not present in child dataset (e.g. "What is the elevation of Gujarat?")
        m_attr = re.search(r"(?:what\s+is\s+the|tell\s+me\s+the|get\s+the|find\s+the)\s+([a-zA-Z_]+)\s+(?:of|in|for)\b", q_lower)
        if not m_attr:
            m_attr = re.search(r"\b([a-zA-Z_]+)\s+(?:of|in|for)\s+[a-zA-Z\s]+", q_lower)
        if m_attr:
            cand_attr = m_attr.group(1).strip().lower()
            stop_words = {
                "capital", "state", "code", "id", "village", "villages", "record",
                "records", "data", "dataset", "details", "info", "information",
                "all", "list", "name", "names", "total", "sum", "average", "mean", "count"
            }
            if cand_attr not in stop_words and not any(cand_attr == c.lower() for c in df.columns):
                avail_cols = ", ".join([c for c in df.columns if not c.startswith("_")])
                return {
                    "operation": "NO_MATCH",
                    "scope": scope,
                    "results": [],
                    "result_count": 0,
                    "answer": f"The attribute '{cand_attr}' is not present in the {entity_name} dataset. Available attributes are: {avail_cols}.",
                    "child_dataset": child_path.name if child_path else None,
                    "columns_used": []
                }

        # General list of villages in this child dataset
        records = df.head(10).to_dict(orient="records")
        ans = f"Found {len(df)} villages for {entity_name}. Showing top {len(records)}: " + ", ".join(str(r.get("Village")) for r in records) + "."
        return {
            "operation": "FILTER",
            "scope": scope,
            "results": records,
            "result_count": len(df),
            "answer": ans,
            "child_dataset": child_path.name,
            "columns_used": ["Village"]
        }

    def _extract_metric_column(self, q_lower: str) -> Optional[str]:
        """Extract canonical child metric column from text."""
        # Longest match first
        for alias in sorted(self.METRIC_ALIASES.keys(), key=lambda x: len(x), reverse=True):
            if alias in q_lower:
                return self.METRIC_ALIASES[alias]
        return None


hierarchical_query_engine = HierarchicalQueryEngine()
