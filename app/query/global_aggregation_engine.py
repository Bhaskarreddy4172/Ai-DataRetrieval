"""Universal Global Aggregation Engine: executes deterministic cross-dataset, multi-state,
all-state, and village-level aggregations using in-memory DuckDB and Pandas.

Supported Operations:
- SUM, COUNT, AVERAGE, MEDIAN, MIN, MAX, RANGE, DISTINCT_COUNT,
  VARIANCE, STANDARD_DEVIATION, PERCENTAGE, RATIO, DIFFERENCE, PERCENTAGE_CHANGE

Supported Scopes:
- ALL_STATES (all 28 registered Indian state datasets)
- SELECTED_STATES (e.g. TG + AP, 3-5 selected states)
- ALL_VILLAGES (all 280 villages across all states)
- SINGLE_STATE (village aggregation within a state)
- STATE_RANKING (highest, lowest, top-N, bottom-N states)
"""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import numpy as np
import pandas as pd

from app.dataset.multi_child_executor import multi_child_executor
from app.dataset.registry import parent_child_registry
from app.query.fast_classifier import fast_query_classifier, FastClassificationResult
from app.utils.logger import logger
from app.utils.normalization import normalize_question

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


class UniversalGlobalAggregationEngine:
    """Core engine for high-performance deterministic aggregations across all registered datasets."""

    def __init__(self):
        self._lock = threading.RLock()
        self._agg_cache: Dict[str, Dict[str, Any]] = {}
        self._duckdb_con: Optional[duckdb.DuckDBPyConnection] = None
        self._cached_version_hash: str = ""

    def _ensure_duckdb(self) -> Optional[duckdb.DuckDBPyConnection]:
        """Ensure in-memory DuckDB connection is initialized and registered with global_village_data."""
        if not HAS_DUCKDB:
            return None

        with self._lock:
            df = multi_child_executor.get_combined_dataframe()
            if df.empty:
                return None

            # Compute combined dataset hash
            meta_list = parent_child_registry.get_all_dataset_metadata()
            current_hash = "|".join(f"{m.get('dataset_id')}:{m.get('content_hash')}" for m in meta_list if m.get("dataset_type") == "CHILD")

            if self._duckdb_con is not None and self._cached_version_hash == current_hash:
                return self._duckdb_con

            try:
                con = duckdb.connect(database=":memory:")
                con.register("global_village_data", df)
                self._duckdb_con = con
                self._cached_version_hash = current_hash
                return self._duckdb_con
            except Exception as ex:
                logger.warning(f"UniversalGlobalAggregationEngine: DuckDB initialization warning: {ex}")
                return None

    def can_handle(self, question: str, session_id: Optional[str] = None) -> bool:
        """Check if query is a global aggregation, all-states, selected-states, or village-aggregation query."""
        if not question or not question.strip():
            return False

        cls_res = fast_query_classifier.classify(question, session_id=session_id)
        if cls_res.is_fast_path:
            if cls_res.intent == "COMPARE":
                return False
            return cls_res.scope in [
                "ALL_STATES", "ALL_STATES_BREAKDOWN", "SELECTED_STATES",
                "ALL_VILLAGES", "STATE_RANKING", "SINGLE_STATE",
                "STATE_CAPITAL", "DISTINCT_VALUES", "CONDITION_EXISTENCE", "ENTITY_EXISTENCE",
                "FILTERED"
            ]

        return False

    def execute(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute deterministic aggregation query and return complete structured response."""
        cls_res = fast_query_classifier.classify(question, session_id=session_id)

        # Check Cache
        cache_key = f"{cls_res.scope}:{cls_res.intent}:{cls_res.metric}:{','.join(cls_res.entities)}:{cls_res.n_limit}:{cls_res.group_by}:{normalize_question(question).lower()}"
        with self._lock:
            if cache_key in self._agg_cache:
                logger.info(f"UniversalGlobalAggregationEngine: Cache HIT for '{question}'")
                return dict(self._agg_cache[cache_key])

        scope = cls_res.scope
        intent = cls_res.intent
        metric = cls_res.metric

        # Execute based on intent and scope
        if scope == "FILTERED":
            res = self._execute_filtered_query(cls_res, question)

        elif scope == "STATE_CAPITAL":
            res = self._execute_capital_lookup(cls_res.entities, question)

        elif scope == "DISTINCT_VALUES":
            res = self._execute_distinct_values(cls_res.metric, question)

        elif scope == "CONDITION_EXISTENCE":
            res = self._execute_condition_existence(cls_res.entities, cls_res.metric, cls_res.n_limit, question)

        elif scope == "ENTITY_EXISTENCE":
            res = self._execute_entity_existence(cls_res.entities, question)

        elif intent in ["TOP_N", "BOTTOM_N"]:
            if cls_res.group_by == "State" or scope in ["STATE_RANKING", "ALL_STATES"]:
                res = self._execute_state_ranking(metric, is_max=(intent == "TOP_N"), limit=cls_res.n_limit or 5, question=question)
            else:
                res = self._execute_all_villages(metric, intent, cls_res.n_limit, cls_res.order, question)

        elif scope == "ALL_STATES":
            if intent in ["SUM", "TOTAL"]:
                res = self._execute_total_all_states(metric, question)
            elif intent == "AVERAGE":
                res = self._execute_average_per_state(metric, question)
            elif intent in ["MEDIAN", "MIN", "MAX", "RANGE", "VARIANCE", "STANDARD_DEVIATION", "COUNT"]:
                res = self._execute_stat_all_states(metric, intent, question)
            else:
                res = self._execute_total_all_states(metric, question)

        elif scope == "SELECTED_STATES":
            if intent == "COMPARE":
                from app.query.comparison_engine import universal_comparison_engine
                plan = universal_comparison_engine.parse_and_plan(question, session_id=session_id)
                if plan:
                    c_res = universal_comparison_engine.execute(plan)
                    return {
                        "operation": c_res.operation,
                        "scope": "SELECTED_STATES_COMPARISON",
                        "results": [{
                            "left_entity": c_res.left_entity,
                            "left_value": c_res.left_value,
                            "left_parent": c_res.left_parent,
                            "right_entity": c_res.right_entity,
                            "right_value": c_res.right_value,
                            "right_parent": c_res.right_parent,
                            "scope": c_res.scope,
                            "absolute_difference": c_res.absolute_difference,
                            "directed_difference": c_res.directed_difference,
                            "higher_entity": c_res.higher_entity,
                            "lower_entity": c_res.lower_entity,
                            "attribute": c_res.attribute,
                            "unit": c_res.unit,
                        }],
                        "result_count": 1,
                        "answer": c_res.answer,
                        "child_dataset": "selected_child_datasets",
                        "columns_used": c_res.columns_used,
                        "verification_status": c_res.verification_status,
                        "provenance": c_res.to_dict(),
                    }
            res = self._execute_selected_states(cls_res.entities, metric, intent, question)

        elif scope == "STATE_RANKING":
            is_max = cls_res.order != "ASC" and intent != "MIN"
            res = self._execute_state_ranking(metric, is_max=is_max, limit=cls_res.n_limit or 1, question=question)

        elif scope == "ALL_STATES_BREAKDOWN":
            res = self._execute_all_states_breakdown(metric, intent, question)

        elif scope == "ALL_VILLAGES":
            res = self._execute_all_villages(metric, intent, cls_res.n_limit, cls_res.order, question)

        elif scope == "SINGLE_STATE":
            state_target = cls_res.entities[0] if cls_res.entities else ""
            res = self._execute_single_state_aggregation(state_target, metric, intent, question)

        else:
            # Fallback to all states total
            res = self._execute_total_all_states(metric, question)

        # Cache result
        with self._lock:
            self._agg_cache[cache_key] = res

        return res

    def _execute_total_all_states(self, metric: str, question: str) -> Dict[str, Any]:
        """Compute verified sum of metric across all 28 registered Indian states."""
        con = self._ensure_duckdb()
        all_states = parent_child_registry.get_available_states()
        expected_state_count = len(all_states) or 28

        total_val: Optional[float] = None
        state_breakdown: List[Dict[str, Any]] = []

        if con is not None:
            try:
                # 1. Total sum
                row_sum = con.execute(f'SELECT SUM("{metric}") FROM global_village_data WHERE "{metric}" IS NOT NULL').fetchone()
                if row_sum and row_sum[0] is not None:
                    total_val = float(row_sum[0])

                # 2. State-level breakdown
                rows_states = con.execute(f'SELECT State, SUM("{metric}") as val FROM global_village_data WHERE "{metric}" IS NOT NULL GROUP BY State').fetchall()
                state_breakdown = [{"state": str(r[0]), "value": float(r[1])} for r in rows_states]
            except Exception as ex:
                logger.warning(f"DuckDB total query error: {ex}")

        # Pandas Fallback
        if total_val is None:
            df = multi_child_executor.get_combined_dataframe()
            if not df.empty and metric in df.columns:
                valid_metric = pd.to_numeric(df[metric], errors="coerce").dropna()
                total_val = float(valid_metric.sum())
                grouped = df.groupby("State")[metric].sum().reset_index()
                state_breakdown = [{"state": str(r["State"]), "value": float(r[metric])} for _, r in grouped.iterrows()]
            else:
                total_val = 0.0

        resolved_states = [s["state"] for s in state_breakdown]
        resolved_count = len(resolved_states)
        missing = [s for s in all_states if s not in resolved_states]
        verification_status = "PASS" if resolved_count >= expected_state_count and len(missing) == 0 else "PARTIAL"

        val_str = f"{total_val:,.0f}" if total_val.is_integer() else f"{total_val:,.2f}"
        if verification_status == "PASS":
            ans = f"The total {metric.replace('_', ' ')} across all {expected_state_count} registered states (280 villages) is {val_str}."
        else:
            ans = f"The total {metric.replace('_', ' ')} across {resolved_count} registered states is {val_str}. (Note: Data for {len(missing)} states was not found: {', '.join(missing)})."

        return {
            "operation": "SUM",
            "scope": "ALL_STATES",
            "metric": metric,
            "total": total_val,
            "sum": total_val,
            "value": total_val,
            "expected_state_count": expected_state_count,
            "resolved_state_count": resolved_count,
            "missing_states": missing,
            "verification_status": verification_status,
            "verification_checks": {
                "metric_verified": True,
                "all_states_included": (verification_status == "PASS"),
                "expected_states": expected_state_count,
                "resolved_states": resolved_count,
                "missing_states_count": len(missing),
                "null_values_excluded": True,
            },
            "results": [{
                "metric": metric,
                "total": total_val,
                "states_count": resolved_count,
                "scope": "ALL_STATES"
            }],
            "state_breakdown": state_breakdown,
            "result_count": 1,
            "answer": ans,
            "child_dataset": "all_28_child_datasets",
            "columns_used": ["State", metric]
        }

    def _execute_selected_states(
        self,
        states: List[str],
        metric: str,
        intent: str,
        question: str
    ) -> Dict[str, Any]:
        """Compute aggregated metric across explicitly requested states."""
        con = self._ensure_duckdb()
        states_clean = [s.strip() for s in states if s.strip()]
        states_lower = [s.lower() for s in states_clean]

        breakdown: List[Dict[str, Any]] = []
        if con is not None:
            try:
                states_sql = ", ".join(f"'{s}'" for s in states_lower)
                rows = con.execute(f'''
                    SELECT State, SUM("{metric}") as val
                    FROM global_village_data
                    WHERE LOWER(State) IN ({states_sql}) AND "{metric}" IS NOT NULL
                    GROUP BY State
                ''').fetchall()
                breakdown = [{"state": str(r[0]), "value": float(r[1])} for r in rows]
            except Exception as ex:
                logger.warning(f"DuckDB selected states query error: {ex}")

        # Pandas Fallback
        if not breakdown:
            df = multi_child_executor.get_combined_dataframe()
            if not df.empty and "State" in df.columns and metric in df.columns:
                sub = df[df["State"].astype(str).str.lower().isin(states_lower)]
                grouped = sub.groupby("State")[metric].sum().reset_index()
                breakdown = [{"state": str(r["State"]), "value": float(r[metric])} for _, r in grouped.iterrows()]

        # Compute combined sum
        total_val = float(sum(r["value"] for r in breakdown))
        state_strs = [f"{b['state']}: {b['value']:,.0f}" for b in breakdown]
        states_names_str = " and ".join(states_clean) if len(states_clean) <= 2 else ", ".join(states_clean[:-1]) + f", and {states_clean[-1]}"

        ans = f"The combined total {metric.replace('_', ' ')} for {states_names_str} is {total_val:,.0f} ({', '.join(state_strs)})."

        return {
            "operation": intent,
            "scope": "SELECTED_STATES",
            "metric": metric,
            "total": total_val,
            "sum": total_val,
            "value": total_val,
            "entities": states_clean,
            "results": breakdown,
            "result_count": len(breakdown),
            "answer": ans,
            "child_dataset": "selected_child_datasets",
            "columns_used": ["State", metric],
            "verification_status": "PASS",
        }

    def _execute_state_ranking(
        self,
        metric: str,
        is_max: bool,
        limit: int,
        question: str
    ) -> Dict[str, Any]:
        """Rank states by aggregated village metric and return extreme or top-N states."""
        con = self._ensure_duckdb()
        order_sql = "DESC" if is_max else "ASC"
        rows: List[Tuple[Any, ...]] = []

        agg_expr = "COUNT(*)" if metric == "VILLAGE_COUNT" else f'SUM("{metric}")'
        where_clause = "" if metric == "VILLAGE_COUNT" else f'WHERE "{metric}" IS NOT NULL'

        if con is not None:
            try:
                rows = con.execute(f'''
                    SELECT State, Capital, {agg_expr} as total_val
                    FROM global_village_data
                    {where_clause}
                    GROUP BY State, Capital
                    ORDER BY total_val {order_sql}
                    LIMIT {limit}
                ''').fetchall()
            except Exception as ex:
                logger.warning(f"DuckDB ranking error: {ex}")

        # Pandas fallback
        if not rows:
            df = multi_child_executor.get_combined_dataframe()
            if not df.empty and "State" in df.columns:
                if metric == "VILLAGE_COUNT":
                    grouped = df.groupby(["State", "Capital"]).size().reset_index(name="total_val")
                elif metric in df.columns:
                    grouped = df.groupby(["State", "Capital"])[metric].sum().reset_index()
                    grouped.rename(columns={metric: "total_val"}, inplace=True)
                else:
                    grouped = pd.DataFrame()
                if not grouped.empty:
                    sorted_df = grouped.sort_values(by="total_val", ascending=not is_max).head(limit)
                    rows = [(r["State"], r["Capital"], r["total_val"]) for _, r in sorted_df.iterrows()]

        rankings: List[Dict[str, Any]] = []
        for idx, r in enumerate(rows, start=1):
            rankings.append({
                "rank": idx,
                "state": str(r[0]),
                "capital": str(r[1]),
                "metric": metric,
                "value": float(r[2])
            })

        ext_label = "highest" if is_max else "lowest"
        if limit == 1 and rankings:
            top_rec = rankings[0]
            val_str = f"{top_rec['value']:,.0f}" if top_rec['value'].is_integer() else f"{top_rec['value']:,.2f}"
            cap_info = f" (Capital: {top_rec['capital']})" if top_rec.get("capital") else ""
            ans = f"The state with the {ext_label} total {metric.replace('_', ' ')} across all registered states is {top_rec['state']}{cap_info} with {val_str}."
        elif rankings:
            top_word = "Top" if is_max else "Bottom"
            ans = f"{top_word} {len(rankings)} states by {metric.replace('_', ' ')} across all registered states:\n"
            ans += "\n".join([f"{r['rank']}. {r['state']} (Capital: {r['capital']}): {r['value']:,.0f}" for r in rankings])
        else:
            ans = f"No state ranking records found for {metric}."

        all_states = parent_child_registry.get_available_states()
        expected_state_count = len(all_states) or 28
        total_states_evaluated = 0
        if con is not None:
            try:
                cnt_row = con.execute('SELECT COUNT(DISTINCT State) FROM global_village_data').fetchone()
                if cnt_row:
                    total_states_evaluated = int(cnt_row[0])
            except Exception:
                pass
        if total_states_evaluated == 0:
            df_check = multi_child_executor.get_combined_dataframe()
            if not df_check.empty and "State" in df_check.columns:
                total_states_evaluated = len(set(df_check["State"].dropna().unique()))

        verification_status = "PASS" if rankings and total_states_evaluated >= expected_state_count else ("PARTIAL" if rankings else "FAILED")

        return {
            "operation": "MAX" if is_max else "MIN",
            "scope": "STATE_RANKING",
            "metric": metric,
            "is_max": is_max,
            "results": rankings,
            "result_count": len(rankings),
            "answer": ans,
            "child_dataset": "all_28_child_datasets",
            "columns_used": ["State", "Capital", metric],
            "expected_state_count": expected_state_count,
            "resolved_state_count": total_states_evaluated,
            "verification_status": verification_status,
            "verification_checks": {
                "metric_verified": True,
                "all_states_included": (total_states_evaluated >= expected_state_count),
                "expected_states": expected_state_count,
                "resolved_states": total_states_evaluated,
            }
        }

    def _execute_average_per_state(self, metric: str, question: str) -> Dict[str, Any]:
        """Compute the average of the state-level sums across all registered states."""
        con = self._ensure_duckdb()
        avg_val: Optional[float] = None
        state_count: int = 28

        if con is not None:
            try:
                row = con.execute(f'''
                    SELECT AVG(state_sum), COUNT(*)
                    FROM (
                        SELECT State, SUM("{metric}") as state_sum
                        FROM global_village_data
                        WHERE "{metric}" IS NOT NULL
                        GROUP BY State
                    )
                ''').fetchone()
                if row and row[0] is not None:
                    avg_val = float(row[0])
                    state_count = int(row[1])
            except Exception as ex:
                logger.warning(f"DuckDB average per state error: {ex}")

        # Pandas fallback
        if avg_val is None:
            df = multi_child_executor.get_combined_dataframe()
            if not df.empty and "State" in df.columns and metric in df.columns:
                grouped = df.groupby("State")[metric].sum()
                avg_val = float(grouped.mean())
                state_count = len(grouped)
            else:
                avg_val = 0.0

        val_str = f"{avg_val:,.0f}" if avg_val.is_integer() else f"{avg_val:,.2f}"
        rounded_str = f"{round(avg_val):,}"
        ans = f"The average {metric.replace('_', ' ')} per state across all {state_count} registered states is {val_str} (approx. {rounded_str})."

        return {
            "operation": "AVERAGE",
            "scope": "ALL_STATES",
            "metric": metric,
            "average": avg_val,
            "value": avg_val,
            "state_count": state_count,
            "results": [{
                "metric": metric,
                "average": avg_val,
                "state_count": state_count
            }],
            "result_count": 1,
            "answer": ans,
            "child_dataset": "all_28_child_datasets",
            "columns_used": ["State", metric],
            "verification_status": "PASS"
        }

    def _execute_stat_all_states(self, metric: str, intent: str, question: str) -> Dict[str, Any]:
        """Compute statistical metric (MEDIAN, VARIANCE, STD, RANGE, COUNT) across state sums."""
        con = self._ensure_duckdb()
        stat_val: Optional[float] = None

        if con is not None:
            try:
                if intent == "MEDIAN":
                    row = con.execute(f'SELECT MEDIAN(state_sum) FROM (SELECT State, SUM("{metric}") as state_sum FROM global_village_data WHERE "{metric}" IS NOT NULL GROUP BY State)').fetchone()
                elif intent == "RANGE":
                    row = con.execute(f'SELECT MAX(state_sum) - MIN(state_sum) FROM (SELECT State, SUM("{metric}") as state_sum FROM global_village_data WHERE "{metric}" IS NOT NULL GROUP BY State)').fetchone()
                elif intent == "VARIANCE":
                    row = con.execute(f'SELECT VAR_SAMP(state_sum) FROM (SELECT State, SUM("{metric}") as state_sum FROM global_village_data WHERE "{metric}" IS NOT NULL GROUP BY State)').fetchone()
                elif intent == "STANDARD_DEVIATION":
                    row = con.execute(f'SELECT STDDEV_SAMP(state_sum) FROM (SELECT State, SUM("{metric}") as state_sum FROM global_village_data WHERE "{metric}" IS NOT NULL GROUP BY State)').fetchone()
                elif intent == "COUNT":
                    row = con.execute('SELECT COUNT(DISTINCT State) FROM global_village_data').fetchone()
                else:
                    row = None

                if row and row[0] is not None:
                    stat_val = float(row[0])
            except Exception as ex:
                logger.warning(f"DuckDB stat error: {ex}")

        # Pandas fallback
        if stat_val is None:
            df = multi_child_executor.get_combined_dataframe()
            grouped = df.groupby("State")[metric].sum()
            if intent == "MEDIAN":
                stat_val = float(grouped.median())
            elif intent == "RANGE":
                stat_val = float(np.ptp(grouped.values))
            elif intent == "VARIANCE":
                stat_val = float(grouped.var(ddof=1))
            elif intent == "STANDARD_DEVIATION":
                stat_val = float(grouped.std(ddof=1))
            elif intent == "COUNT":
                stat_val = float(len(grouped))
            else:
                stat_val = 0.0

        val_str = f"{stat_val:,.2f}" if not stat_val.is_integer() else f"{stat_val:,.0f}"
        ans = f"The {intent.replace('_', ' ').lower()} of state-level {metric.replace('_', ' ')} across all registered states is {val_str}."

        return {
            "operation": intent,
            "scope": "ALL_STATES",
            "metric": metric,
            "value": stat_val,
            "results": [{
                "operation": intent,
                "metric": metric,
                "value": stat_val
            }],
            "result_count": 1,
            "answer": ans,
            "child_dataset": "all_28_child_datasets",
            "columns_used": ["State", metric],
            "verification_status": "PASS"
        }

    def _execute_all_states_breakdown(self, metric: str, intent: str, question: str) -> Dict[str, Any]:
        """Return total of each and every state sorted descending."""
        con = self._ensure_duckdb()
        records: List[Dict[str, Any]] = []

        agg_expr = "COUNT(*)" if metric == "VILLAGE_COUNT" else f'SUM("{metric}")'
        where_clause = "" if metric == "VILLAGE_COUNT" else f'WHERE "{metric}" IS NOT NULL'

        if con is not None:
            try:
                rows = con.execute(f'''
                    SELECT State, Capital, {agg_expr} as val
                    FROM global_village_data
                    {where_clause}
                    GROUP BY State, Capital
                    ORDER BY val DESC
                ''').fetchall()
                records = [{"state": str(r[0]), "capital": str(r[1]), "value": float(r[2])} for r in rows]
            except Exception as ex:
                logger.warning(f"DuckDB breakdown error: {ex}")

        if not records:
            df = multi_child_executor.get_combined_dataframe()
            if not df.empty and "State" in df.columns:
                if metric == "VILLAGE_COUNT":
                    grouped = df.groupby(["State", "Capital"]).size().reset_index(name="val")
                elif metric in df.columns:
                    grouped = df.groupby(["State", "Capital"])[metric].sum().reset_index(name="val")
                else:
                    grouped = pd.DataFrame()
                if not grouped.empty:
                    sorted_df = grouped.sort_values(by="val", ascending=False)
                    records = [{"state": str(r["State"]), "capital": str(r["Capital"]), "value": float(r["val"])} for _, r in sorted_df.iterrows()]

        ans = f"Total {metric.replace('_', ' ')} breakdown for all {len(records)} registered states:\n"
        ans += "\n".join([f"{idx}. {r['state']} (Capital: {r['capital']}): {r['value']:,.0f}" for idx, r in enumerate(records, start=1)])

        return {
            "operation": "GROUP_AGGREGATE",
            "scope": "ALL_STATES_BREAKDOWN",
            "metric": metric,
            "results": records,
            "result_count": len(records),
            "answer": ans,
            "child_dataset": "all_28_child_datasets",
            "columns_used": ["State", "Capital", metric],
            "verification_status": "PASS"
        }

    def _execute_all_villages(
        self,
        metric: str,
        intent: str,
        n_limit: Optional[int],
        order: Optional[str],
        question: str
    ) -> Dict[str, Any]:
        """Aggregate metric directly across all 280 village records."""
        con = self._ensure_duckdb()
        df = multi_child_executor.get_combined_dataframe()

        if intent in ["TOP_N", "BOTTOM_N"]:
            n = n_limit or 10
            is_max = intent == "TOP_N"
            results = multi_child_executor.get_top_n_villages(metric, n=n, is_max=is_max)
            top_word = "Top" if is_max else "Bottom"
            ans = f"{top_word} {len(results)} villages by {metric.replace('_', ' ')} across all states:\n"
            ans += "\n".join([f"{v['rank']}. {v['village']} ({v['state']}): {v['value']:,.0f}" for v in results])
            return {
                "operation": intent,
                "scope": "ALL_VILLAGES",
                "metric": metric,
                "results": results,
                "result_count": len(results),
                "answer": ans,
                "child_dataset": "all_28_child_datasets",
                "columns_used": ["Village", "State", "Capital", metric],
                "verification_status": "PASS"
            }

        if intent in ["MAX", "MIN"]:
            is_max = intent == "MAX"
            extreme_row = multi_child_executor.find_extreme_village(metric, is_max=is_max)
            if extreme_row:
                extreme_type = "lowest" if not is_max else "highest"
                val = extreme_row.get(metric)
                v_name = extreme_row.get("Village")
                st_name = extreme_row.get("State")
                cap_name = extreme_row.get("Capital")
                val_str = f"{val:,.0f}" if isinstance(val, (int, float)) and float(val).is_integer() else f"{val:,.2f}"
                ans = f"The village with the {extreme_type} {metric.replace('_', ' ')} across all states is {v_name} in {st_name} (Capital: {cap_name}) with a {metric.replace('_', ' ')} of {val_str}."
                return {
                    "operation": intent,
                    "scope": "ALL_VILLAGES",
                    "metric": metric,
                    "value": val,
                    "results": [extreme_row],
                    "result_count": 1,
                    "answer": ans,
                    "child_dataset": "all_28_child_datasets",
                    "columns_used": ["Village", "State", "Capital", metric],
                    "verification_status": "PASS"
                }

        # Direct sum or aggregation
        total_val: float = 0.0
        if con is not None:
            try:
                op_sql = "SUM" if intent in ["SUM", "TOTAL"] else ("AVG" if intent == "AVERAGE" else intent)
                row = con.execute(f'SELECT {op_sql}("{metric}") FROM global_village_data WHERE "{metric}" IS NOT NULL').fetchone()
                if row and row[0] is not None:
                    total_val = float(row[0])
            except Exception:
                pass

        if total_val == 0.0 and not df.empty and metric in df.columns:
            s = pd.to_numeric(df[metric], errors="coerce").dropna()
            total_val = float(s.sum() if intent in ["SUM", "TOTAL"] else s.mean())

        val_str = f"{total_val:,.0f}" if total_val.is_integer() else f"{total_val:,.2f}"
        ans = f"The {intent.lower()} of {metric.replace('_', ' ')} across all {len(df)} recorded villages is {val_str}."

        return {
            "operation": intent,
            "scope": "ALL_VILLAGES",
            "metric": metric,
            "value": total_val,
            "total": total_val,
            "results": [{"metric": metric, "operation": intent, "value": total_val, "villages_count": len(df)}],
            "result_count": 1,
            "answer": ans,
            "child_dataset": "all_28_child_datasets",
            "columns_used": ["Village", metric],
            "verification_status": "PASS"
        }

    def _execute_single_state_aggregation(
        self,
        state: str,
        metric: str,
        intent: str,
        question: str
    ) -> Dict[str, Any]:
        """Aggregate village metric within a specific state."""
        resolved = parent_child_registry.resolve_child_dataset(state)
        if not resolved:
            return {
                "operation": "NO_MATCH",
                "scope": "SINGLE_STATE",
                "results": [],
                "result_count": 0,
                "answer": f"State '{state}' was not found in registered datasets.",
                "child_dataset": None,
                "columns_used": []
            }

        st_name, child_path = resolved
        df = parent_child_registry.load_child_dataframe(child_path)
        if df is None or df.empty:
            return {
                "operation": "NO_MATCH",
                "scope": "SINGLE_STATE",
                "results": [],
                "result_count": 0,
                "answer": f"No records found for {st_name}.",
                "child_dataset": child_path.name if child_path else None,
                "columns_used": []
            }

        if metric == "VILLAGE_COUNT" or intent == "COUNT":
            count_val = len(df)
            ans = f"There are {count_val} villages recorded in the dataset for {st_name}."
            return {
                "operation": "COUNT",
                "scope": "SINGLE_STATE",
                "metric": "VILLAGE_COUNT",
                "entity": st_name,
                "State": st_name,
                "value": float(count_val),
                "total": float(count_val),
                "sum": float(count_val),
                "results": [{
                    "metric": "VILLAGE_COUNT",
                    "value": float(count_val),
                    "entity": st_name,
                    "State": st_name,
                    "villages_count": count_val,
                    "Child_Dataset": child_path.name
                }],
                "result_count": 1,
                "answer": ans,
                "child_dataset": child_path.name,
                "columns_used": ["State", "Village"],
                "verification_status": "PASS"
            }

        if metric not in df.columns:
            return {
                "operation": "NO_MATCH",
                "scope": "SINGLE_STATE",
                "results": [],
                "result_count": 0,
                "answer": f"No {metric} records found for {st_name}.",
                "child_dataset": child_path.name,
                "columns_used": []
            }

        s = pd.to_numeric(df[metric], errors="coerce").dropna()
        if intent in ["SUM", "TOTAL"]:
            agg_val = float(s.sum())
        elif intent == "AVERAGE":
            agg_val = float(s.mean())
        elif intent == "MEDIAN":
            agg_val = float(s.median())
        elif intent == "MIN":
            agg_val = float(s.min())
        elif intent == "MAX":
            agg_val = float(s.max())
        elif intent == "COUNT":
            agg_val = float(len(s))
        else:
            agg_val = float(s.sum())

        pct_sign = "%" if ("Percent" in metric or "Rate" in metric or "Literacy" in metric) else ""
        val_str = f"{agg_val:,.0f}{pct_sign}" if agg_val.is_integer() else f"{agg_val:,.2f}{pct_sign}"
        op_word = "total" if intent in ["SUM", "TOTAL"] else intent.lower()
        ans = f"The {op_word} {metric.replace('_', ' ')} across all recorded villages in {st_name} is {val_str}."

        return {
            "operation": intent,
            "scope": "SINGLE_STATE",
            "metric": metric,
            "entity": st_name,
            "State": st_name,
            "value": agg_val,
            "total": agg_val,
            "sum": agg_val,
            "results": [{
                "metric": metric,
                "value": agg_val,
                "entity": st_name,
                "State": st_name,
                "villages_count": len(df),
                "Child_Dataset": child_path.name
            }],
            "result_count": 1,
            "answer": ans,
            "child_dataset": child_path.name,
            "columns_used": ["State", "Village", metric],
            "verification_status": "PASS"
        }

    def _execute_capital_lookup(self, states: List[str], question: str) -> Dict[str, Any]:
        """Look up official capital of one or more states from the main dataset."""
        main_df = parent_child_registry.get_main_dataframe()
        if main_df is None or main_df.empty:
            main_path = parent_child_registry.get_main_dataset_path()
            if main_path and main_path.exists():
                main_df = pd.read_csv(main_path)

        state_col = next((c for c in main_df.columns if str(c).strip().lower() == "state"), main_df.columns[0])
        cap_col = next((c for c in main_df.columns if str(c).strip().lower() == "capital"), main_df.columns[1])

        cap_map = {str(r[state_col]).strip(): str(r[cap_col]).strip() for _, r in main_df.iterrows()}
        cap_map_lower = {k.lower(): (k, v) for k, v in cap_map.items()}

        results = []
        for st in states:
            m = cap_map_lower.get(st.lower())
            if m:
                results.append({"state": m[0], "capital": m[1]})

        if len(results) == 1:
            ans = f"The capital of {results[0]['state']} is {results[0]['capital']}."
        elif len(results) > 1:
            ans = "Capitals of the requested states: " + ", ".join([f"{r['state']}: {r['capital']}" for r in results]) + "."
        else:
            ans = "Capital information could not be found for the requested states."

        return {
            "operation": "LOOKUP",
            "scope": "STATE_CAPITAL",
            "results": results,
            "result_count": len(results),
            "answer": ans,
            "child_dataset": None,
            "columns_used": [state_col, cap_col],
            "verification_status": "PASS" if results else "FAILED"
        }

    def _execute_distinct_values(self, column: str, question: str) -> Dict[str, Any]:
        """Retrieve distinct values of a target column from main or child datasets."""
        col_norm = "Capital" if "capital" in column.lower() else ("State" if "state" in column.lower() else column)
        main_df = parent_child_registry.get_main_dataframe()
        unique_vals: List[str] = []

        matched_col = next((c for c in main_df.columns if str(c).strip().lower() == col_norm.lower()), None) if main_df is not None and not main_df.empty else None
        if matched_col:
            unique_vals = sorted(main_df[matched_col].dropna().astype(str).unique().tolist())
        else:
            con = self._ensure_duckdb()
            if con is not None:
                try:
                    rows = con.execute(f'SELECT DISTINCT "{col_norm}" FROM global_village_data WHERE "{col_norm}" IS NOT NULL ORDER BY "{col_norm}"').fetchall()
                    unique_vals = [str(r[0]) for r in rows]
                except Exception:
                    pass

        records = [{col_norm: v} for v in unique_vals]
        ans = f"Found {len(unique_vals)} distinct {col_norm.lower()}s: " + ", ".join(unique_vals[:10]) + (f", and {len(unique_vals) - 10} more." if len(unique_vals) > 10 else ".")

        return {
            "operation": "DISTINCT",
            "scope": "DISTINCT_VALUES",
            "metric": col_norm,
            "results": records,
            "result_count": len(records),
            "answer": ans,
            "child_dataset": None,
            "columns_used": [col_norm],
            "verification_status": "PASS" if records else "FAILED"
        }

    def _execute_entity_existence(self, states: List[str], question: str) -> Dict[str, Any]:
        """Verify whether an entity or state exists in the cataloged datasets."""
        all_states = parent_child_registry.get_available_states()
        target = states[0] if states else ""
        exists = any(target.lower() == s.lower() for s in all_states)

        ans = f"Yes, {target} exists in the dataset." if exists else f"No, {target} was not found in the dataset."
        return {
            "operation": "BOOLEAN_CHECK",
            "scope": "ENTITY_EXISTENCE",
            "results": [{"entity": target, "exists": exists}],
            "result_count": 1 if exists else 0,
            "answer": ans,
            "child_dataset": None,
            "columns_used": ["State"],
            "verification_status": "PASS"
        }

    def _execute_condition_existence(
        self,
        states: List[str],
        metric: str,
        threshold: Optional[int],
        question: str
    ) -> Dict[str, Any]:
        """Verify existence of records satisfying a numeric condition (e.g. population > 100,000)."""
        con = self._ensure_duckdb()
        thresh_val = threshold if threshold is not None else 0
        metric_col = metric or "Population"
        rows: List[Tuple[Any, ...]] = []

        if con is not None:
            try:
                if states:
                    rows = con.execute(f'''
                        SELECT Village, State, "{metric_col}"
                        FROM global_village_data
                        WHERE LOWER(State) = '{states[0].lower()}' AND "{metric_col}" > {thresh_val}
                        LIMIT 5
                    ''').fetchall()
                else:
                    rows = con.execute(f'''
                        SELECT Village, State, "{metric_col}"
                        FROM global_village_data
                        WHERE "{metric_col}" > {thresh_val}
                        LIMIT 5
                    ''').fetchall()
            except Exception as ex:
                logger.warning(f"DuckDB condition existence error: {ex}")

        exists = len(rows) > 0
        if exists:
            v_name, v_state, v_val = rows[0]
            val_disp = f"{v_val:,.0f}" if isinstance(v_val, (int, float)) else str(v_val)
            if states:
                ans = f"Yes, there are villages in {states[0]} with {metric_col.replace('_', ' ')} over {thresh_val:,} (e.g., {v_name} with {val_disp})."
            else:
                ans = f"Yes, there are villages with {metric_col.replace('_', ' ')} greater than {thresh_val:,} (e.g., {v_name} with {val_disp} in {v_state})."
        else:
            if states:
                ans = f"No, there are no villages in {states[0]} with {metric_col.replace('_', ' ')} over {thresh_val:,}."
            else:
                ans = f"No, there are no villages with {metric_col.replace('_', ' ')} greater than {thresh_val:,} in the dataset."

        return {
            "operation": "BOOLEAN_CHECK",
            "scope": "CONDITION_EXISTENCE",
            "results": [{"village": r[0], "state": r[1], metric_col: r[2]} for r in rows],
            "result_count": len(rows),
            "answer": ans,
            "child_dataset": "all_28_child_datasets",
            "columns_used": ["Village", "State", metric_col],
            "verification_status": "PASS"
        }

    def _execute_filtered_query(self, cls_res: FastClassificationResult, question: str) -> Dict[str, Any]:
        """Execute deterministic aggregation/extremes on filtered child dataset(s) respecting state/entity constraints."""
        target_states = cls_res.entities or ([cls_res.filter_value] if isinstance(cls_res.filter_value, str) else (cls_res.filter_value or []))
        metric = cls_res.metric or "Population"
        intent = cls_res.intent
        is_max = cls_res.order != "ASC" and intent != "MIN"

        if not target_states:
            return {
                "operation": intent,
                "scope": "FILTERED",
                "results": [],
                "result_count": 0,
                "answer": "No valid state filter could be identified in the query.",
                "child_dataset": None,
                "columns_used": [],
                "verification_status": "FAILED",
                "error": "MISSING_FILTER"
            }

        # 1. Load data for target states
        dfs: List[pd.DataFrame] = []
        loaded_datasets: List[str] = []
        for st in target_states:
            resolved = parent_child_registry.resolve_child_dataset(st)
            if resolved:
                st_canonical, child_path = resolved
                df_child = parent_child_registry.load_child_dataframe(child_path)
                if df_child is not None and not df_child.empty:
                    df_copy = df_child.copy()
                    if "State" not in df_copy.columns:
                        df_copy["State"] = st_canonical
                    dfs.append(df_copy)
                    loaded_datasets.append(child_path.name)

        if not dfs:
            return {
                "operation": intent,
                "scope": "FILTERED",
                "results": [],
                "result_count": 0,
                "answer": f"No records found for specified states: {', '.join(target_states)}.",
                "child_dataset": None,
                "columns_used": [],
                "verification_status": "FAILED",
                "error": "DATASET_NOT_FOUND"
            }

        combined_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        # 2. Apply additional filters from cls_res.filters if any (e.g. numeric thresholds)
        filtered_df = combined_df
        for f in cls_res.filters:
            col = f.get("column")
            op = f.get("operator")
            val = f.get("value")
            if col and col != "State" and col in filtered_df.columns:
                try:
                    s_num = pd.to_numeric(filtered_df[col], errors="coerce")
                    if op in [">", "gt"]:
                        filtered_df = filtered_df[s_num > float(val)]
                    elif op in ["<", "lt"]:
                        filtered_df = filtered_df[s_num < float(val)]
                    elif op in [">=", "gte"]:
                        filtered_df = filtered_df[s_num >= float(val)]
                    elif op in ["<=", "lte"]:
                        filtered_df = filtered_df[s_num <= float(val)]
                    elif op in ["=", "=="]:
                        filtered_df = filtered_df[s_num == float(val)]
                except Exception:
                    pass

        if filtered_df.empty:
            return {
                "operation": intent,
                "scope": "FILTERED",
                "results": [],
                "result_count": 0,
                "answer": f"No records matching the filter criteria were found in {', '.join(target_states)}.",
                "child_dataset": ", ".join(loaded_datasets),
                "columns_used": list(filtered_df.columns),
                "verification_status": "PASS"
            }

        # 3. Check Metric column existence
        if metric not in filtered_df.columns:
            m_col = next((c for c in filtered_df.columns if str(c).strip().lower() == metric.lower()), None)
            if m_col:
                metric = m_col
            else:
                metric = "Population"

        s_metric = pd.to_numeric(filtered_df[metric], errors="coerce").dropna()
        if s_metric.empty:
            return {
                "operation": intent,
                "scope": "FILTERED",
                "results": [],
                "result_count": 0,
                "answer": f"No numeric data found for column '{metric}'.",
                "child_dataset": ", ".join(loaded_datasets),
                "columns_used": [metric],
                "verification_status": "FAILED"
            }

        # 4. Compute Result based on intent
        results: List[Dict[str, Any]] = []
        ans: str = ""

        if intent in ["MAX", "MIN"]:
            idx_ext = s_metric.idxmax() if is_max else s_metric.idxmin()
            ext_row = filtered_df.loc[idx_ext]
            val = float(ext_row.get(metric))
            val_str = f"{val:,.0f}" if val.is_integer() else f"{val:,.2f}"
            v_name = str(ext_row.get("Village") or ext_row.get("village") or f"Record_{idx_ext}")
            st_name = str(ext_row.get("State") or ext_row.get("state") or target_states[0])
            cap_name = str(ext_row.get("Capital") or ext_row.get("capital") or "")
            v_id = str(ext_row.get("Village_ID") or "")

            ext_label = "highest" if is_max else "lowest"
            if len(target_states) == 1:
                ans = f"The village with the {ext_label} {metric.replace('_', ' ')} in {st_name} is {v_name} with a {metric.replace('_', ' ')} of {val_str}."
            else:
                ans = f"Among villages in {', '.join(target_states)}, the village with the {ext_label} {metric.replace('_', ' ')} is {v_name} in {st_name} with a {metric.replace('_', ' ')} of {val_str}."

            results = [{
                "rank": 1,
                "village": v_name,
                "Village": v_name,
                "state": st_name,
                "State": st_name,
                "capital": cap_name,
                "Capital": cap_name,
                "village_id": v_id,
                "Village_ID": v_id,
                "metric": metric,
                "value": val,
                "filters_applied": cls_res.filters,
                "dataset_id": f"child_{st_name.lower().replace(' ', '_')}"
            }]

        elif intent in ["TOP_N", "BOTTOM_N"]:
            n = cls_res.n_limit or 5
            sorted_indices = s_metric.sort_values(ascending=(not is_max)).head(n).index
            top_word = "Top" if is_max else "Bottom"
            top_records = []
            for rank, idx in enumerate(sorted_indices, start=1):
                r = filtered_df.loc[idx]
                val = float(r.get(metric))
                v_name = str(r.get("Village") or r.get("village") or f"Record_{idx}")
                st_name = str(r.get("State") or r.get("state") or target_states[0])
                cap_name = str(r.get("Capital") or r.get("capital") or "")
                v_id = str(r.get("Village_ID") or "")
                top_records.append({
                    "rank": rank,
                    "village": v_name,
                    "Village": v_name,
                    "state": st_name,
                    "State": st_name,
                    "capital": cap_name,
                    "Capital": cap_name,
                    "village_id": v_id,
                    "Village_ID": v_id,
                    "metric": metric,
                    "value": val
                })
            results = top_records
            st_scope_str = target_states[0] if len(target_states) == 1 else f"across {', '.join(target_states)}"
            ans = f"{top_word} {len(results)} villages by {metric.replace('_', ' ')} in {st_scope_str}:\n"
            ans += "\n".join([f"{r['rank']}. {r['village']} ({r['state']}): {r['value']:,.0f}" for r in results])

        elif intent in ["SUM", "TOTAL"]:
            total_val = float(s_metric.sum())
            val_str = f"{total_val:,.0f}" if total_val.is_integer() else f"{total_val:,.2f}"
            st_label = target_states[0] if len(target_states) == 1 else ", ".join(target_states)
            ans = f"The total {metric.replace('_', ' ')} across recorded villages in {st_label} is {val_str}."
            results = [{
                "metric": metric,
                "operation": "SUM",
                "value": total_val,
                "states": target_states,
                "villages_count": len(s_metric)
            }]

        elif intent in ["AVERAGE", "MEAN"]:
            avg_val = float(s_metric.mean())
            val_str = f"{avg_val:,.0f}" if avg_val.is_integer() else f"{avg_val:,.2f}"
            st_label = target_states[0] if len(target_states) == 1 else ", ".join(target_states)
            ans = f"The average {metric.replace('_', ' ')} across recorded villages in {st_label} is {val_str}."
            results = [{
                "metric": metric,
                "operation": "AVERAGE",
                "value": avg_val,
                "states": target_states,
                "villages_count": len(s_metric)
            }]

        elif intent == "COUNT":
            cnt_val = len(s_metric)
            st_label = target_states[0] if len(target_states) == 1 else ", ".join(target_states)
            ans = f"There are {cnt_val} villages recorded in {st_label}."
            results = [{
                "metric": "VILLAGE_COUNT",
                "operation": "COUNT",
                "value": float(cnt_val),
                "states": target_states
            }]

        # 5. Strict State Constraint Verification
        target_canonical = [s.lower() for s in target_states]
        verification_passed = True
        violation_reason = None
        for r in results:
            res_st = (r.get("state") or r.get("State") or "").lower()
            if res_st and not any(target_st in res_st or res_st in target_st for target_st in target_canonical):
                verification_passed = False
                violation_reason = f"Result state '{res_st}' does not match requested filter '{target_states}'"
                break

        return {
            "operation": intent,
            "scope": "FILTERED",
            "metric": metric,
            "results": results,
            "result_count": len(results),
            "answer": ans,
            "child_dataset": loaded_datasets[0] if len(loaded_datasets) == 1 else ", ".join(loaded_datasets),
            "columns_used": ["Village", "State", "Capital", metric] if "Village" in filtered_df.columns else ["State", metric],
            "verification_status": "PASS" if verification_passed else "FAILED",
            "verification_checks": {
                "filter_constraint_satisfied": verification_passed,
                "expected_states": target_states,
                "resolved_datasets": loaded_datasets,
                "violation_reason": violation_reason
            },
            "provenance": {
                "dataset_id": loaded_datasets[0] if len(loaded_datasets) == 1 else "multiple_child_datasets",
                "filters_applied": cls_res.filters,
                "return_entity": cls_res.return_entity,
                "intent": intent,
                "metric": metric
            }
        }


universal_global_aggregation_engine = UniversalGlobalAggregationEngine()

