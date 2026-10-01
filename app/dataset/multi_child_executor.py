"""Multi-Child Dataset Executor: executes cross-child aggregations, ranking, and comparisons
across all village child datasets in the hierarchical parent-child architecture.
"""

import threading
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import pandas as pd

from app.dataset.registry import parent_child_registry
from app.utils.logger import logger


class MultiChildExecutor:
    """Executes cross-dataset queries across all registered village datasets."""

    def __init__(self):
        self._lock = threading.RLock()
        self._combined_df: Optional[pd.DataFrame] = None
        self._combined_hash: str = ""

    def get_combined_dataframe(self, force_reload: bool = False) -> pd.DataFrame:
        """Load and concatenate all registered child datasets into a unified DataFrame."""
        with self._lock:
            if self._combined_df is not None and not force_reload and not self._combined_df.empty:
                return self._combined_df.copy()

            child_map = parent_child_registry.get_registered_child_datasets()
            if not child_map:
                return pd.DataFrame()

            frames: List[pd.DataFrame] = []
            for state_or_capital, path in child_map.items():
                df = parent_child_registry.load_child_dataframe(path)
                if df is not None and not df.empty:
                    frames.append(df)

            if not frames:
                try:
                    from app.database.repositories import state_village_repo
                    db_df = state_village_repo.get_all_villages_dataframe()
                    if not db_df.empty:
                        frames.append(db_df)
                except Exception:
                    pass

            if not frames:
                self._combined_df = pd.DataFrame()
                return self._combined_df

            combined = pd.concat(frames, ignore_index=True)
            # Ensure numeric columns are properly converted
            numeric_cols = ["Population", "No_of_Males", "No_of_Females", "Literacy_Rate_Percent", "Area_Sq_Km", "Households"]
            for col in numeric_cols:
                if col in combined.columns:
                    combined[col] = pd.to_numeric(combined[col], errors="coerce")

            self._combined_df = combined
            logger.info(f"MultiChildExecutor: Combined {len(frames)} child datasets ({len(combined)} total rows).")
            return self._combined_df.copy()

    def find_extreme_village(
        self,
        metric_col: str,
        is_max: bool = True,
        filter_states: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Find the village with maximum or minimum of a specific metric across child datasets."""
        df = self.get_combined_dataframe()
        if df.empty or metric_col not in df.columns:
            return None

        filtered = df
        if filter_states:
            filter_lower = [s.strip().lower() for s in filter_states]
            if "State" in filtered.columns:
                filtered = filtered.loc[filtered["State"].astype(str).str.lower().isin(filter_lower)]

        if filtered.empty:
            return None

        sorted_df = cast(pd.DataFrame, filtered).sort_values(by=metric_col, ascending=not is_max).dropna(subset=[metric_col])
        if sorted_df.empty:
            return None

        top_row = sorted_df.iloc[0].to_dict()
        return top_row

    def aggregate_by_state(
        self,
        metric_col: str,
        agg_func: str = "sum",
        filter_states: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Aggregate village metric grouped by State or Capital."""
        df = self.get_combined_dataframe()
        if df.empty or metric_col not in df.columns or "State" not in df.columns:
            return []

        filtered = df
        if filter_states:
            filter_lower = [s.strip().lower() for s in filter_states]
            filtered = filtered[filtered["State"].astype(str).str.lower().isin(filter_lower)]

        if filtered.empty:
            return []

        grouped = filtered.groupby("State")[metric_col].agg(agg_func).reset_index()
        return grouped.to_dict(orient="records")

    def compare_states(
        self,
        state_a: str,
        state_b: str,
        metric_col: str,
        agg_func: str = "sum"
    ) -> Dict[str, Any]:
        """Compare aggregated metric between two states."""
        df = self.get_combined_dataframe()
        if df.empty or metric_col not in df.columns or "State" not in df.columns:
            return {"error": f"Metric {metric_col} or State column not found"}

        a_lower = state_a.strip().lower()
        b_lower = state_b.strip().lower()

        df_a = df[df["State"].astype(str).str.lower() == a_lower]
        df_b = df[df["State"].astype(str).str.lower() == b_lower]

        vals_a: List[float] = []
        for item in df_a[metric_col]:
            try:
                if pd.notna(item):
                    vals_a.append(float(item))
            except (ValueError, TypeError):
                pass

        vals_b: List[float] = []
        for item in df_b[metric_col]:
            try:
                if pd.notna(item):
                    vals_b.append(float(item))
            except (ValueError, TypeError):
                pass

        val_a = float(sum(vals_a) if agg_func == "sum" else (sum(vals_a) / len(vals_a) if vals_a else 0.0))
        val_b = float(sum(vals_b) if agg_func == "sum" else (sum(vals_b) / len(vals_b) if vals_b else 0.0))

        diff = val_a - val_b
        return {
            "state_a": state_a,
            "value_a": val_a,
            "state_b": state_b,
            "value_b": val_b,
            "metric": metric_col,
            "difference": diff,
            "higher": state_a if diff > 0 else (state_b if diff < 0 else "Equal")
        }

    def aggregate_all_states(
        self,
        metric_col: str,
        agg_func: str = "sum"
    ) -> List[Dict[str, Any]]:
        """Aggregate village metric across all registered states and return sorted results."""
        df = self.get_combined_dataframe()
        if df.empty or metric_col not in df.columns or "State" not in df.columns:
            return []

        grouped = df.groupby("State")[metric_col].agg(agg_func).reset_index()
        sorted_grouped = grouped.sort_values(by=metric_col, ascending=False)
        return sorted_grouped.to_dict(orient="records")

    def rank_states_by_metric(
        self,
        metric_col: str,
        agg_func: str = "sum",
        is_max: bool = True
    ) -> Dict[str, Any]:
        """Rank states by aggregated metric and return extreme state and full breakdown."""
        records = self.aggregate_all_states(metric_col, agg_func=agg_func)
        if not records:
            return {
                "metric": metric_col,
                "operation": agg_func.upper(),
                "top_state": None,
                "top_value": 0.0,
                "rankings": []
            }

        if not is_max:
            records = list(reversed(records))

        rankings: List[Dict[str, Any]] = []
        for idx, rec in enumerate(records, start=1):
            rankings.append({
                "rank": idx,
                "state": rec["State"],
                "value": float(rec[metric_col])
            })

        best = rankings[0]
        return {
            "metric": metric_col,
            "operation": agg_func.upper(),
            "is_max": is_max,
            "top_state": best["state"],
            "top_value": best["value"],
            "rankings": rankings
        }

    def rank_states_by_village_count(self, is_max: bool = True) -> Dict[str, Any]:
        """Rank states by the number of recorded villages."""
        df = self.get_combined_dataframe()
        if df.empty or "State" not in df.columns:
            return {
                "metric": "Villages_Count",
                "top_state": None,
                "top_value": 0,
                "rankings": []
            }

        counts = df.groupby("State").size().reset_index()
        counts.columns = ["State", "village_count"]
        counts_sorted = counts.sort_values(by=["village_count"], ascending=not is_max)
        records = counts_sorted.to_dict(orient="records")

        rankings: List[Dict[str, Any]] = []
        for idx, rec in enumerate(records, start=1):
            rankings.append({
                "rank": idx,
                "state": rec["State"],
                "value": int(rec["village_count"])
            })

        best = rankings[0] if rankings else {"state": None, "value": 0}
        return {
            "metric": "Villages_Count",
            "is_max": is_max,
            "top_state": best["state"],
            "top_value": best["value"],
            "rankings": rankings
        }

    def get_top_n_villages(
        self,
        metric_col: str,
        n: int = 10,
        is_max: bool = True,
        filter_states: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Return top N villages ranked by a metric across all states or filtered states."""
        df = self.get_combined_dataframe()
        if df.empty or metric_col not in df.columns:
            return []

        filtered = df
        if filter_states:
            filter_lower = [s.strip().lower() for s in filter_states]
            if "State" in filtered.columns:
                filtered = filtered.loc[filtered["State"].astype(str).str.lower().isin(filter_lower)]

        if filtered.empty:
            return []

        sorted_df = cast(pd.DataFrame, filtered).sort_values(by=metric_col, ascending=not is_max).dropna(subset=[metric_col])
        top_rows = sorted_df.head(n)

        results: List[Dict[str, Any]] = []
        for idx, (_, row) in enumerate(top_rows.iterrows(), start=1):
            r_dict = row.to_dict()
            results.append({
                "rank": idx,
                "village": r_dict.get("Village"),
                "village_id": r_dict.get("Village_ID"),
                "state": r_dict.get("State"),
                "capital": r_dict.get("Capital"),
                "metric": metric_col,
                "value": r_dict.get(metric_col)
            })

        return results


multi_child_executor = MultiChildExecutor()

