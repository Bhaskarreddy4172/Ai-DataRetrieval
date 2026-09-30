"""Dynamic, dataset-agnostic suggestion and summary engine.

Inspects DataFrame schema, data types, statistical distributions, and real values
to formulate high-value suggested natural language questions and summaries
without any hardcoding or domain assumptions.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from app.utils.logger import logger


class DatasetSuggestionEngine:
    """Generates dynamic natural language questions and dataset summaries based on actual data."""

    @staticmethod
    def generate_suggestions(
        df: pd.DataFrame,
        dataset_name: str = "",
        profile: Optional[Dict[str, Any]] = None,
        max_suggestions: int = 6,
    ) -> List[str]:
        """Generate 4-6 dynamic, diverse suggested questions for any uploaded dataset."""
        if df is None or df.empty:
            return ["Upload a dataset with records to see suggested questions."]

        cols = [c for c in df.columns if c != "_internal_row_id"]
        if not cols:
            return []

        suggestions: List[str] = []
        row_count = len(df)

        # Categorize columns
        metric_cols: List[str] = []
        entity_cols: List[str] = []
        category_cols: List[str] = []
        date_cols: List[str] = []

        # If profile provided, use semantic tags
        if profile and "columns" in profile:
            for c_info in profile["columns"]:
                name = c_info.get("name")
                if not name or name not in cols:
                    continue
                tag = c_info.get("semantic_tag", "")
                inferred_type = c_info.get("type", "")
                unique_count = c_info.get("unique_count", 0)

                if tag == "metric" or inferred_type == "numeric":
                    if tag != "identifier":
                        metric_cols.append(name)
                elif tag == "temporal" or inferred_type == "date":
                    date_cols.append(name)
                elif tag in ("location", "entity_name") or unique_count > min(10, row_count * 0.3):
                    entity_cols.append(name)
                else:
                    category_cols.append(name)
        else:
            # Infer directly from DataFrame dtypes
            for col in cols:
                series = df[col]
                col_l = col.lower()
                if pd.api.types.is_numeric_dtype(series):
                    if not any(id_w in col_l for id_w in ["id", "code", "zip", "pin"]):
                        metric_cols.append(col)
                    else:
                        entity_cols.append(col)
                elif pd.api.types.is_datetime64_any_dtype(series):
                    date_cols.append(col)
                else:
                    unique_count = series.dropna().nunique()
                    if any(loc_w in col_l for loc_w in ["city", "state", "country", "name", "title"]):
                        entity_cols.append(col)
                    elif unique_count <= 15:
                        category_cols.append(col)
                    else:
                        entity_cols.append(col)

        # 1. Pair relationship lookup (e.g. entity + attribute/metric/category)
        if entity_cols and len(cols) >= 2:
            primary_entity = entity_cols[0]
            # Pick a target column that isn't the primary entity
            secondary_candidates = [c for c in cols if c != primary_entity]
            if secondary_candidates:
                target_col = secondary_candidates[0]
                # Sample a real value from primary_entity
                sample_vals = df[primary_entity].dropna().unique()
                if len(sample_vals) > 0:
                    sample_val = str(sample_vals[0]).strip()
                    suggestions.append(f"What is the {target_col} of {sample_val}?")

        # 2. Extreme / Ranking question (Max / Min)
        if metric_cols:
            primary_metric = metric_cols[0]
            if entity_cols:
                primary_entity = entity_cols[0]
                suggestions.append(f"Which {primary_entity} has the highest {primary_metric}?")
            else:
                suggestions.append(f"What is the maximum {primary_metric}?")

        # 3. Aggregation question (Average / Total)
        if metric_cols:
            primary_metric = metric_cols[0]
            suggestions.append(f"What is the average {primary_metric}?")

        # 4. Filter / Conditional question
        if category_cols:
            cat_col = category_cols[0]
            cat_vals = df[cat_col].dropna().unique()
            if len(cat_vals) > 0:
                cat_val = str(cat_vals[0]).strip()
                suggestions.append(f"Show records where {cat_col} is '{cat_val}'")
        elif len(metric_cols) > 0:
            primary_metric = metric_cols[0]
            mean_val = df[primary_metric].dropna().mean()
            if pd.notnull(mean_val):
                rounded_val = round(float(mean_val), 1) if isinstance(mean_val, (int, float)) else mean_val
                suggestions.append(f"Show records where {primary_metric} is greater than {rounded_val}")

        # 5. Reverse lookup if two text/entity columns exist
        if len(entity_cols) >= 2:
            col1 = entity_cols[0]
            col2 = entity_cols[1]
            sample_vals = df[col2].dropna().unique()
            if len(sample_vals) > 0:
                sample_val = str(sample_vals[0]).strip()
                suggestions.append(f"Which {col1} corresponds to {sample_val}?")

        # 6. Counting / Summary questions
        if entity_cols:
            primary_entity = entity_cols[0]
            suggestions.append(f"How many distinct {primary_entity} are in the dataset?")
        else:
            suggestions.append(f"How many total records are in this dataset?")

        # 7. Date-based question if temporal columns exist
        if date_cols:
            date_col = date_cols[0]
            suggestions.append(f"What is the earliest and latest {date_col}?")

        # Fallback if list is short
        if len(suggestions) < 3:
            for col in cols[:3]:
                q = f"List all distinct values for {col}"
                if q not in suggestions:
                    suggestions.append(q)

        # De-duplicate while preserving order and limit to max_suggestions
        seen = set()
        final_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                final_suggestions.append(s)
                if len(final_suggestions) >= max_suggestions:
                    break

        return final_suggestions

    @staticmethod
    def generate_summary(
        df: pd.DataFrame,
        dataset_name: str = "",
        profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a natural language overview of the uploaded dataset."""
        if df is None or df.empty:
            return "No active dataset loaded. Please upload a CSV, Excel, or JSON file to begin."

        cols = [c for c in df.columns if c != "_internal_row_id"]
        row_count = len(df)
        col_count = len(cols)
        name_display = dataset_name or "Uploaded Dataset"

        # Categorize columns
        metric_cols = []
        category_cols = []
        date_cols = []
        entity_cols = []

        for col in cols:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                if not any(id_w in col.lower() for id_w in ["id", "code", "zip", "pin"]):
                    metric_cols.append(col)
                else:
                    entity_cols.append(col)
            elif pd.api.types.is_datetime64_any_dtype(series):
                date_cols.append(col)
            else:
                unique_count = series.dropna().nunique()
                if unique_count <= 15:
                    category_cols.append(col)
                else:
                    entity_cols.append(col)

        parts = [
            f"**{name_display}** has been loaded successfully with **{row_count:,} records** and **{col_count} columns**."
        ]

        fields_summary = f"Fields: `{', '.join(cols[:8])}`"
        if len(cols) > 8:
            fields_summary += f" and {len(cols) - 8} more."
        parts.append(fields_summary)

        insights = []
        if entity_cols:
            sample_entity_vals = df[entity_cols[0]].dropna().head(3).tolist()
            sample_str = ", ".join(f"'{v}'" for v in sample_entity_vals)
            insights.append(f"Key entities include **{entity_cols[0]}** (e.g., {sample_str})")

        if metric_cols:
            insights.append(f"Numeric metrics available: **{', '.join(metric_cols[:4])}**")

        if category_cols:
            insights.append(f"Categories: **{', '.join(category_cols[:3])}**")

        if date_cols:
            insights.append(f"Time dimension: **{', '.join(date_cols[:2])}**")

        if insights:
            parts.append(". ".join(insights) + ".")

        parts.append("You can ask any natural language question about these records below.")

        return "\n\n".join(parts)


dataset_suggestion_engine = DatasetSuggestionEngine()

