"""Schema Detector: Infers data types and semantic roles of dataset columns."""

import re
from typing import Any, Dict, List, Optional
import pandas as pd


# Semantic patterns mapped to canonical semantic types
SEMANTIC_PATTERNS = {
    "STATE": [r"^state(_name)?$", r"^province$", r"^state_ut$"],
    "CAPITAL": [r"^capital(_city)?$", r"^headquarter(s)?$"],
    "VILLAGE": [r"^village(_name)?$", r"^settlement$", r"^locality$", r"^town$"],
    "POPULATION": [r"^population$", r"^total_population$", r"^pop$", r"^residents$", r"^inhabitants$"],
    "MALE_COUNT": [r"^no_of_males$", r"^males?$", r"^male_population$", r"^male_count$"],
    "FEMALE_COUNT": [r"^no_of_females$", r"^females?$", r"^female_population$", r"^female_count$"],
    "LITERACY_RATE": [r"^literacy(_rate)?(_percent)?$", r"^literacy_pct$", r"^literacy_percentage$"],
    "AREA": [r"^area(_sq_km)?$", r"^area_km2$", r"^area_in_sq_km$", r"^geographical_area$"],
    "HOUSEHOLDS": [r"^(no_of_)?households?$", r"^total_households?$"],
    "CODE": [r"^state_code$", r"^code$", r"^pin_code$", r"^postal_code$"],
    "IDENTIFIER": [r"^id$", r"^[a-z_]+_id$"],
}


class SchemaDetector:
    """Detects column types, semantic types, and query capabilities."""

    @staticmethod
    def clean_column_name(col: str) -> str:
        """Normalize a column name for matching."""
        s = str(col).strip().lower()
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\s+", "_", s)
        return s

    @classmethod
    def detect_semantic_type(cls, col_name: str, sample_series: Optional[pd.Series] = None) -> str:
        """Infer the semantic type of a column by name and sample values."""
        clean_name = cls.clean_column_name(col_name)

        for sem_type, patterns in SEMANTIC_PATTERNS.items():
            for pat in patterns:
                if re.match(pat, clean_name):
                    return sem_type

        # Check sample data if available
        if sample_series is not None and not sample_series.dropna().empty:
            if pd.api.types.is_numeric_dtype(sample_series):
                return "GENERAL_NUMERIC"
            return "GENERAL_TEXT"

        return "GENERAL_TEXT"

    @classmethod
    def detect_data_type(cls, series: pd.Series) -> str:
        """Detect the SQL-compatible storage data type."""
        non_null = series.dropna()
        if non_null.empty:
            return "string"

        if pd.api.types.is_bool_dtype(non_null):
            return "boolean"
        if pd.api.types.is_integer_dtype(non_null):
            return "integer"
        if pd.api.types.is_float_dtype(non_null):
            return "float"
        if pd.api.types.is_datetime64_any_dtype(non_null):
            return "timestamp"

        # Check if object series actually contains numbers formatted as strings
        try:
            converted = pd.to_numeric(non_null.astype(str).str.replace(",", "").str.rstrip("%"), errors="coerce")
            if converted.notnull().mean() > 0.8:
                if (converted.dropna() % 1 == 0).all():
                    return "integer"
                return "float"
        except Exception:
            pass

        return "string"

    @classmethod
    def analyze_dataframe(cls, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Inspect all columns of a dataframe and return schema metadata."""
        schema_info = []
        for col in df.columns:
            series = df[col]
            clean_name = cls.clean_column_name(col)
            sem_type = cls.detect_semantic_type(col, series)
            dtype = cls.detect_data_type(series)

            is_numeric = dtype in ("integer", "float") or sem_type in (
                "POPULATION", "MALE_COUNT", "FEMALE_COUNT", "LITERACY_RATE", "AREA", "HOUSEHOLDS"
            )
            is_aggregatable = is_numeric and sem_type not in ("IDENTIFIER", "CODE")

            schema_info.append({
                "original_name": str(col),
                "normalized_name": clean_name,
                "semantic_type": sem_type,
                "data_type": dtype,
                "is_numeric": is_numeric,
                "is_filterable": True,
                "is_aggregatable": is_aggregatable,
            })
        return schema_info


schema_detector = SchemaDetector()
