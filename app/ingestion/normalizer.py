"""Normalizer: Cleans and standardizes raw values, strings, numbers, and dates."""

import re
from typing import Any, Optional, Union
import numpy as np
import pandas as pd


class DataNormalizer:
    """Standardizes tabular row values for database insertion."""

    @staticmethod
    def clean_string(val: Any) -> Optional[str]:
        if val is None or pd.isna(val):
            return None
        s = str(val).strip()
        # Remove multiple spaces and control characters
        s = re.sub(r"[\r\n\t]+", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s if s else None

    @staticmethod
    def parse_numeric(val: Any, default: float = 0.0) -> float:
        if val is None or pd.isna(val):
            return default
        if isinstance(val, (int, float)):
            if np.isnan(val) or np.isinf(val):
                return default
            return float(val)

        s = str(val).strip().replace(",", "")
        # Remove percentage signs if present
        s = s.rstrip("%").strip()
        # Extract the numeric component if surrounded by text
        match = re.search(r"[-+]?(?:\d*\.\d+|\d+)", s)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return default
        return default

    @classmethod
    def normalize_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Deep copy and clean dataframe values."""
        clean_df = df.copy()

        for col in clean_df.columns:
            # Check if this column is expected to be numeric
            col_lower = str(col).lower()
            if any(num_kw in col_lower for num_kw in ["population", "males", "females", "literacy", "area", "households"]):
                clean_df[col] = clean_df[col].apply(lambda x: cls.parse_numeric(x, default=0.0))
            else:
                # String / Categorical column
                clean_df[col] = clean_df[col].apply(lambda x: cls.clean_string(x))

        return clean_df


normalizer = DataNormalizer()
