"""Column Mapper: Maps dataset header variations to canonical database column names."""

import re
from typing import Dict, List, Optional
import pandas as pd


CANONICAL_MAPPINGS = {
    "state": [
        "state", "state_name", "statename", "states", "province", "state_ut", "state_title"
    ],
    "capital": [
        "capital", "capital_city", "capitalcity", "capitals", "headquarter", "headquarters"
    ],
    "state_code": [
        "state_code", "code", "statecode", "st_code", "iso_code", "abbr", "abbreviation"
    ],
    "village": [
        "village", "village_name", "villagename", "villages", "settlement", "locality", "town"
    ],
    "population": [
        "population", "total_population", "totalpopulation", "pop", "residents", "inhabitants"
    ],
    "no_of_males": [
        "no_of_males", "males", "male", "male_population", "male_count", "total_males", "no_of_male"
    ],
    "no_of_females": [
        "no_of_females", "females", "female", "female_population", "female_count", "total_females", "no_of_female"
    ],
    "literacy_rate_percent": [
        "literacy_rate_percent", "literacy_rate", "literacy", "literacy_pct", "literacy_percentage",
        "literacy_rate_%", "literacy_percent", "literacy_%"
    ],
    "area_sq_km": [
        "area_sq_km", "area", "area_km2", "area_in_sq_km", "geographical_area", "area_(sq_km)", "area_sqkm"
    ],
    "households": [
        "households", "no_of_households", "total_households", "household_count", "hh_count"
    ],
}


class ColumnMapper:
    """Normalizes raw tabular headers to canonical database attributes."""

    @staticmethod
    def _simplify(name: str) -> str:
        s = str(name).strip().lower()
        s = re.sub(r"[^\w]", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    @classmethod
    def get_canonical_name(cls, col_name: str) -> str:
        simplified = cls._simplify(col_name)

        for canonical, aliases in CANONICAL_MAPPINGS.items():
            if simplified == canonical or simplified in aliases:
                return canonical
            for alias in aliases:
                if simplified == cls._simplify(alias):
                    return canonical

        return simplified

    @classmethod
    def standardize_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Rename dataframe columns to canonical names without modifying original data."""
        rename_dict = {}
        for col in df.columns:
            canonical = cls.get_canonical_name(str(col))
            rename_dict[col] = canonical

        renamed_df = df.rename(columns=rename_dict)
        return renamed_df


column_mapper = ColumnMapper()
