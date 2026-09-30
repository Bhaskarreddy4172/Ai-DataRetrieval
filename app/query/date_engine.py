"""Deterministic Date Parsing, Comparison, and Interval Engine.

Supports:
- Parsing arbitrary date formats (ISO, European, US, month name)
- Temporal operators: BEFORE, AFTER, BETWEEN, DURING, LATEST, EARLIEST
- Interval calculations: days_between, months_between, years_between
"""

import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd


class DateEngine:
    """Deterministic date processor for dataset queries."""

    DATE_FORMATS = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%Y",
    ]

    def parse_date(self, val: Any) -> Optional[date]:
        """Convert arbitrary string, datetime, or date to a Python date object."""
        if val is None or pd.isna(val):
            return None
        if isinstance(val, date) and not isinstance(val, datetime):
            return val
        if isinstance(val, (datetime, pd.Timestamp)):
            return val.date()

        s = str(val).strip()
        if not s:
            return None

        # Try standard formats
        for fmt in self.DATE_FORMATS:
            try:
                dt = datetime.strptime(s, fmt)
                return dt.date()
            except ValueError:
                continue

        # Try dateparser if available
        try:
            import dateparser
            parsed = dateparser.parse(s)
            if parsed:
                return parsed.date()
        except Exception:
            pass

        return None

    def days_between(self, d1: Any, d2: Any) -> Optional[int]:
        """Calculate signed days between d1 and d2 (d2 - d1)."""
        dt1 = self.parse_date(d1)
        dt2 = self.parse_date(d2)
        if not dt1 or not dt2:
            return None
        return (dt2 - dt1).days

    def months_between(self, d1: Any, d2: Any) -> Optional[int]:
        """Calculate approximate months between d1 and d2."""
        dt1 = self.parse_date(d1)
        dt2 = self.parse_date(d2)
        if not dt1 or not dt2:
            return None
        return (dt2.year - dt1.year) * 12 + (dt2.month - dt1.month)

    def years_between(self, d1: Any, d2: Any) -> Optional[int]:
        """Calculate complete years between d1 and d2."""
        dt1 = self.parse_date(d1)
        dt2 = self.parse_date(d2)
        if not dt1 or not dt2:
            return None
        return dt2.year - dt1.year

    def filter_date_column(
        self,
        df: pd.DataFrame,
        date_col: str,
        operator: str,
        target_val: Any,
        target_val2: Optional[Any] = None
    ) -> pd.DataFrame:
        """Filter DataFrame by temporal operator: BEFORE (<), AFTER (>), EQUAL (=), BETWEEN."""
        if date_col not in df.columns or df.empty:
            return df

        target_dt = self.parse_date(target_val)
        if not target_dt:
            return df

        parsed_series = df[date_col].apply(self.parse_date)
        valid_mask = parsed_series.notna()

        op = operator.upper().strip()
        if op in {"BEFORE", "<", "EARLIER"}:
            mask = valid_mask & (parsed_series < target_dt)
        elif op in {"AFTER", ">", "LATER"}:
            mask = valid_mask & (parsed_series > target_dt)
        elif op in {"EQUAL", "=", "ON", "DURING"}:
            mask = valid_mask & (parsed_series == target_dt)
        elif op in {"BETWEEN"} and target_val2 is not None:
            target_dt2 = self.parse_date(target_val2)
            if target_dt2:
                lower = min(target_dt, target_dt2)
                upper = max(target_dt, target_dt2)
                mask = valid_mask & (parsed_series >= lower) & (parsed_series <= upper)
            else:
                mask = valid_mask
        else:
            mask = valid_mask

        return df[mask]

    def find_extreme_date_row(
        self,
        df: pd.DataFrame,
        date_col: str,
        find_latest: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Identify the record with the earliest or latest date."""
        if date_col not in df.columns or df.empty:
            return None

        parsed_series = df[date_col].apply(self.parse_date)
        valid_indices = parsed_series.dropna().index
        if valid_indices.empty:
            return None

        extreme_idx = parsed_series.loc[valid_indices].idxmax() if find_latest else parsed_series.loc[valid_indices].idxmin()
        return df.loc[extreme_idx].to_dict()


date_engine = DateEngine()

