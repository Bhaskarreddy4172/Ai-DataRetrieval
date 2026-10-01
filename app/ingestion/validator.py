"""Data Ingestion Validator: Validates rows and columns against domain constraints."""

from typing import Any, Dict, List, Optional
import pandas as pd


class IngestionValidator:
    """Validates dataframe integrity, bounds, and entity relationships before DB loading."""

    @classmethod
    def validate_dataset(
        cls,
        df: pd.DataFrame,
        dataset_type: str = "CHILD",
        dataset_name: str = ""
    ) -> Dict[str, Any]:
        issues = []
        status = "PASS"
        total_rows = len(df)

        if total_rows == 0:
            return {
                "status": "FAIL",
                "issues": ["Dataset is empty (0 rows)."],
                "row_count": 0,
                "valid_count": 0,
            }

        # Check required columns based on dataset type
        cols_lower = [str(c).lower() for c in df.columns]

        if dataset_type == "MAIN":
            if not any("state" in c for c in cols_lower):
                issues.append("Missing required 'State' column in main dataset.")
                status = "FAIL"
            if not any("capital" in c for c in cols_lower):
                issues.append("Missing required 'Capital' column in main dataset.")
                status = "WARNING" if status != "FAIL" else status

        elif dataset_type == "CHILD":
            if not any("village" in c for c in cols_lower):
                issues.append("Missing required 'Village' column in child dataset.")
                status = "FAIL"

        # Check numeric bounds on standard demographic columns
        for col in df.columns:
            col_low = str(col).lower()
            series = df[col]

            if any(k in col_low for k in ["population", "area", "households", "males", "females"]):
                # Ensure non-negative
                numeric_vals = pd.to_numeric(series, errors="coerce").dropna()
                negatives = (numeric_vals < 0).sum()
                if negatives > 0:
                    issues.append(f"Column '{col}' contains {negatives} negative values.")
                    status = "WARNING" if status != "FAIL" else status

            if "literacy" in col_low:
                numeric_vals = pd.to_numeric(series, errors="coerce").dropna()
                out_of_bounds = ((numeric_vals < 0) | (numeric_vals > 100)).sum()
                if out_of_bounds > 0:
                    issues.append(f"Literacy column '{col}' has {out_of_bounds} values outside 0-100 range.")
                    status = "WARNING" if status != "FAIL" else status

        # Check male + female consistency if both present
        male_col = next((c for c in df.columns if "male" in str(c).lower() and "female" not in str(c).lower()), None)
        female_col = next((c for c in df.columns if "female" in str(c).lower()), None)
        pop_col = next((c for c in df.columns if str(c).lower() == "population" or "total_pop" in str(c).lower()), None)

        if male_col and female_col and pop_col:
            m = pd.to_numeric(df[male_col], errors="coerce").fillna(0)
            f = pd.to_numeric(df[female_col], errors="coerce").fillna(0)
            tot = pd.to_numeric(df[pop_col], errors="coerce").fillna(0)
            sum_mf = m + f
            # Check rows where total is non-zero but variance > 10%
            nonzero = tot > 0
            if nonzero.any():
                diff_pct = (sum_mf[nonzero] - tot[nonzero]).abs() / tot[nonzero]
                high_diff = (diff_pct > 0.10).sum()
                if high_diff > 0:
                    issues.append(f"Notice: {high_diff} rows have sum of males+females differing from population by >10%.")
                    status = "WARNING" if status != "FAIL" else status

        return {
            "status": status,
            "issues": issues,
            "row_count": total_rows,
            "valid_count": total_rows,
            "columns": list(df.columns),
        }


ingestion_validator = IngestionValidator()
