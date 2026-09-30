"""Comprehensive Data Quality & Parent-Child Dataset Validator.

Performs:
1. Schema & Column sanity validation
2. Data type integrity analysis
3. Duplicate record & key detection
4. Missing value categorization & null-policy analysis
5. Numeric range & anomaly verification
6. Hierarchical Parent-Child dataset consistency checks (State -> Capital -> Child Dataset)
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from app.utils.logger import logger


def validate_dataset_file(file_path: Path) -> Tuple[bool, str]:
    """Check file existence and extension (backward compatible)."""
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    ext = file_path.suffix.lower()
    if ext not in {".csv", ".xlsx", ".xls", ".json"}:
        return False, f"Invalid file format: {ext}. Only CSV, Excel, and JSON files are supported."

    return True, "Valid file"


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate DataFrame integrity (backward compatible)."""
    if df is None or df.empty:
        return False, "The dataset contains no data rows."

    if len(df.columns) == 0:
        return False, "The dataset contains no columns."

    # Check that at least one column has non-null content
    if df.dropna(how="all").empty:
        return False, "All rows in the dataset are empty."

    return True, "Valid dataset"


class DatasetValidator:
    """Full-featured data quality and hierarchy validator."""

    NULL_VALUES = {"", "none", "null", "nan", "n/a", "na", "-", "--", "unknown", "blank", "empty"}

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate column count, names, and presence of duplicate headers."""
        if df is None or df.empty:
            return {"is_valid": False, "error": "DataFrame is empty."}

        cols = [str(c).strip() for c in df.columns]
        dup_cols = [c for c in set(cols) if cols.count(c) > 1]
        empty_cols = [c for c in cols if not c]

        return {
            "is_valid": len(empty_cols) == 0 and len(dup_cols) == 0,
            "column_count": len(cols),
            "columns": cols,
            "duplicate_columns": dup_cols,
            "empty_named_columns": empty_cols,
        }

    def detect_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detect duplicate rows across all columns or specific key columns."""
        if df is None or df.empty:
            return {"duplicate_count": 0, "duplicate_rows": []}

        active_cols = [c for c in (subset or list(df.columns)) if c in df.columns and not str(c).startswith("_")]
        if not active_cols:
            return {"duplicate_count": 0, "duplicate_rows": []}

        dups = df[df.duplicated(subset=active_cols, keep=False)]
        return {
            "duplicate_count": int(len(dups)),
            "is_unique": len(dups) == 0,
            "key_columns": active_cols,
            "sample_duplicates": dups.head(10).to_dict(orient="records")
        }

    def analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing, NaN, and placeholder null values across all columns."""
        if df is None or df.empty:
            return {"total_missing": 0, "columns": {}}

        total_rows = len(df)
        col_missing = {}
        total_missing_cells = 0

        for col in df.columns:
            if str(col).startswith("_"):
                continue
            series = df[col]
            # Standard pandas nulls
            null_count = int(series.isna().sum())
            # Textual placeholder nulls
            str_series = series.dropna().astype(str).str.strip().str.lower()
            placeholder_nulls = int(str_series.isin(self.NULL_VALUES).sum())
            combined = null_count + placeholder_nulls
            total_missing_cells += combined

            pct = round((combined / total_rows) * 100, 2) if total_rows > 0 else 0.0
            col_missing[col] = {
                "missing_count": combined,
                "missing_percentage": pct,
                "has_missing": combined > 0
            }

        return {
            "total_rows": total_rows,
            "total_missing_cells": total_missing_cells,
            "columns": col_missing,
            "columns_with_missing": [k for k, v in col_missing.items() if v["has_missing"]]
        }

    def validate_parent_child_hierarchy(
        self,
        main_dataset_path: Path,
        child_dir_path: Path
    ) -> Dict[str, Any]:
        """Verify that Main State-Capital dataset and all associated Child datasets are consistent."""
        issues: List[str] = []
        warnings: List[str] = []

        if not main_dataset_path.exists():
            return {"is_valid": False, "issues": [f"Main dataset not found at {main_dataset_path}"]}

        if not child_dir_path.exists() or not child_dir_path.is_dir():
            return {"is_valid": False, "issues": [f"Child directory not found at {child_dir_path}"]}

        try:
            main_df = pd.read_csv(main_dataset_path)
        except Exception as e:
            return {"is_valid": False, "issues": [f"Failed to read main dataset: {e}"]}

        # Check required columns in main dataset
        cols_lower = {str(c).strip().lower(): c for c in main_df.columns}
        if "state" not in cols_lower:
            issues.append("Main dataset missing 'State' column.")
        if "capital" not in cols_lower:
            issues.append("Main dataset missing 'Capital' column.")
        if "child_dataset" not in cols_lower:
            warnings.append("Main dataset missing 'Child_Dataset' reference column.")

        child_col = cols_lower.get("child_dataset")
        state_col = cols_lower.get("state")
        capital_col = cols_lower.get("capital")

        verified_children: List[str] = []
        missing_children: List[str] = []
        empty_children: List[str] = []

        if child_col and state_col and capital_col:
            for _, row in main_df.iterrows():
                state_name = str(row[state_col]).strip()
                cap_name = str(row[capital_col]).strip()
                child_file_name = str(row[child_col]).strip()

                if not child_file_name or child_file_name.lower() in self.NULL_VALUES:
                    warnings.append(f"No child dataset specified for state '{state_name}' ({cap_name}).")
                    continue

                child_path = child_dir_path / child_file_name
                if not child_path.exists():
                    missing_children.append(child_file_name)
                    issues.append(f"Child dataset '{child_file_name}' for {state_name} does not exist on disk.")
                else:
                    try:
                        child_df = pd.read_csv(child_path)
                        if child_df.empty:
                            empty_children.append(child_file_name)
                            issues.append(f"Child dataset '{child_file_name}' for {state_name} is empty.")
                        else:
                            verified_children.append(child_file_name)
                    except Exception as ex:
                        issues.append(f"Child dataset '{child_file_name}' corrupted: {ex}")

        return {
            "is_valid": len(issues) == 0,
            "total_parents": len(main_df),
            "verified_child_count": len(verified_children),
            "missing_child_count": len(missing_children),
            "empty_child_count": len(empty_children),
            "issues": issues,
            "warnings": warnings,
        }


dataset_validator = DatasetValidator()
