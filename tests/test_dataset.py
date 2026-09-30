"""Unit tests for dataset loading, Excel/CSV ingestion, profiling, and validation."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest
from app.dataset.loader import DatasetLoader
from app.dataset.profiler import dataset_profiler
from app.dataset.metadata import DatasetMetadata
from app.dataset.validator import validate_dataframe, validate_dataset_file


def test_excel_loading():
    xlsx_path = Path("data/sample_customer_data.xlsx")
    if not xlsx_path.exists():
        pytest.skip("sample_customer_data.xlsx was removed per user dataset cleanup")
    loader = DatasetLoader(initial_path=str(xlsx_path))
    df = loader.dataframe
    assert not df.empty
    assert len(df) == 100
    assert "Customer Name" in df.columns
    assert "Service Amount" in df.columns
    assert "City" in df.columns


def test_csv_loading():
    csv_path = Path(__file__).resolve().parent.parent / "data" / "indian_states_capitals.csv"
    loader = DatasetLoader(initial_path=str(csv_path))
    df = loader.dataframe
    assert not df.empty
    assert len(df) == 28
    assert "state" in df.columns
    assert "capital" in df.columns


def test_dataset_profiling():
    xlsx_path = Path("data/sample_customer_data.xlsx")
    if not xlsx_path.exists():
        pytest.skip("sample_customer_data.xlsx was removed per user dataset cleanup")
    loader = DatasetLoader(initial_path=str(xlsx_path))
    df = loader.dataframe
    profile = dataset_profiler.profile(df, "sample_customer_data.xlsx")
    assert profile["basic_info"]["row_count"] == 100
    assert profile["basic_info"]["col_count"] == 8
    assert len(profile["columns"]) == 8

    # Verify numeric column profiling
    amt_col = [c for c in profile["columns"] if c["name"] == "Service Amount"][0]
    assert amt_col["type"] == "numeric"
    assert "min" in amt_col["stats"]
    assert "max" in amt_col["stats"]
    assert amt_col["stats"]["max"] >= amt_col["stats"]["min"]


def test_compact_schema_generation():
    xlsx_path = Path("data/sample_customer_data.xlsx")
    if not xlsx_path.exists():
        pytest.skip("sample_customer_data.xlsx was removed per user dataset cleanup")
    loader = DatasetLoader(initial_path=str(xlsx_path))
    profile = loader.profile
    schema = DatasetMetadata.get_compact_schema(profile["columns"], loader.dataset_name, len(loader.dataframe))
    assert schema["dataset_name"] == loader.dataset_name
    assert schema["row_count"] == 100
    assert len(schema["columns"]) == 8


def test_column_alias_matching():
    cols = ["Customer Name", "City", "Service Amount", "Service Date", "Warranty Expiry"]
    assert DatasetMetadata.match_column("service cost", cols) == "Service Amount"
    assert DatasetMetadata.match_column("amount paid", cols) == "Service Amount"
    assert DatasetMetadata.match_column("customer", cols) == "Customer Name"
    assert DatasetMetadata.match_column("location", cols) == "City"
    assert DatasetMetadata.match_column("non_existent_column", cols) is None


def test_invalid_file_validation():
    val, msg = validate_dataset_file(Path("fake_file.pdf"))
    assert not val
    assert "Invalid file format" in msg or "not found" in msg
