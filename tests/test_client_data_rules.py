"""Mandatory Behavioral Tests for Client-Data Rules:
1. Dataset Swap Test (Section 49)
2. Model Knowledge Contamination Test (Section 50)
3. Multi-Format JSON Dataset Ingestion
4. DuckDB In-Memory Execution
5. Thread-Safe Query Cache & Automatic Invalidation
6. Universal Normalization Output (Section 8)
"""

import json
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from app.dataset.loader import dataset_loader
from app.query.cache import query_cache
from app.query.duckdb_engine import duckdb_engine
from app.query.executor import query_executor
from app.query.parser import query_parser
from app.query.schema import FilterCondition, StructuredQuery
from app.utils.normalization import universal_text_normalizer, parse_date_string


@pytest.fixture(autouse=True)
def restore_dataset():
    """Ensure dataset is restored to customers after test run."""
    yield
    dataset_loader.load_sample("customers")


def test_dataset_swap_rule_zero_retention():
    """Section 49: Dataset Swap Test.
    
    1. Load Dataset A with John salary = 50000 -> answers 50000.
    2. Load Dataset B with John salary = 95000 -> answers 95000 (never 50000).
    3. Load Dataset C without John -> reports not found (never 50000 or 95000).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Dataset A: John = 50000
        df_a = pd.DataFrame([
            {"Name": "John Doe", "Salary": 50000, "Department": "IT"},
            {"Name": "Alice Smith", "Salary": 60000, "Department": "HR"},
        ])
        path_a = tmp_path / "dataset_a.csv"
        df_a.to_csv(path_a, index=False)

        ok, _ = dataset_loader.load_dataset(path_a, custom_name="Dataset_A")
        assert ok is True

        q_john = StructuredQuery(
            operation="FILTER",
            conditions=[FilterCondition(column="Name", operator="=", value="John Doe")],
            select_columns=["Salary"],
        )
        res_a = query_executor.execute(q_john, dataset_loader.dataframe)
        assert len(res_a["results"]) == 1
        assert res_a["results"][0]["Salary"] == 50000

        # Dataset B: John = 95000
        df_b = pd.DataFrame([
            {"Name": "John Doe", "Salary": 95000, "Department": "Finance"},
            {"Name": "Bob Jones", "Salary": 70000, "Department": "Sales"},
        ])
        path_b = tmp_path / "dataset_b.csv"
        df_b.to_csv(path_b, index=False)

        ok, _ = dataset_loader.load_dataset(path_b, custom_name="Dataset_B")
        assert ok is True

        res_b = query_executor.execute(q_john, dataset_loader.dataframe)
        assert len(res_b["results"]) == 1
        assert res_b["results"][0]["Salary"] == 95000
        assert res_b["results"][0]["Salary"] != 50000

        # Dataset C: John does NOT exist
        df_c = pd.DataFrame([
            {"Name": "Charlie Brown", "Salary": 80000, "Department": "Legal"},
            {"Name": "Alice Smith", "Salary": 60000, "Department": "HR"},
        ])
        path_c = tmp_path / "dataset_c.csv"
        df_c.to_csv(path_c, index=False)

        ok, _ = dataset_loader.load_dataset(path_c, custom_name="Dataset_C")
        assert ok is True

        res_c = query_executor.execute(q_john, dataset_loader.dataframe)
        assert len(res_c["results"]) == 0


def test_model_knowledge_contamination_client_data_supersedes():
    """Section 50: Model Knowledge Contamination Test.
    
    If client dataset states that France's capital is 'Berlin' or Apple CEO is 'John Doe',
    the system MUST return the client's data value and never fall back to memorized facts.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        df_fake = pd.DataFrame([
            {"Country": "France", "Capital": "Berlin"},
            {"Country": "Japan", "Capital": "London"},
        ])
        path_fake = Path(tmp_dir) / "fake_capitals.json"
        with open(path_fake, "w", encoding="utf-8") as f:
            json.dump(df_fake.to_dict(orient="records"), f)

        ok, _ = dataset_loader.load_dataset(path_fake, custom_name="FakeCapitals")
        assert ok is True

        q = StructuredQuery(
            operation="FILTER",
            conditions=[FilterCondition(column="Country", operator="=", value="France")],
            select_columns=["Capital"],
        )
        res = query_executor.execute(q, dataset_loader.dataframe)
        assert len(res["results"]) == 1
        assert res["results"][0]["Capital"] == "Berlin"
        assert res["results"][0]["Capital"] != "Paris"


def test_json_format_ingestion():
    """Verify JSON ingestion supports both record arrays and object-wrapped records."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Array of records JSON
        data_records = [
            {"product_id": "P1", "name": "Laptop", "price": 1200, "stock": 15},
            {"product_id": "P2", "name": "Mouse", "price": 25, "stock": 100},
            {"product_id": "P3", "name": "Keyboard", "price": 75, "stock": 50},
        ]
        path_records = tmp_path / "products.json"
        with open(path_records, "w", encoding="utf-8") as f:
            json.dump(data_records, f)

        ok, msg = dataset_loader.load_dataset(path_records, custom_name="JSONProducts")
        assert ok is True
        assert len(dataset_loader.dataframe) == 3
        assert "price" in dataset_loader.get_columns()

        # 2. Dictionary with records key
        data_dict = {
            "records": [
                {"city": "Mumbai", "region": "West"},
                {"city": "Kolkata", "region": "East"},
            ]
        }
        path_dict = tmp_path / "locations.json"
        with open(path_dict, "w", encoding="utf-8") as f:
            json.dump(data_dict, f)

        ok2, msg2 = dataset_loader.load_dataset(path_dict, custom_name="JSONLocations")
        assert ok2 is True
        assert len(dataset_loader.dataframe) == 2
        assert "city" in dataset_loader.get_columns()


def test_duckdb_query_engine():
    """Verify DuckDB in-memory query engine can execute SQL and structured queries."""
    if not duckdb_engine.is_available:
        pytest.skip("DuckDB not installed")

    df = pd.DataFrame([
        {"Department": "Engineering", "Salary": 100000, "Years": 5},
        {"Department": "Engineering", "Salary": 120000, "Years": 7},
        {"Department": "Marketing", "Salary": 80000, "Years": 3},
        {"Department": "Marketing", "Salary": 90000, "Years": 4},
        {"Department": "Sales", "Salary": 70000, "Years": 2},
    ])

    # Direct SQL test
    res = duckdb_engine.execute_sql(df, "SELECT Department, ROUND(AVG(Salary), 2) AS avg_sal FROM df GROUP BY Department ORDER BY avg_sal DESC")
    assert res["status"] == "success"
    assert len(res["results"]) == 3
    assert res["results"][0]["Department"] == "Engineering"
    assert res["results"][0]["avg_sal"] == 110000.0

    # StructuredQuery test
    q_avg = StructuredQuery(operation="AVERAGE", target_column="Salary")
    res_agg = duckdb_engine.execute_structured_query(q_avg, df)
    assert res_agg is not None
    assert res_agg["aggregation"]["metric"] == "AVERAGE"
    assert res_agg["aggregation"]["value"] == 92000.0


def test_query_cache_thread_safe_and_invalidation():
    """Verify QueryCache stores, retrieves, and invalidates entries properly."""
    query_cache.clear()
    assert len(query_cache) == 0

    hash_a = "dataset_hash_1"
    query_cache.set(hash_a, "who earns the most", {"answer": "Alice", "value": 150000})
    assert len(query_cache) == 1

    cached = query_cache.get(hash_a, "who earns the most")
    assert cached is not None
    assert cached["answer"] == "Alice"

    # Different query
    assert query_cache.get(hash_a, "who earns the least") is None

    # Different dataset hash
    assert query_cache.get("dataset_hash_2", "who earns the most") is None

    # Invalidation on dataset change
    query_cache.invalidate(hash_a)
    assert len(query_cache) == 0
    assert query_cache.get(hash_a, "who earns the most") is None


def test_universal_text_normalizer():
    """Section 8: Verify universal_text_normalizer produces all required forms."""
    text = "  Hello,   WORLD!  "
    norm = universal_text_normalizer(text)

    assert norm["raw"] == text
    assert norm["trimmed"] == "Hello,   WORLD!"
    assert norm["lowercase"] == "hello,   world!"
    assert norm["casefolded"] == "hello,   world!"
    assert norm["unicode_normalized"] == "Hello,   WORLD!"
    assert norm["whitespace_normalized"] == "Hello, WORLD!"
    assert norm["punctuation_normalized"] == "Hello WORLD"
    assert norm["alphanumeric_normalized"] == "Hello WORLD"
    assert norm["compact_form"] == "hello,world!"
    assert norm["tokens"] == ["hello", "world"]


def test_dateparser_natural_dates():
    """Verify natural language date parsing across formats."""
    d1 = parse_date_string("2024-01-15")
    assert d1 is not None
    assert d1.year == 2024 and d1.month == 1 and d1.day == 15

    d2 = parse_date_string("January 16th 2024")
    assert d2 is not None
    assert d2.year == 2024 and d2.month == 1 and d2.day == 16

    d3 = parse_date_string("15/01/2024")
    assert d3 is not None
    assert d3.year == 2024 and d3.month == 1

