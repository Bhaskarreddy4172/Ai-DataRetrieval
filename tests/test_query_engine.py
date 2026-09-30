"""Unit tests for the deterministic Python QueryEngine across diverse operations."""

import pytest
from app.dataset.loader import DatasetLoader
from app.query.engine import QueryEngine
from app.query.schema import FilterCondition, StructuredQuery


@pytest.fixture
def loader_and_engine():
    from pathlib import Path
    data_path = Path("data/sample_customer_data.xlsx")
    if not data_path.exists():
        pytest.skip("sample_customer_data.xlsx was removed per user dataset cleanup")
    loader = DatasetLoader(initial_path=str(data_path))
    engine = QueryEngine()
    return loader, engine


def test_filter_numeric_greater_than(loader_and_engine):
    loader, engine = loader_and_engine
    query = StructuredQuery(
        operation="FILTER",
        conditions=[FilterCondition(column="Service Amount", operator=">", value=10000)],
        limit=50
    )
    result = engine.execute(query, loader.dataframe)
    assert result["status"] == "success"
    assert result["result_count"] > 0
    for r in result["results"]:
        assert r["Service Amount"] > 10000


def test_filter_string_equals(loader_and_engine):
    loader, engine = loader_and_engine
    query = StructuredQuery(
        operation="FILTER",
        conditions=[FilterCondition(column="City", operator="=", value="Hyderabad")],
        limit=50
    )
    result = engine.execute(query, loader.dataframe)
    assert result["result_count"] > 0
    for r in result["results"]:
        assert r["City"].lower() == "hyderabad"


def test_multi_condition_and(loader_and_engine):
    loader, engine = loader_and_engine
    query = StructuredQuery(
        operation="FILTER",
        conditions=[
            FilterCondition(column="City", operator="=", value="Hyderabad"),
            FilterCondition(column="Service Amount", operator=">", value=5000),
        ],
        logical_operator="AND",
        limit=50
    )
    result = engine.execute(query, loader.dataframe)
    for r in result["results"]:
        assert r["City"].lower() == "hyderabad"
        assert r["Service Amount"] > 5000


def test_count_operation(loader_and_engine):
    loader, engine = loader_and_engine
    query = StructuredQuery(operation="COUNT")
    result = engine.execute(query, loader.dataframe)
    assert result["aggregation"]["metric"] == "COUNT"
    assert result["aggregation"]["value"] == 100


def test_sum_operation(loader_and_engine):
    loader, engine = loader_and_engine
    query = StructuredQuery(operation="SUM", target_column="Service Amount")
    result = engine.execute(query, loader.dataframe)
    assert result["aggregation"]["metric"] == "SUM"
    assert result["aggregation"]["value"] > 100000


def test_average_operation(loader_and_engine):
    loader, engine = loader_and_engine
    query = StructuredQuery(operation="AVERAGE", target_column="Service Amount")
    result = engine.execute(query, loader.dataframe)
    assert result["aggregation"]["metric"] == "AVERAGE"
    assert 3000 <= result["aggregation"]["value"] <= 30000


def test_min_and_max(loader_and_engine):
    loader, engine = loader_and_engine
    q_max = StructuredQuery(operation="MAX", target_column="Service Amount")
    res_max = engine.execute(q_max, loader.dataframe)
    q_min = StructuredQuery(operation="MIN", target_column="Service Amount")
    res_min = engine.execute(q_min, loader.dataframe)

    assert res_max["aggregation"]["value"] >= res_min["aggregation"]["value"]


def test_sorting(loader_and_engine):
    loader, engine = loader_and_engine
    query = StructuredQuery(
        operation="SORT",
        sort_column="Service Amount",
        sort_order="DESC",
        limit=10
    )
    result = engine.execute(query, loader.dataframe)
    amounts = [r["Service Amount"] for r in result["results"]]
    assert amounts == sorted(amounts, reverse=True)
