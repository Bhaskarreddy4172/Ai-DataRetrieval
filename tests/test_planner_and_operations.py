"""Unit tests for query planner and discrete operations across all datasets."""

import pytest
import pandas as pd
from app.dataset.indexer import dataset_indexer
from app.query.planner import query_planner
from app.query.engine import query_engine


@pytest.fixture(scope="module")
def employee_df():
    df = pd.DataFrame({
        "Employee ID": [f"EMP-{i}" for i in range(1, 6)],
        "Employee Name": ["Aarav Sharma", "Neha Patel", "Rohan Verma", "Pooja Reddy", "Vikram Iyer"],
        "Department": ["Engineering", "Engineering", "Sales", "Sales", "Human Resources"],
        "Salary": [800000, 1200000, 600000, 750000, 500000],
        "City": ["Bengaluru", "Hyderabad", "Mumbai", "Hyderabad", "Bengaluru"]
    })
    from app.dataset.loader import dataset_loader
    dataset_loader.dataframe = df
    dataset_indexer.build_index(df, "employees")
    return df


@pytest.fixture(scope="module")
def product_df():
    df = pd.DataFrame({
        "Product ID": ["P-1", "P-2", "P-3", "P-4"],
        "Product Name": ["Sony Headphones", "Apple MacBook Air", "Samsung Galaxy S24", "Boat Rockerz"],
        "Category": ["Audio", "Laptops", "Smartphones", "Audio"],
        "Price": [19999, 89999, 74999, 1499],
        "Stock Quantity": [45, 12, 28, 150]
    })
    from app.dataset.loader import dataset_loader
    dataset_loader.dataframe = df
    dataset_indexer.build_index(df, "products")
    return df


def test_planner_anti_hallucination(employee_df):
    cols = list(employee_df.columns)
    q = query_planner.plan_query("what is the weather in Delhi?", cols, {})
    assert q.operation == "UNKNOWN"

    q2 = query_planner.plan_query("who is the prime minister?", cols, {})
    assert q2.operation == "UNKNOWN"


def test_planner_median(employee_df):
    cols = list(employee_df.columns)
    q = query_planner.plan_query("what is the median salary?", cols, {})
    assert q.operation == "MEDIAN"
    assert q.target_column == "Salary"

    res = query_engine.execute(q, employee_df)
    assert res["aggregation"]["metric"] == "MEDIAN"
    assert res["aggregation"]["value"] == 750000.0


def test_planner_abbreviation_and_slang(employee_df):
    cols = list(employee_df.columns)
    # "how many in hyd" -> Hyderabad count
    q = query_planner.plan_query("how many in hyd?", cols, {})
    assert q.operation == "COUNT"
    assert any(c.column == "City" and c.value == "Hyderabad" for c in q.conditions)

    res = query_engine.execute(q, employee_df)
    assert res["aggregation"]["value"] == 2


def test_planner_distinct(employee_df):
    cols = list(employee_df.columns)
    q = query_planner.plan_query("distinct departments", cols, {})
    assert q.operation == "DISTINCT"
    assert q.target_column == "Department"

    res = query_engine.execute(q, employee_df)
    assert res["aggregation"]["value"] == 3


def test_planner_duplicates(employee_df):
    cols = list(employee_df.columns)
    q = query_planner.plan_query("any duplicate records?", cols, {})
    assert q.operation == "DUPLICATES"

    res = query_engine.execute(q, employee_df)
    assert res["status"] == "success"
    assert res["aggregation"]["metric"] == "DUPLICATES"


def test_planner_top_n_products(product_df):
    cols = list(product_df.columns)
    q = query_planner.plan_query("top 2 most expensive products", cols, {})
    assert q.sort_order == "DESC"
    assert q.limit == 2

    res = query_engine.execute(q, product_df)
    assert len(res["results"]) == 2
    assert res["results"][0]["Product Name"] == "Apple MacBook Air"

