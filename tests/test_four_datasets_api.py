"""End-to-end API integration tests for all 4 predefined sample datasets."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_samples_listing():
    res = client.get("/dataset/samples")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 4
    sample_ids = [s["id"] for s in data]
    assert "customers" in sample_ids
    assert "states" in sample_ids
    assert "products" in sample_ids
    assert "employees" in sample_ids


def test_switch_and_query_states():
    # 1. Switch to states
    res = client.post("/dataset/sample/states")
    assert res.status_code == 200

    # 2. Query states
    res = client.post("/query", json={"question": "What is the capital of Karnataka?"})
    assert res.status_code == 200
    data = res.json()
    assert data["grounded"] is True
    assert "Bengaluru" in data["answer"] or "Bangalore" in data["answer"]


def test_switch_and_query_customers():
    from pathlib import Path
    if not (Path("data/sample_customers.csv").exists()):
        pytest.skip("sample_customers.csv was removed per user dataset cleanup")
    # 1. Switch to customers
    res = client.post("/dataset/sample/customers")
    assert res.status_code == 200

    # 2. Query customers with slang / abbreviation
    res = client.post("/query", json={"question": "Who spent more than 10k?"})
    assert res.status_code == 200
    data = res.json()
    assert data["grounded"] is True
    assert data["result_count"] > 0


def test_switch_and_query_products():
    from pathlib import Path
    if not (Path("data/sample_products.csv").exists()):
        pytest.skip("sample_products.csv was removed per user dataset cleanup")
    # 1. Switch to products
    res = client.post("/dataset/sample/products")
    assert res.status_code == 200

    # 2. Query products
    res = client.post("/query", json={"question": "What is the costliest product?"})
    assert res.status_code == 200
    data = res.json()
    assert data["grounded"] is True
    assert data["result_count"] > 0


def test_switch_and_query_employees():
    from pathlib import Path
    if not (Path("data/sample_employees.xlsx").exists()):
        pytest.skip("sample_employees.xlsx was removed per user dataset cleanup")
    # 1. Switch to employees
    res = client.post("/dataset/sample/employees")
    assert res.status_code == 200

    # 2. Query employees with abbreviation
    res = client.post("/query", json={"question": "Who makes the most in Engineering?"})
    assert res.status_code == 200
    data = res.json()
    assert data["grounded"] is True
    assert data["result_count"] > 0

    # 3. Test relationships
    res_rel = client.get("/dataset/relationships")
    assert res_rel.status_code == 200
    rel_data = res_rel.json()
    assert "dimensions" in rel_data
    assert "metrics" in rel_data

