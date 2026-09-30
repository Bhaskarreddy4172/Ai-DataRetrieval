"""Automated test suite for pure Dataset questions."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.dataset.loader import dataset_loader

client = TestClient(app)


@pytest.fixture(autouse=True)
def ensure_employees_dataset():
    data_path = Path(__file__).resolve().parent.parent / "data" / "sample_employees.xlsx"
    if not data_path.exists():
        pytest.skip("sample_employees.xlsx was removed per user dataset cleanup")
    dataset_loader.load_dataset(data_path)


def test_highest_paid_employee():
    resp = client.post("/api/query", json={"question": "Who is the highest paid employee?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "DATASET"
    assert data["dataset_used"] is True
    assert data["general_knowledge_used"] is False
    assert any(name in data["answer"] or name in str(data["results"]) for name in ["Ranbir Singh", "Jitesh Sharma", "Arjun Patel", "Tanvi Iyer"])


def test_city_filter():
    resp = client.post("/api/query", json={"question": "Who works in Hyderabad?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "DATASET"
    assert data["dataset_used"] is True
    assert len(data["results"]) > 0
    for r in data["results"]:
        assert r["City"].lower() == "hyderabad"


def test_salary_threshold():
    resp = client.post("/api/query", json={"question": "Who gets more than 10L?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "DATASET"
    assert len(data["results"]) > 0
    for r in data["results"]:
        assert r["Salary"] > 1_000_000


def test_highest_average_salary_department():
    resp = client.post("/api/query", json={"question": "Which department has the highest average salary?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "DATASET"
    assert "Engineering" in data["answer"] or (data["results"] and data["results"][0].get("Department") == "Engineering")


def test_department_count():
    resp = client.post("/api/query", json={"question": "How many employees are in Sales?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "DATASET"
    assert data["result_count"] > 0 or (data.get("aggregation") and data["aggregation"].get("value") > 0)
