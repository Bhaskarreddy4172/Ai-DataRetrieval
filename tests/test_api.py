"""Integration tests for all REST API endpoints."""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.dataset.loader import dataset_loader

client = TestClient(app)


@pytest.fixture(autouse=True)
def ensure_states_dataset():
    data_path = Path(__file__).resolve().parent.parent / "data" / "indian_states_capitals.csv"
    if data_path.exists():
        dataset_loader.load_dataset(data_path)


def test_get_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "active_dataset" in data
    assert "rows" in data
    assert data["rows"] > 0


def test_get_dataset_profile():
    res = client.get("/dataset/profile")
    assert res.status_code == 200
    data = res.json()
    assert "basic_info" in data
    assert "columns" in data
    assert len(data["columns"]) > 0


def test_get_dataset_schema():
    res = client.get("/dataset/schema")
    assert res.status_code == 200
    data = res.json()
    assert "columns" in data
    assert "row_count" in data


def test_post_query():
    res = client.post("/query", json={"question": "What is the capital of Telangana?", "session_id": "test_api"})
    assert res.status_code == 200
    data = res.json()
    assert data["operation"] in {"LOOKUP", "FILTER"}
    assert data["result_count"] > 0
    assert len(data["results"]) > 0
    assert "Hyderabad" in data["answer"]
    assert data["grounded"] is True


def test_post_query_out_of_scope():
    res = client.post("/query", json={"question": "What is the GDP of Mars?", "session_id": "test_api"})
    assert res.status_code == 200
    data = res.json()
    assert data["operation"] == "UNKNOWN"
    assert "does not contain" in data["answer"]


def test_get_query_history():
    # Make a query first
    client.post("/query", json={"question": "How many customers are there?"})
    res = client.get("/query/history?limit=5")
    assert res.status_code == 200
    history = res.json()
    assert isinstance(history, list)
    assert len(history) > 0


def test_upload_csv_dataset():
    csv_content = b"Product,Category,Price\nLaptop,Electronics,55000\nPhone,Electronics,25000\nChair,Furniture,4500\n"
    file = {"file": ("test_products_api.csv", io.BytesIO(csv_content), "text/csv")}
    res = client.post("/dataset/upload", files=file)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["profile"]["basic_info"]["row_count"] == 3

    # Query the newly uploaded dataset
    q_res = client.post("/query", json={"question": "Show all items in test_products_api"})
    assert q_res.status_code == 200
    assert q_res.json()["result_count"] == 3
