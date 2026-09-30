"""Automated test suite for Hybrid queries combining dataset and general knowledge."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.dataset.loader import dataset_loader

client = TestClient(app)


@pytest.fixture(autouse=True)
def ensure_employees_dataset():
    data_path = Path(__file__).resolve().parent.parent / "data" / "sample_employees.xlsx"
    dataset_loader.load_dataset(data_path)


def test_hybrid_bangalore():
    resp = client.post("/api/query", json={"question": "Who works in Bangalore and which state is that city in?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "HYBRID"
    assert data["dataset_used"] is True
    assert data["general_knowledge_used"] is True
    assert "Karnataka" in data["answer"]
    assert len(data["results"]) > 0


def test_hybrid_hyderabad():
    resp = client.post("/api/query", json={"question": "Which employees are from Hyderabad and which state is Hyderabad in?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "HYBRID"
    assert data["dataset_used"] is True
    assert data["general_knowledge_used"] is True
    assert "Telangana" in data["answer"]
    assert len(data["results"]) > 0


def test_hybrid_pune():
    resp = client.post("/api/query", json={"question": "Who is in Pune and what state is Pune in?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "HYBRID"
    assert data["dataset_used"] is True
    assert data["general_knowledge_used"] is True
    assert "Maharashtra" in data["answer"]

