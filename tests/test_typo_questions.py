"""Automated test suite for queries with typos and spelling variations."""

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


def test_city_typos_general():
    typos = [
        ("which state does banglroe belong?", "Karnataka"),
        ("which state does bangluru belong to?", "Karnataka"),
        ("hydrabad which state?", "Telangana"),
        ("mumbaii which state?", "Maharashtra"),
        ("delh which state?", "Delhi"),
    ]
    for q, expected_state in typos:
        resp = client.post("/api/query", json={"question": q})
        assert resp.status_code == 200
        data = resp.json()
        assert data["source_type"] == "GENERAL_KNOWLEDGE"
        assert expected_state in data["answer"]


def test_dataset_typos():
    resp = client.post("/api/query", json={"question": "who is eraning highest in hydrabad?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "DATASET"
    assert len(data["results"]) >= 1
    assert data["results"][0]["City"] == "Hyderabad"


def test_dataset_slang_and_typo():
    resp = client.post("/api/query", json={"question": "who all are above 10L in hydrabad?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "DATASET"
    for r in data["results"]:
        assert r["City"] == "Hyderabad"
        assert r["Salary"] > 1_000_000

