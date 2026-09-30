"""Automated test suite for unprofessional language, slang, and colloquial queries."""

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


def test_who_gettin_paid_most():
    resp = client.post("/api/query", json={"question": "who gettin paid most?"})
    assert resp.status_code == 200
    data = resp.json()
    assert any(name in data["answer"] or name in str(data["results"]) for name in ["Ranbir Singh", "Jitesh Sharma", "Arjun Patel", "Tanvi Iyer"])


def test_earning_highest_in_hyd_bro():
    resp = client.post("/api/query", json={"question": "who is earning highest in hyd bro"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "DATASET"
    assert len(data["results"]) >= 1
    assert data["results"][0]["City"] == "Hyderabad"


def test_banglore_which_state_bro():
    resp = client.post("/api/query", json={"question": "banglore which state bro?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "GENERAL_KNOWLEDGE"
    assert "Karnataka" in data["answer"]


def test_ppl_from_hyd():
    resp = client.post("/api/query", json={"question": "show me ppl from hyd"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "DATASET"
    assert len(data["results"]) > 0
    for r in data["results"]:
        assert r["City"] == "Hyderabad"


def test_who_joined_recently():
    resp = client.post("/api/query", json={"question": "who joined recently?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "DATASET"
    assert len(data["results"]) >= 1


def test_who_joined_first():
    resp = client.post("/api/query", json={"question": "who joined first?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "DATASET"
    assert len(data["results"]) >= 1
