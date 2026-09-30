"""Automated test suite for conversational follow-up questions and pronoun resolution."""

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


def test_conversational_followup_sequence():
    session_id = "test_followup_seq_1"

    # Turn 1: Highest paid employee
    r1 = client.post("/api/query", json={"question": "Who is the highest paid employee?", "session_id": session_id})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["source_type"] == "DATASET"
    assert len(d1["results"]) == 1
    top_person = d1["results"][0]["Employee Name"]

    # Turn 2: Where does he work?
    r2 = client.post("/api/query", json={"question": "where does he work?", "session_id": session_id})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["source_type"] == "DATASET"
    # Must retrieve City of that exact same person
    assert len(d2["results"]) >= 1
    assert d2["results"][0].get("City") == d1["results"][0].get("City")

    # Turn 3: What is his department?
    r3 = client.post("/api/query", json={"question": "what is his department?", "session_id": session_id})
    assert r3.status_code == 200
    d3 = r3.json()
    assert d3["source_type"] == "DATASET"
    assert len(d3["results"]) >= 1
    assert d3["results"][0].get("Department") == d1["results"][0].get("Department")


def test_ordinal_followup():
    session_id = "test_ordinal_seq_2"

    # Turn 1: Highest salary
    r1 = client.post("/api/query", json={"question": "who has the highest salary?", "session_id": session_id})
    assert r1.status_code == 200

    # Turn 2: Next one / Second guy
    r2 = client.post("/api/query", json={"question": "what about the second guy?", "session_id": session_id})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["source_type"] == "DATASET"
    assert len(d2["results"]) >= 1

