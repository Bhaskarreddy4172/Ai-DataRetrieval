"""Automated test suite for ambiguous queries and clarification prompts."""

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
        data_path = Path(__file__).resolve().parent.parent / "data" / "indian_states_capitals.csv"
    dataset_loader.load_dataset(data_path)
    yield
    default_path = Path(__file__).resolve().parent.parent / "data" / "indian_states_capitals.csv"
    if default_path.exists():
        dataset_loader.load_dataset(default_path)


def test_single_word_ambiguous_query():
    resp = client.post("/api/query", json={"question": "xyz"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"]
    assert len(data["answer"].strip()) > 0
    # Must not crash or return null/empty
    assert data["source_type"] in {"AMBIGUOUS", "DATASET", "UNSUPPORTED"}


def test_empty_query_rejected():
    resp = client.post("/api/query", json={"question": "   "})
    # Sanitizer rejects blank inputs with 400 Bad Request
    assert resp.status_code == 400


def test_never_empty_response():
    queries = [
        "tell me something",
        "hello",
        "find",
        "show me",
        "which one",
    ]
    for q in queries:
        resp = client.post("/api/query", json={"question": q})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] is not None
        assert len(data["answer"].strip()) > 0

