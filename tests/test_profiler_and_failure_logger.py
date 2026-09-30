"""Tests for Dynamic Dataset Profiler, Version Fingerprinting, and Failure Analysis Logger."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.dataset.loader import dataset_loader
from app.verification.failure_logger import failure_logger
from pathlib import Path


@pytest.fixture(scope="module")
def client():
    dataset_loader.load_dataset(Path("data/sample_employees.xlsx"))
    return TestClient(app)


def test_dataset_fingerprint_endpoint(client):
    response = client.get("/api/dataset/fingerprint")
    assert response.status_code == 200
    data = response.json()
    assert "dataset_id" in data
    assert "schema_hash" in data
    assert "dataset_version" in data


def test_failure_analysis_endpoint(client):
    failure_logger.clear()
    failure_logger.log_failure(
        category="UNMAPPED_COLUMN",
        question="What is the stock price?",
        dataset_name="sample_employees.xlsx",
        operation="UNKNOWN",
        details={"missing_column": "stock price"}
    )
    response = client.get("/api/failure-analysis")
    assert response.status_code == 200
    data = response.json()
    assert data["total_failures"] >= 1
    assert data["category_breakdown"]["UNMAPPED_COLUMN"] >= 1
    assert len(data["recent_records"]) >= 1


def test_multi_question_api_execution(client):
    req = {
        "question": "How many employees are in IT and what is their average salary?",
        "session_id": "test_multi"
    }
    response = client.post("/api/query", json=req)
    assert response.status_code == 200
    data = response.json()
    assert data["operation"] == "MULTI_QUESTION"
    assert "1." in data["answer"]
    assert "2." in data["answer"]
    assert data["source_type"] == "DATASET"

