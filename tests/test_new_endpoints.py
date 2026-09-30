"""Unit tests verifying all Deliverable 17 API endpoints."""

import io
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """GET /health returns 200 with status, rows, columns, and Ollama status."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "active_dataset" in data
    assert "rows" in data
    assert "columns" in data
    assert "ollama" in data


def test_dataset_profile_endpoint():
    """GET /dataset/profile returns 200 with dataset columns, rows, and types."""
    res = client.get("/dataset/profile")
    assert res.status_code == 200
    data = res.json()
    assert "dataset_name" in data
    assert "columns" in data
    assert "row_count" in data


def test_chat_endpoint():
    """POST /chat answers natural language query with zero hallucination and grounded response."""
    payload = {
        "question": "What is the capital of Telangana?",
        "session_id": "test_session_1"
    }
    res = client.post("/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "Hyderabad" in data["answer"]
    assert data["grounded"] is True
    assert data["result_count"] >= 1
    assert data["confidence"] > 0.8


def test_history_endpoint():
    """GET /history retrieves past dialogue turns for a session."""
    # First query
    client.post("/chat", json={"question": "Capital of Karnataka?", "session_id": "test_session_hist"})
    
    # Check history
    res = client.get("/history?session_id=test_session_hist")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["turns_count"] >= 1
    assert any("Karnataka" in t["question"] for t in data["history"])


def test_download_chat_endpoint():
    """GET /download-chat exports conversation history in JSON and TXT format."""
    # JSON download
    res_json = client.get("/download-chat?session_id=test_session_hist&format=json")
    assert res_json.status_code == 200
    assert "application/json" in res_json.headers.get("content-type", "")
    json_data = res_json.json()
    assert "turns" in json_data

    # TXT download
    res_txt = client.get("/download-chat?session_id=test_session_hist&format=txt")
    assert res_txt.status_code == 200
    assert "text/plain" in res_txt.headers.get("content-type", "")
    assert "Universal Dataset Chatbot Dialogue Log" in res_txt.text


def test_clear_chat_endpoint():
    """POST /clear-chat wipes session memory."""
    res = client.post("/clear-chat", json={"session_id": "test_session_hist"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"

    # History should now be empty for that session
    hist_res = client.get("/history?session_id=test_session_hist")
    assert hist_res.json()["turns_count"] == 0


def test_new_session_endpoint():
    """POST /new-session returns a unique UUID session identifier."""
    res = client.post("/new-session")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "session_id" in data
    assert len(data["session_id"]) >= 30


def test_upload_endpoint():
    """POST /upload ingests new CSV dataset and resets dialogue context."""
    csv_content = "Product,Category,Price\nLaptop,Electronics,1200\nMouse,Electronics,25\nDesk,Furniture,300\n"
    file = io.BytesIO(csv_content.encode("utf-8"))
    files = {"file": ("test_products.csv", file, "text/csv")}

    res = client.post("/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["rows"] == 3
    assert "Product" in data["columns"]
    assert "Price" in data["columns"]

    # Verify query on new dataset
    chat_res = client.post("/chat", json={"question": "What is the price of Laptop?"})
    assert chat_res.status_code == 200
    assert "1200" in chat_res.json()["answer"]

