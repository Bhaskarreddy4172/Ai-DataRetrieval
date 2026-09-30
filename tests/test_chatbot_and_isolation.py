"""Tests for Universal Dataset Chatbot, multi-dataset isolation, dynamic suggestions, and session clearing."""

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.conversation.context import conversation_manager
from app.dataset.loader import dataset_loader
from app.dataset.suggestions import dataset_suggestion_engine
from app.query.schema import StructuredQuery, FilterCondition


@pytest.fixture
def client():
    return TestClient(app)


def test_dataset_suggestion_engine_states():
    """Test dynamic suggestions on states dataset."""
    df = pd.DataFrame({
        "State": ["Karnataka", "Maharashtra", "Tamil Nadu"],
        "Capital": ["Bengaluru", "Mumbai", "Chennai"],
        "Type": ["State", "State", "State"],
    })
    suggestions = dataset_suggestion_engine.generate_suggestions(df, "indian_states.csv")
    assert len(suggestions) >= 3
    # Check that suggestions refer to actual columns in df
    for s in suggestions:
        assert any(col in s for col in ["State", "Capital", "Type", "record", "dataset"])

    summary = dataset_suggestion_engine.generate_summary(df, "indian_states.csv")
    assert "3 records" in summary
    assert "State" in summary
    assert "Capital" in summary


def test_dataset_suggestion_engine_hr():
    """Test dynamic suggestions on HR dataset without hardcoding."""
    df = pd.DataFrame({
        "Employee": ["Alice", "Bob", "Charlie"],
        "Department": ["Engineering", "Sales", "Engineering"],
        "Salary": [90000, 75000, 110000],
    })
    suggestions = dataset_suggestion_engine.generate_suggestions(df, "employees.xlsx")
    assert len(suggestions) >= 3
    # Ensure HR questions do NOT mention 'State' or 'Capital'
    for s in suggestions:
        assert "State" not in s
        assert "Capital" not in s
        assert any(col in s for col in ["Employee", "Department", "Salary", "record"])

    summary = dataset_suggestion_engine.generate_summary(df, "employees.xlsx")
    assert "3 records" in summary
    assert "Salary" in summary
    assert "Department" in summary


def test_session_isolation_and_clear():
    """Test that session context does not bleed across datasets."""
    session_id = "test_isolation_sess_1"
    
    # 1. Add turn to session
    dummy_query = StructuredQuery(
        operation="FILTER",
        conditions=[FilterCondition(column="State", operator="EQUALS", value="Karnataka")]
    )
    conversation_manager.add_turn(
        session_id=session_id,
        question="What is the capital of Karnataka?",
        query=dummy_query,
        result_count=1,
        results=[{"State": "Karnataka", "Capital": "Bengaluru"}]
    )

    history = conversation_manager.get_history(session_id)
    assert len(history) == 1
    assert history[0]["question"] == "What is the capital of Karnataka?"

    # 2. Clear session
    conversation_manager.clear_session(session_id)
    history_after = conversation_manager.get_history(session_id)
    assert len(history_after) == 0

    # 3. Test clear_all
    conversation_manager.add_turn(
        session_id="sess_a",
        question="Q1",
        query=dummy_query,
        result_count=1
    )
    conversation_manager.add_turn(
        session_id="sess_b",
        question="Q2",
        query=dummy_query,
        result_count=1
    )
    assert len(conversation_manager.get_history("sess_a")) == 1
    assert len(conversation_manager.get_history("sess_b")) == 1

    conversation_manager.clear_all()
    assert len(conversation_manager.get_history("sess_a")) == 0
    assert len(conversation_manager.get_history("sess_b")) == 0


def test_api_suggestions_and_summary_endpoints(client):
    """Verify GET /dataset/suggestions and GET /dataset/summary return valid 200 responses."""
    res_sugg = client.get("/dataset/suggestions")
    assert res_sugg.status_code == 200
    data_sugg = res_sugg.json()
    assert "suggestions" in data_sugg
    assert isinstance(data_sugg["suggestions"], list)
    assert len(data_sugg["suggestions"]) > 0

    res_sum = client.get("/dataset/summary")
    assert res_sum.status_code == 200
    data_sum = res_sum.json()
    assert "summary" in data_sum
    assert len(data_sum["summary"]) > 0
    assert "rows" in data_sum
    assert "columns" in data_sum


def test_api_clear_conversation_endpoint(client):
    """Verify POST /conversation/clear endpoint."""
    res = client.post("/conversation/clear", json={"session_id": "test_clear_session"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "test_clear_session" in data["message"]


def test_conversational_query_flow(client):
    """Verify full end-to-end question answering through conversational endpoint."""
    dataset_loader.load_sample("states")
    sess_id = "test_flow_sess"
    # Ask capital of Karnataka
    res = client.post("/query", json={
        "question": "What is the capital of Karnataka?",
        "session_id": sess_id
    })
    assert res.status_code == 200
    data = res.json()
    assert "Bengaluru" in data["answer"] or "Bangalore" in data["answer"]
    assert data["result_count"] >= 1
    assert data["grounded"] is True

    # Check conversation history recorded
    hist = conversation_manager.get_history(sess_id)
    assert len(hist) >= 1
