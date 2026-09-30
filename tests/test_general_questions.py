"""Automated test suite for General Knowledge questions."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_bangalore_variations():
    queries = [
        "Which state does banglore belongs to?",
        "which state banglore?",
        "banglore which state bro?",
        "where is banglore?",
        "what state is banglore in?",
        "banglore is under which state?",
        "which state does Bengaluru belong to?",
        "which state does blr belong to?",
    ]
    for q in queries:
        resp = client.post("/api/query", json={"question": q})
        assert resp.status_code == 200, f"Failed on {q}"
        data = resp.json()
        assert data["source_type"] in {"GENERAL_KNOWLEDGE", "DATASET"}, f"Wrong source_type for {q}: {data['source_type']}"
        assert "Karnataka" in data["answer"], f"Karnataka not in answer for {q}: {data['answer']}"


def test_other_indian_cities():
    city_expectations = [
        ("which state is Hyderabad in?", "Telangana"),
        ("where is Mumbai?", "Maharashtra"),
        ("where does Chennai belong to?", "Tamil Nadu"),
        ("which state does Kolkata belong to?", "West Bengal"),
        ("which state does Pune belong to?", "Maharashtra"),
        ("which state is Jaipur in?", "Rajasthan"),
    ]
    for q, expected_state in city_expectations:
        resp = client.post("/api/query", json={"question": q})
        assert resp.status_code == 200
        data = resp.json()
        assert data["source_type"] in {"GENERAL_KNOWLEDGE", "DATASET"}
        assert expected_state in data["answer"], f"Expected {expected_state} in answer for '{q}': {data['answer']}"


def test_national_facts():
    facts = [
        ("what is the capital of India?", "New Delhi"),
        ("who is the president of India?", "Droupadi Murmu"),
        ("how many states are there in India?", "28 states"),
    ]
    for q, expected in facts:
        resp = client.post("/api/query", json={"question": q})
        assert resp.status_code == 200
        data = resp.json()
        assert data["source_type"] == "GENERAL_KNOWLEDGE"
        assert expected.lower() in data["answer"].lower()


def test_technical_definitions():
    tech_queries = [
        ("what is Python?", "programming language"),
        ("what does API mean?", "Application Programming Interface"),
        ("what is machine learning?", "Artificial Intelligence"),
        ("what is the difference between AI and ML?", "Machine Learning"),
    ]
    for q, expected in tech_queries:
        resp = client.post("/api/query", json={"question": q})
        assert resp.status_code == 200
        data = resp.json()
        assert data["source_type"] == "GENERAL_KNOWLEDGE"
        assert expected.lower() in data["answer"].lower(), f"Failed on {q}: {data['answer']}"

