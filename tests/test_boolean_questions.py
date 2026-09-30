"""Comprehensive automated test suite for Boolean & Fact-Check Questions across General Knowledge, Dataset, and Hybrid queries."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dataset.loader import dataset_loader

client = TestClient(app)


@pytest.fixture(autouse=True)
def ensure_employees_dataset(request):
    data_path = Path(__file__).resolve().parent.parent / "data" / "sample_employees.xlsx"
    if not data_path.exists():
        if not request.node.name.startswith("test_gk_boolean"):
            pytest.skip("sample_employees.xlsx was removed per user dataset cleanup")
        else:
            dataset_loader.load_dataset(Path("data/indian_states_capitals.csv"))
    else:
        dataset_loader.load_dataset(data_path)
    yield
    states_path = Path(__file__).resolve().parent.parent / "data" / "indian_states_capitals.csv"
    if states_path.exists():
        dataset_loader.load_dataset(states_path)


# -----------------------------------------------------------------------------
# 1. GENERAL KNOWLEDGE BOOLEAN QUESTIONS
# -----------------------------------------------------------------------------
def test_gk_boolean_hyderabad_false():
    """User problem question: 'Is hyd belongs to Andra Pradesh?' -> False."""
    resp = client.post("/api/query", json={"question": "Is hyd belongs to Andra Pradesh?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["operation"] == "BOOLEAN_CHECK"
    assert data["source_type"] in {"GENERAL_KNOWLEDGE", "DATASET"}
    assert "False" in data["answer"]
    assert "Telangana" in data["answer"]
    assert "Andhra Pradesh" in data["answer"]


def test_gk_boolean_hyderabad_true():
    """'Is hyd in Telangana?' -> True."""
    resp = client.post("/api/query", json={"question": "Is hyd in Telangana?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["operation"] == "BOOLEAN_CHECK"
    assert data["source_type"] in {"GENERAL_KNOWLEDGE", "DATASET"}
    assert "True" in data["answer"]
    assert "Telangana" in data["answer"]


def test_gk_boolean_informal_variations():
    """Test slang, contractions, typos, and punctuation variations."""
    variations = [
        ("hyd under telangana?", True, "Telangana"),
        ("hyd is telangana right?", True, "Telangana"),
        ("hyd is ap right?", False, "Telangana"),
        ("is hyd in ap?", False, "Telangana"),
        ("hyd telangana?", True, "Telangana"),
        ("hyd belongs ap?", False, "Telangana"),
    ]
    for q, expected_bool, expected_state in variations:
        resp = client.post("/api/query", json={"question": q})
        assert resp.status_code == 200
        data = resp.json()
        assert data["operation"] == "BOOLEAN_CHECK"
        assert ("True" if expected_bool else "False") in data["answer"]
        assert expected_state in data["answer"]


def test_gk_boolean_bangalore():
    """Bangalore variations and capitals."""
    resp1 = client.post("/api/query", json={"question": "Does Bangalore belong to Karnataka?"})
    assert resp1.status_code == 200
    assert "True" in resp1.json()["answer"]
    assert "Karnataka" in resp1.json()["answer"]

    resp2 = client.post("/api/query", json={"question": "Is bangalore part of tamil nadu?"})
    assert resp2.status_code == 200
    assert "False" in resp2.json()["answer"]
    assert "Karnataka" in resp2.json()["answer"]

    resp3 = client.post("/api/query", json={"question": "Is Bangalore the capital of Karnataka?"})
    assert resp3.status_code == 200
    assert "True" in resp3.json()["answer"]


# -----------------------------------------------------------------------------
# 2. DATASET BOOLEAN QUESTIONS
# -----------------------------------------------------------------------------
def test_dataset_boolean_equality_true():
    """'Does Ananya work in Finance?' -> True (Ananya Reddy is in Finance)."""
    resp = client.post("/api/query", json={"question": "Does Ananya work in Finance?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["operation"] == "BOOLEAN_CHECK"
    assert data["source_type"] == "DATASET"
    assert data["dataset_used"] is True
    assert "True" in data["answer"]
    assert "Finance" in data["answer"]


def test_dataset_boolean_equality_false():
    """'Is Rahul working in Hyderabad?' -> False (Rahul is in Delhi)."""
    resp = client.post("/api/query", json={"question": "Is Rahul working in Hyderabad?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["operation"] == "BOOLEAN_CHECK"
    assert data["source_type"] == "DATASET"
    assert "False" in data["answer"]
    assert "Delhi" in data["answer"]
    assert "Hyderabad" in data["answer"]


def test_dataset_boolean_missing_column():
    """'Is Rahul 25 years old?' -> CANNOT_VERIFY / UNKNOWN, NEVER False."""
    resp = client.post("/api/query", json={"question": "Is Rahul 25 years old?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["operation"] == "BOOLEAN_CHECK"
    assert data["source_type"] == "DATASET"
    # Must NOT state False
    assert not data["answer"].startswith("False")
    assert "can't verify" in data["answer"].lower() or "cannot verify" in data["answer"].lower()
    assert "age" in data["answer"].lower()


def test_dataset_boolean_numeric_comparison():
    """Numerical salary threshold comparisons for Ananya (Salary 2,100,000)."""
    # Ananya's salary is 2,100,000 (> 10L is True, > 25L is False)
    resp1 = client.post("/api/query", json={"question": "Is salary of Ananya greater than 10L?"})
    assert resp1.status_code == 200
    assert "True" in resp1.json()["answer"]

    resp2 = client.post("/api/query", json={"question": "Is salary of Ananya greater than 25L?"})
    assert resp2.status_code == 200
    assert "False" in resp2.json()["answer"]


def test_dataset_boolean_date_comparison():
    """Date comparison: Rahul joined on 2023-08-20 (> 2022 is True)."""
    resp = client.post("/api/query", json={"question": "Did Rahul join after 2022?"})
    assert resp.status_code == 200
    assert "True" in resp.json()["answer"]


def test_dataset_boolean_entity_comparison():
    """Ananya earns 2.10M, Rahul earns 0.60M."""
    resp = client.post("/api/query", json={"question": "Does Ananya earn more than Rahul?"})
    assert resp.status_code == 200
    assert "True" in resp.json()["answer"]


def test_dataset_boolean_cross_row():
    """Both Rahul Dravid and Ananya Reddy are in Finance."""
    resp = client.post("/api/query", json={"question": "Do Rahul and Ananya work in the same department?"})
    assert resp.status_code == 200
    assert "True" in resp.json()["answer"]
    assert "Finance" in resp.json()["answer"]


def test_dataset_boolean_existence():
    """Existence of employees in Hyderabad with/without salary threshold."""
    resp1 = client.post("/api/query", json={"question": "Are there employees from Hyderabad?"})
    assert resp1.status_code == 200
    assert "True" in resp1.json()["answer"]

    resp2 = client.post("/api/query", json={"question": "Is there anyone from Hyderabad earning above 10L?"})
    assert resp2.status_code == 200
    assert "True" in resp2.json()["answer"]


def test_dataset_boolean_non_existence():
    """'Is there nobody from Hyderabad?' -> False (since 15 employees exist)."""
    resp = client.post("/api/query", json={"question": "Is there nobody from Hyderabad?"})
    assert resp.status_code == 200
    assert "False" in resp.json()["answer"]


def test_dataset_boolean_ranking():
    """Highest paid employee is from Hyderabad (Tanvi Iyer)."""
    resp = client.post("/api/query", json={"question": "Is the highest paid employee from Hyderabad?"})
    assert resp.status_code == 200
    assert "True" in resp.json()["answer"]


def test_dataset_boolean_duplicates():
    """Are there two employees with the same salary? (Yes, Tanvi and Arjun both have 2.37M)."""
    resp = client.post("/api/query", json={"question": "Are there two employees with the same salary?"})
    assert resp.status_code == 200
    assert "True" in resp.json()["answer"]


# -----------------------------------------------------------------------------
# 3. HYBRID BOOLEAN QUESTIONS
# -----------------------------------------------------------------------------
def test_hybrid_boolean_true():
    """Ananya is in Hyderabad, which is in Telangana."""
    resp = client.post("/api/query", json={"question": "Does Ananya work in a city that belongs to Telangana?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "HYBRID"
    assert data["dataset_used"] is True
    assert data["general_knowledge_used"] is True
    assert "True" in data["answer"]
    assert "Hyderabad" in data["answer"]
    assert "Telangana" in data["answer"]


def test_hybrid_boolean_false():
    """Rahul is in Delhi, which is not in Tamil Nadu."""
    resp = client.post("/api/query", json={"question": "Does Rahul work in a city that belongs to Tamil Nadu?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "HYBRID"
    assert "False" in data["answer"]
    assert "Delhi" in data["answer"]
    assert "Tamil Nadu" in data["answer"]


# -----------------------------------------------------------------------------
# 4. CONVERSATIONAL BOOLEAN FOLLOW-UP
# -----------------------------------------------------------------------------
def test_conversational_boolean_followup():
    """Turn 1 asks who is highest paid, Turn 2 asks 'Is she from Hyderabad?'."""
    session_id = "test_bool_session"
    resp1 = client.post("/api/query", json={"question": "Who is the highest paid employee?", "session_id": session_id})
    assert resp1.status_code == 200
    assert any(name in resp1.json()["answer"] or name in str(resp1.json()["results"]) for name in ["Ranbir", "Jitesh", "Tanvi", "Singh", "Sharma"])

    resp2 = client.post("/api/query", json={"question": "Is she from Hyderabad?", "session_id": session_id})
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["operation"] == "BOOLEAN_CHECK"
    assert "True" in data["answer"] or "False" in data["answer"]
    assert any(c in data["answer"] for c in ["Bengaluru", "Hyderabad", "Delhi", "Chennai", "Pune"])

