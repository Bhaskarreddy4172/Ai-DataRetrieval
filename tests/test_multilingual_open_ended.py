"""Comprehensive tests for Universal Open-Ended Multilingual Dataset Chatbot.

Verifies:
1. Open-ended unseen question understanding (no memorized question dependency)
2. Multilingual & code-switching resilience (Hinglish, Telugu-English, Indian English)
3. Broken grammar and word-order independence ("salary max who?", "who highest salary?")
4. Zero-shot arbitrary dataset testing (completely unseen schema: University Students GPA)
"""

import tempfile
from pathlib import Path
import pandas as pd
import pytest
from app.ai.intent_parser import intent_parser
from app.dataset.loader import dataset_loader
from app.query.planner import query_planner
from app.query.executor import query_executor
from app.utils.normalization import normalize_question


def test_open_ended_unseen_questions_intent():
    """Verify system recognizes semantic intent across diverse unseen phrasings."""
    cols = ["Employee_Name", "Department", "Salary", "City"]

    unseen_top_n = [
        "who is getting more money?",
        "tell me the top paid person",
        "which employee makes maximum?",
        "who takes biggest salary?",
        "can u tell me who gets paid highest",
        "who haz highest salry?",
        "who is earning most amount?",
        "highest paid guy?",
        "salary max who?",
        "highest salary who?",
        "who salary highest?",
        "who is highest salary wala employee?",
        "which employee ki salary highest hai?",
        "sabse jyada salary kiski hai?",
    ]

    for q in unseen_top_n:
        norm_q = normalize_question(q)
        intent, _ = intent_parser.classify_with_rules(norm_q, cols)
        assert intent == "TOP_N", f"Failed to detect TOP_N for: '{q}' (normalized: '{norm_q}', detected: '{intent}')"


def test_code_switched_and_broken_grammar_normalization():
    """Verify multilingual normalizer handles Hindi/Telugu/Broken English idioms."""
    cases = [
        ("TS ka capital kya hai?", ["capital", "of", "TS"]),
        ("telangana capital enti?", ["capital", "of", "telangana"]),
        ("Hyderabad kis state mein hai?", ["state", "Hyderabad"]),
        ("hyd which state?", ["which state is hyd in"]),
        ("tell me hyd belongs which state", ["which state does hyd belong to"]),
        ("who highest salary?", ["who has highest salary"]),
        ("salary max who?", ["who has highest salary"]),
        ("what salary ananya have?", ["what is ananya's salary"]),
        ("where ananya working?", ["where does ananya work"]),
    ]

    for raw, expected_tokens in cases:
        norm = normalize_question(raw)
        norm_lower = norm.lower()
        for token in expected_tokens:
            assert token.lower() in norm_lower, f"Token '{token}' missing from normalized '{norm}' for raw '{raw}'"


def test_states_dataset_code_switching():
    """Verify state & capital questions answer correctly with code-switched inputs."""
    # Ensure active dataset is indian_states_capitals.csv
    states_path = Path("data/indian_states_capitals.csv")
    if states_path.exists():
        dataset_loader.load_dataset(states_path)

    cols = dataset_loader.get_columns()
    df = dataset_loader.dataframe

    # 1. "TS ka capital kya hai?"
    q1 = "TS ka capital kya hai?"
    plan1 = query_planner.plan_query(q1, cols)
    res1 = query_executor.execute(plan1, df)
    ans1 = res1.get("results", [])
    assert len(ans1) >= 1
    assert any("Hyderabad" in str(r.values()) for r in ans1)

    # 2. "telangana capital enti?"
    q2 = "telangana capital enti?"
    plan2 = query_planner.plan_query(q2, cols)
    res2 = query_executor.execute(plan2, df)
    ans2 = res2.get("results", [])
    assert len(ans2) >= 1
    assert any("Hyderabad" in str(r.values()) for r in ans2)

    # 3. "hyd which state?"
    q3 = "hyd which state?"
    plan3 = query_planner.plan_query(q3, cols)
    res3 = query_executor.execute(plan3, df)
    ans3 = res3.get("results", [])
    assert len(ans3) >= 1
    assert any("Telangana" in str(r.values()) for r in ans3)


def test_zero_shot_unseen_dataset_generalization():
    """Upload a completely new, unseen dataset (Student GPA) and test arbitrary questions."""
    original_dataset_name = dataset_loader.dataset_name

    # Create temporary student dataset
    student_data = pd.DataFrame({
        "Student_ID": ["S101", "S102", "S103", "S104", "S105"],
        "Student_Name": ["Alice Smith", "Bob Jones", "Charlie Brown", "Diana Prince", "Evan Wright"],
        "Major": ["Computer Science", "Mathematics", "Physics", "Computer Science", "Biology"],
        "GPA": [3.95, 3.40, 3.85, 3.90, 3.10],
        "Credits_Completed": [110, 85, 95, 105, 60],
    })

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8") as f:
        student_data.to_csv(f.name, index=False)
        temp_csv_path = Path(f.name)

    try:
        # Load unseen dataset
        success, msg = dataset_loader.load_dataset(temp_csv_path, custom_name="university_students.csv")
        assert success is True
        df = dataset_loader.dataframe
        cols = dataset_loader.get_columns()

        # Query 1: Unseen metric extreme — "Who has the highest GPA?"
        q1 = "Who has the highest GPA?"
        plan1 = query_planner.plan_query(q1, cols)
        assert plan1.operation in {"TOP_N", "MAX"}
        assert plan1.sort_column == "GPA" or plan1.target_column == "GPA"
        res1 = query_executor.execute(plan1, df)
        assert len(res1["results"]) == 1
        assert res1["results"][0]["Student_Name"] == "Alice Smith"

        # Query 2: Broken / inverted syntax — "GPA max who?"
        q2 = "GPA max who?"
        plan2 = query_planner.plan_query(q2, cols)
        assert plan2.operation in {"TOP_N", "MAX"}
        res2 = query_executor.execute(plan2, df)
        assert len(res2["results"]) == 1
        assert res2["results"][0]["Student_Name"] == "Alice Smith"

        # Query 3: Lookup — "Major of Charlie Brown"
        q3 = "Major of Charlie Brown"
        plan3 = query_planner.plan_query(q3, cols)
        assert plan3.operation == "LOOKUP"
        res3 = query_executor.execute(plan3, df)
        assert len(res3["results"]) >= 1
        assert res3["results"][0]["Major"] == "Physics"

        # Query 4: Filtering — "Which students are in Computer Science?"
        q4 = "Which students are in Computer Science?"
        plan4 = query_planner.plan_query(q4, cols)
        res4 = query_executor.execute(plan4, df)
        names = [r["Student_Name"] for r in res4["results"]]
        assert "Alice Smith" in names
        assert "Diana Prince" in names
        assert len(names) == 2

        # Query 5: Metric threshold — "Students with GPA greater than 3.8"
        q5 = "Students with GPA greater than 3.8"
        plan5 = query_planner.plan_query(q5, cols)
        res5 = query_executor.execute(plan5, df)
        gpas = [r["GPA"] for r in res5["results"]]
        assert all(g > 3.8 for g in gpas)
        assert len(res5["results"]) == 2

    finally:
        # Cleanup temp file
        temp_csv_path.unlink(missing_ok=True)
        # Restore original dataset
        states_path = Path("data/indian_states_capitals.csv")
        if states_path.exists():
            dataset_loader.load_dataset(states_path)
