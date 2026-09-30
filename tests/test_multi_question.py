"""Tests for Multi-Question Decomposition Engine."""

import pytest
from app.dataset.loader import dataset_loader
from app.query.decomposer import multi_question_decomposer
from pathlib import Path


@pytest.fixture(scope="module", autouse=True)
def load_employees():
    data_path = Path("data/sample_employees.xlsx")
    dataset_loader.load_dataset(data_path)


def test_split_question_compound():
    q = "Who earns > 10L in Hyderabad and what department are they in and who has the best performance?"
    parts = multi_question_decomposer.split_question(q)
    assert len(parts) == 3
    assert "Hyderabad" in parts[0]
    assert "department" in parts[1]
    assert "best performance" in parts[2]


def test_preserve_single_question_multi_condition():
    q = "Show employees with salary > 50000 and department IT"
    parts = multi_question_decomposer.split_question(q)
    assert len(parts) == 1
    assert parts[0] == q


def test_execute_decomposed_employee_query():
    df = dataset_loader.dataframe
    cols = dataset_loader.get_columns()
    schema_info = dataset_loader.schema_intelligence

    q = "How many employees are in IT and what is their average salary?"
    res = multi_question_decomposer.execute_decomposed(q, df, cols, schema_info)
    assert res["is_compound"] is True
    assert len(res["sub_questions"]) == 2
    assert "1." in res["answer"]
    assert "2." in res["answer"]
    assert len(res["sub_answers"]) == 2

