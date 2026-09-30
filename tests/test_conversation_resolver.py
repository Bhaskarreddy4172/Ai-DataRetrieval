"""Unit tests for conversation context and pronoun resolution."""

import pytest
from app.conversation.resolver import conversation_resolver
from app.query.schema import FilterCondition, StructuredQuery


def test_followup_city_switch():
    cols = ["Customer Name", "City", "Service Amount", "Vehicle"]
    last_query = StructuredQuery(
        operation="AVERAGE",
        target_column="Service Amount",
        conditions=[FilterCondition(column="City", operator="=", value="Mumbai")]
    )
    history = [{
        "question": "What is the average service amount in Mumbai?",
        "query": last_query,
        "results": [],
        "result_count": 5
    }]

    clarified, resolved = conversation_resolver.resolve("What about Hyderabad?", history, cols)
    assert resolved is not None
    assert resolved.operation == "AVERAGE"
    assert resolved.target_column == "Service Amount"
    assert any(c.column == "City" and c.value == "Hyderabad" for c in resolved.conditions)


def test_pronoun_resolution():
    cols = ["Employee Name", "Department", "Salary", "City"]
    history = [{
        "question": "Who has the highest salary?",
        "query": StructuredQuery(operation="MAX", target_column="Salary"),
        "results": [{"Employee Name": "Neha Patel", "Department": "Engineering", "Salary": 1200000}],
        "result_count": 1
    }]

    clarified, resolved = conversation_resolver.resolve("What is their salary?", history, cols)
    assert "Neha Patel" in clarified

