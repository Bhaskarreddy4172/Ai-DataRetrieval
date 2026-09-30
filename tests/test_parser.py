"""Unit tests for QueryParser and conversational follow-ups."""

import pytest
from app.conversation.context import conversation_manager
from app.dataset.loader import DatasetLoader
from app.dataset.metadata import DatasetMetadata
from app.query.parser import QueryParser
from app.query.schema import StructuredQuery, FilterCondition


@pytest.fixture
def parser_setup():
    from pathlib import Path
    data_path = Path("data/sample_customer_data.xlsx")
    if not data_path.exists():
        pytest.skip("sample_customer_data.xlsx was removed per user dataset cleanup")
    loader = DatasetLoader(initial_path=str(data_path))
    parser = QueryParser()
    cols = loader.get_columns()
    schema = DatasetMetadata.get_compact_schema(loader.profile["columns"], loader.dataset_name, len(loader.dataframe))
    return parser, cols, schema


def test_parse_filter_more_than(parser_setup):
    parser, cols, schema = parser_setup
    query = parser.parse("Who spent more than 10000?", cols, schema)
    assert query.operation == "FILTER"
    assert len(query.conditions) == 1
    assert query.conditions[0].column == "Service Amount"
    assert query.conditions[0].operator == ">"
    assert query.conditions[0].value == 10000


def test_parse_count(parser_setup):
    parser, cols, schema = parser_setup
    query = parser.parse("How many customers are there?", cols, schema)
    assert query.operation == "COUNT"


def test_parse_sum(parser_setup):
    parser, cols, schema = parser_setup
    query = parser.parse("What is the total service revenue?", cols, schema)
    assert query.operation == "SUM"
    assert query.target_column == "Service Amount"


def test_parse_city_filter(parser_setup):
    parser, cols, schema = parser_setup
    query = parser.parse("Show all customers from Hyderabad", cols, schema)
    assert query.operation == "FILTER"
    assert any(c.column == "City" and c.value == "Hyderabad" for c in query.conditions)


def test_conversational_followup_context():
    session_id = "test_session_123"
    q1 = StructuredQuery(
        operation="FILTER",
        conditions=[FilterCondition(column="Service Amount", operator=">", value=10000)]
    )
    conversation_manager.add_turn(session_id, "Who spent more than 10000?", q1, 24)

    # Follow-up question: "What about only in Hyderabad?"
    q2 = StructuredQuery(
        operation="FILTER",
        conditions=[FilterCondition(column="City", operator="=", value="Hyderabad")]
    )
    merged = conversation_manager.contextualize_followup("What about only in Hyderabad?", session_id, q2)
    assert len(merged.conditions) == 2
    cols = [c.column for c in merged.conditions]
    assert "Service Amount" in cols
    assert "City" in cols
