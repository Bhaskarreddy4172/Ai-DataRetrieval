"""Automated Evaluation & Unit Test Suite for OpenAI-Style Tool Architecture."""

import pytest
import pandas as pd
from app.llm import model_manager, ToolDefinition, ToolCall, ToolResult
from app.tools import (
    tool_registry,
    SearchDatasetTool,
    FilterDatasetTool,
    AggregateDatasetTool,
    SortDatasetTool,
    TopNTool,
    BottomNTool,
    GroupByTool,
    CompareTool,
    LookupTool,
    StatisticsTool,
    SchemaInspectionTool,
    RelationshipTool,
)
from app.agent import dataset_agent, hallucination_firewall, conversation_memory


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "David", "Emma"],
        "department": ["Engineering", "Sales", "Engineering", "Marketing", "Engineering"],
        "salary": [100000, 80000, 120000, 75000, 110000],
        "city": ["Bangalore", "Hyderabad", "Bangalore", "Mumbai", "Hyderabad"],
    })


def test_tool_registry_registration():
    definitions = tool_registry.get_definitions()
    assert len(definitions) >= 11
    tool_names = [d.name for d in definitions]
    assert "search_dataset" in tool_names
    assert "filter_dataset" in tool_names
    assert "aggregate_dataset" in tool_names
    assert "sort_dataset" in tool_names
    assert "top_n" in tool_names
    assert "bottom_n" in tool_names
    assert "group_by" in tool_names
    assert "compare" in tool_names
    assert "lookup" in tool_names
    assert "statistics" in tool_names
    assert "schema_inspection" in tool_names
    assert "relationship" in tool_names


def test_aggregate_dataset_tool(sample_df):
    tool = tool_registry.get_tool("aggregate_dataset")
    assert tool is not None

    # Test MAX
    res_max = tool.execute(sample_df, {"operation": "MAX", "column": "salary"})
    assert res_max.success
    assert res_max.aggregation is not None
    assert res_max.aggregation["value"] == 120000

    # Test MIN
    res_min = tool.execute(sample_df, {"operation": "MIN", "column": "salary"})
    assert res_min.success
    assert res_min.aggregation is not None
    assert res_min.aggregation["value"] == 75000

    # Test AVG
    res_avg = tool.execute(sample_df, {"operation": "AVG", "column": "salary"})
    assert res_avg.success
    assert res_avg.aggregation is not None
    assert res_avg.aggregation["value"] == 97000.0

    # Test COUNT
    res_count = tool.execute(sample_df, {"operation": "COUNT", "column": "name"})
    assert res_count.success
    assert res_count.aggregation is not None
    assert res_count.aggregation["value"] == 5


def test_filter_dataset_tool(sample_df):
    tool = tool_registry.get_tool("filter_dataset")
    assert tool is not None

    res = tool.execute(sample_df, {
        "conditions": [{"column": "department", "operator": "EQUALS", "value": "Engineering"}]
    })
    assert res.success
    assert res.row_count == 3
    for r in res.results:
        assert r["department"] == "Engineering"


def test_top_n_and_bottom_n_tools(sample_df):
    top_tool = tool_registry.get_tool("top_n")
    assert top_tool is not None
    res_top = top_tool.execute(sample_df, {"column": "salary", "n": 2})
    assert res_top.success
    assert len(res_top.results) == 2
    assert res_top.results[0]["name"] == "Charlie"

    bottom_tool = tool_registry.get_tool("bottom_n")
    assert bottom_tool is not None
    res_bottom = bottom_tool.execute(sample_df, {"column": "salary", "n": 1})
    assert res_bottom.success
    assert res_bottom.results[0]["name"] == "David"


def test_group_by_tool(sample_df):
    tool = tool_registry.get_tool("group_by")
    assert tool is not None
    res = tool.execute(sample_df, {"group_column": "department", "operation": "COUNT"})
    assert res.success
    assert len(res.results) == 3


def test_compare_tool(sample_df):
    tool = tool_registry.get_tool("compare")
    assert tool is not None
    res = tool.execute(sample_df, {
        "entity_column": "name",
        "entity_1": "Charlie",
        "entity_2": "Bob",
        "metric_column": "salary",
    })
    assert res.success
    assert res.aggregation is not None
    assert res.aggregation["val_1"] == 120000
    assert res.aggregation["val_2"] == 80000
    assert res.aggregation["difference"] == 40000.0


def test_lookup_tool(sample_df):
    tool = tool_registry.get_tool("lookup")
    assert tool is not None
    res = tool.execute(sample_df, {
        "entity_name": "Charlie",
        "entity_column": "name",
        "target_attribute": "city",
    })
    assert res.success
    assert res.aggregation is not None
    assert res.aggregation["value"] == "Bangalore"


def test_statistics_and_schema_tools(sample_df):
    schema_tool = tool_registry.get_tool("schema_inspection")
    assert schema_tool is not None
    res_schema = schema_tool.execute(sample_df, {})
    assert res_schema.success
    assert len(res_schema.results) == 4

    stats_tool = tool_registry.get_tool("statistics")
    assert stats_tool is not None
    res_stats = stats_tool.execute(sample_df, {"column": "salary"})
    assert res_stats.success
    assert res_stats.aggregation is not None
    assert res_stats.aggregation["salary"]["mean"] == 97000.0


def test_hallucination_firewall():
    results = [{"name": "Charlie", "salary": 120000, "city": "Bangalore"}]
    aggregation = {"value": 120000}

    # Verified answer
    valid, disc = hallucination_firewall.verify("Charlie has the highest salary of 120000.", results, aggregation)
    assert valid
    assert len(disc) == 0

    # Hallucinated number
    valid_h, disc_h = hallucination_firewall.verify("Charlie has the highest salary of 5000000.", results, aggregation)
    assert not valid_h
    assert len(disc_h) > 0


def test_dataset_agent_query(sample_df):
    res = dataset_agent.process_query("Who has the highest salary?", session_id="test_agent", df=sample_df)
    assert res["grounded"]
    assert res["result_count"] >= 1
    assert "Charlie" in res["answer"] or "120000" in res["answer"]
    assert "debug_trace" in res

