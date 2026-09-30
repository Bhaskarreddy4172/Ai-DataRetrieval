"""Automated Evaluation & Unit Test Suite for Master Agent Orchestrator Architecture."""

import pytest
import pandas as pd
from app.agent import (
    agent_orchestrator,
    context_resolver,
    conversation_memory,
    AgentStateEnum,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "name": ["Ananya Reddy", "Aditya Kulkarni", "Rahul Sharma", "Rahul Verma"],
        "department": ["Engineering", "Marketing", "Sales", "Engineering"],
        "salary": [1490000, 1270000, 850000, 950000],
        "city": ["Hyderabad", "Chennai", "Bangalore", "Hyderabad"],
    })


def test_agent_state_machine_flow(sample_df):
    res = agent_orchestrator.process_request("Who has the highest salary?", session_id="sess_flow", df=sample_df)
    assert res["grounded"]
    assert res["result_count"] >= 1
    assert "Ananya" in res["answer"] or "1490000" in res["answer"]

    # Verify debug trace events
    events = res["debug_trace"]["events"]
    states = [e["state"] for e in events]
    assert AgentStateEnum.RECEIVED in states
    assert AgentStateEnum.UNDERSTANDING in states
    assert AgentStateEnum.CONTEXT_RESOLUTION in states
    assert AgentStateEnum.ROUTING in states
    assert AgentStateEnum.PLANNING in states
    assert AgentStateEnum.TOOL_SELECTION in states
    assert AgentStateEnum.EXECUTING in states
    assert AgentStateEnum.OBSERVING in states
    assert AgentStateEnum.VERIFICATION in states
    assert AgentStateEnum.ANSWER_GENERATION in states
    assert AgentStateEnum.COMPLETED in states


def test_context_and_pronoun_resolution(sample_df):
    # Turn 1
    res1 = agent_orchestrator.process_request("Who has the highest salary?", session_id="sess_pronoun", df=sample_df)
    assert "Ananya" in res1["answer"] or "1490000" in res1["answer"]

    # Turn 2: Pronoun resolution ("Where does she work?")
    resolved_q, meta = context_resolver.resolve("Where does she work?", session_id="sess_pronoun")
    assert "Ananya" in resolved_q or meta.get("resolved_entity") == "Ananya Reddy"

    res2 = agent_orchestrator.process_request("Where does she work?", session_id="sess_pronoun", df=sample_df)
    assert res2["grounded"]


def test_ambiguity_clarification(sample_df):
    res = agent_orchestrator.process_request("Show me Rahul's salary", session_id="sess_ambiguity", df=sample_df)
    assert res["is_clarification"] or "Rahul" in res["answer"]

