"""Comprehensive Test Suite for Hierarchical Parent-Child Dataset Architecture & Universal Retrieval.

Tests:
1. Parent State-Capital Lookups & Abbreviations (TG, TS, AP, KA, BLR, HYD, etc.)
2. Hierarchical Navigation (Main State -> Child Village Aggregations)
3. Direct Child Dataset Queries (Village Names, Village IDs, Literacy, Population, Households)
4. Multi-Child Cross-Dataset Aggregation & Comparisons (All-India Extremes, Two-State Comparisons)
5. Multi-Turn Context & Pronoun Resolution across Hierarchy (State -> Village -> Village Metric)
6. Typo, Phonetic, Hinglish, and Informal Query Robustness
7. Zero-Hallucination Guardrails (Fictional States/Villages)
8. Independent Sentiment Attachment Non-Interference
"""

import pytest
from app.api.routes import process_query, QueryRequest
from app.dataset.loader import dataset_loader
from app.dataset.registry import parent_child_registry
from app.dataset.indexer import dataset_indexer
from app.dataset.spell_checker import dataset_spell_checker
from app.dataset.alias_resolver import entity_alias_resolver
from app.conversation.context import conversation_manager

DEFAULT_TEST_SESSION_ID = "test-parent-child-session"


@pytest.fixture
def session_id() -> str:
    """Provide a deterministic session ID for conversation and retrieval isolation."""
    return DEFAULT_TEST_SESSION_ID


@pytest.fixture(scope="module", autouse=True)
def setup_parent_child_environment():
    """Ensure parent dataset and registry are initialized."""
    dataset_loader.load_sample("states")
    df = dataset_loader.dataframe
    dataset_indexer.build_index(df, "indian_states_capitals.csv")
    dataset_spell_checker.build_vocabulary(df, "indian_states_capitals.csv")
    entity_alias_resolver.index_dataset(df)
    parent_child_registry._initialize_paths()


@pytest.fixture(autouse=True)
def reset_conversation():
    """Ensure clean conversation state for each test."""
    conversation_manager.clear_all()
    yield
    conversation_manager.clear_all()


class TestParentStateCapitalLookups:
    """Test standard main/parent dataset queries."""

    @pytest.mark.parametrize("query,expected_capital", [
        ("What is the capital of Telangana?", "Hyderabad"),
        ("capital of Bihar", "Patna"),
        ("Karnataka capital", "Bengaluru"),
        ("capital of TG", "Hyderabad"),
        ("capital of TS", "Hyderabad"),
        ("capital of AP", "Amaravati"),
        ("capital of KA", "Bengaluru"),
        ("capital of MH", "Mumbai"),
        ("Telangana's capital?", "Hyderabad"),
        ("What is Goa's capital?", "Panaji"),
    ])
    def test_state_to_capital_lookup(self, query: str, expected_capital: str):
        res = process_query(QueryRequest(question=query, session_id=DEFAULT_TEST_SESSION_ID))
        assert res.operation == "LOOKUP"
        assert expected_capital.lower() in res.answer.lower()
        assert res.hierarchy_scope == "MAIN_ONLY"

    @pytest.mark.parametrize("query,expected_state", [
        ("HYD is the capital of which state?", "Telangana"),
        ("BLR is the capital of which state?", "Karnataka"),
        ("BOM is the capital of which state?", "Maharashtra"),
        ("Which state has Bangalore?", "Karnataka"),
        ("Which state has Kolkata?", "West Bengal"),
    ])
    def test_city_reverse_lookup(self, query: str, expected_state: str):
        res = process_query(QueryRequest(question=query, session_id=DEFAULT_TEST_SESSION_ID))
        assert res.operation == "LOOKUP"
        assert expected_state.lower() in res.answer.lower()


class TestHierarchicalMainToChildNavigation:
    """Test navigating from Parent State/Capital down to Child Village Dataset."""

    def test_highest_population_village_in_state(self):
        res = process_query(QueryRequest(question="What is the highest population village in Telangana?", session_id=DEFAULT_TEST_SESSION_ID))
        assert res.operation == "MAX"
        assert "hyderabad_village_01" in res.answer.lower()
        assert "117,305" in res.answer or "117305" in res.answer
        assert res.hierarchy_scope == "MAIN_TO_CHILD"
        assert res.child_dataset == "tg_hyderabad_villages.csv"

    def test_highest_literacy_village_in_bengaluru(self):
        res = process_query(QueryRequest(question="Which village in Bengaluru has maximum literacy rate?", session_id=DEFAULT_TEST_SESSION_ID))
        assert res.operation == "MAX"
        assert res.hierarchy_scope == "MAIN_TO_CHILD"
        assert "bengaluru" in res.answer.lower() or "karnataka" in res.answer.lower()
        assert res.child_dataset == "ka_bengaluru_villages.csv"

    def test_village_count_in_state(self):
        res = process_query(QueryRequest(question="How many villages are in Bihar?", session_id=DEFAULT_TEST_SESSION_ID))
        assert res.operation == "COUNT"
        assert "10" in res.answer
        assert res.hierarchy_scope == "MAIN_TO_CHILD"
        assert res.child_dataset == "br_patna_villages.csv"

    def test_abbreviated_main_to_child(self):
        res1 = process_query(QueryRequest(question="TG highest population village", session_id=DEFAULT_TEST_SESSION_ID))
        assert res1.operation == "MAX"
        assert "hyderabad_village_01" in res1.answer.lower()

        res2 = process_query(QueryRequest(question="BLR max literacy village", session_id=DEFAULT_TEST_SESSION_ID))
        assert res2.operation == "MAX"
        assert "ka_bengaluru_villages.csv" in (res2.child_dataset or "")

    def test_state_code_population_aggregation_gujarat(self):
        """Test 'what is the population gj' against main dataset which has no population column."""
        res = process_query(QueryRequest(question="what is the population gj", session_id=DEFAULT_TEST_SESSION_ID))
        assert res.operation == "SUM"
        assert res.hierarchy_scope == "MAIN_TO_CHILD"
        assert res.child_dataset == "gj_gandhinagar_villages.csv"
        assert "369,332" in res.answer or "369332" in res.answer
        assert "Gujarat" in res.answer
        assert res.grounded is True

    def test_state_code_total_population_aggregation_telangana(self):
        """Test 'what is the total population in tg' against main dataset."""
        res = process_query(QueryRequest(question="what is the total population in tg", session_id=DEFAULT_TEST_SESSION_ID))
        assert res.operation == "SUM"
        assert res.hierarchy_scope == "MAIN_TO_CHILD"
        assert res.child_dataset == "tg_hyderabad_villages.csv"
        assert "919,813" in res.answer or "919813" in res.answer
        assert "Telangana" in res.answer
        assert res.grounded is True


class TestDirectChildDatasetQueries:
    """Test querying child dataset entities and attributes directly."""

    def test_village_literacy_rate(self):
        res = process_query(QueryRequest(question="What is the literacy rate of Hyderabad_Village_01?", session_id=DEFAULT_TEST_SESSION_ID))
        assert res.operation == "LOOKUP"
        assert "96.9" in res.answer
        assert res.hierarchy_scope in ["CHILD_ONLY", "MAIN_TO_CHILD"]

    def test_village_id_lookup(self):
        res = process_query(QueryRequest(question="What is the population of TG-001?", session_id=DEFAULT_TEST_SESSION_ID))
        assert res.operation == "LOOKUP"
        assert "117,305" in res.answer or "117305" in res.answer

    def test_village_households(self):
        res = process_query(QueryRequest(question="How many households are in Hyderabad_Village_02?", session_id=DEFAULT_TEST_SESSION_ID))
        assert res.operation == "LOOKUP"
        assert "22,551" in res.answer or "22551" in res.answer


class TestMultiChildCrossDatasetComparison:
    """Test cross-child aggregations and comparisons across all registered states."""

    def test_all_india_highest_population_village(self):
        res = process_query(QueryRequest(question="Which capital has the village with highest population across all states?", session_id=DEFAULT_TEST_SESSION_ID))
        assert res.operation == "MAX"
        assert res.hierarchy_scope == "MULTI_CHILD_COMPARISON"
        assert "kolkata_village_05" in res.answer.lower()
        assert "west bengal" in res.answer.lower()

    def test_all_india_average_literacy(self):
        res = process_query(QueryRequest(question="What is the average village literacy rate in India?", session_id=DEFAULT_TEST_SESSION_ID))
        assert res.operation == "AVERAGE"
        assert res.hierarchy_scope == "MULTI_CHILD_COMPARISON"
        assert "%" in res.answer or "literacy" in res.answer.lower()

    def test_two_state_village_population_comparison(self):
        res = process_query(QueryRequest(question="Compare total village population of Telangana vs Karnataka", session_id=DEFAULT_TEST_SESSION_ID))
        assert res.operation == "COMPARE"
        assert res.hierarchy_scope == "MULTI_CHILD_COMPARISON"
        assert "telangana" in res.answer.lower()
        assert "karnataka" in res.answer.lower()


class TestMultiTurnConversationalHierarchy:
    """Test follow-up queries resolving context across parent and child scopes."""

    def test_parent_to_child_followup(self):
        session_id = "test_hierarchical_session_001"
        conversation_manager.clear_session(session_id)

        # Turn 1: State capital lookup (Parent)
        t1 = process_query(QueryRequest(question="What is the capital of Telangana?", session_id=session_id))
        assert "hyderabad" in t1.answer.lower()

        # Turn 2: Follow-up navigation to child (State -> Largest village)
        t2 = process_query(QueryRequest(question="What is its highest population village?", session_id=session_id))
        assert t2.operation == "MAX"
        assert "hyderabad_village_01" in t2.answer.lower()

        # Turn 3: Follow-up on that specific village (Village -> Female population)
        t3 = process_query(QueryRequest(question="What is its female population?", session_id=session_id))
        assert t3.operation == "LOOKUP"
        assert "59,260" in t3.answer or "59260" in t3.answer


class TestTypoAndInformalRobustness:
    """Test typos, phonetic matching, and informal Hinglish phrasings."""

    def test_typo_in_state_village_query(self):
        res = process_query(QueryRequest(question="telengana highest population village", session_id=DEFAULT_TEST_SESSION_ID))
        assert res.operation == "MAX"
        assert "hyderabad_village_01" in res.answer.lower()

    def test_hinglish_state_village_query(self):
        res = process_query(QueryRequest(question="telangana ka highest population village konsa hai?", session_id=DEFAULT_TEST_SESSION_ID))
        assert res.operation == "MAX"
        assert "hyderabad_village_01" in res.answer.lower()


class TestZeroHallucinationGuardrails:
    """Test that fictional entities return NO_MATCH with zero hallucinations."""

    def test_fictional_state(self):
        res = process_query(QueryRequest(question="What is the capital of Wakanda?", session_id=DEFAULT_TEST_SESSION_ID))
        assert res.operation == "NO_MATCH"
        assert res.result_count == 0

    def test_fictional_village(self):
        res = process_query(QueryRequest(question="What is the literacy rate of Atlantis_Village_99?", session_id=DEFAULT_TEST_SESSION_ID))
        assert res.operation == "NO_MATCH"
        assert res.result_count == 0


class TestSentimentAnalysisNonInterference:
    """Test that independent sentiment analysis classifies accurately without modifying factual retrieval."""

    def test_positive_query_attachment(self):
        res = process_query(QueryRequest(question="Thank you very much! What is the capital of Telangana?", session_id=DEFAULT_TEST_SESSION_ID))
        assert res.sentiment == "POSITIVE"
        assert res.sentiment_score is not None, "Expected sentiment_score to be computed"
        assert res.sentiment_score > 0
        assert "hyderabad" in res.answer.lower()
        assert res.operation == "LOOKUP"

    def test_negative_query_attachment(self):
        res = process_query(QueryRequest(question="This is terrible and horrible, what is the capital of Goa?", session_id=DEFAULT_TEST_SESSION_ID))
        assert res.sentiment == "NEGATIVE"
        assert res.sentiment_score is not None, "Expected sentiment_score to be computed"
        assert res.sentiment_score < 0
        assert "panaji" in res.answer.lower()
        assert res.operation == "LOOKUP"

    def test_neutral_factual_query(self):
        res = process_query(QueryRequest(question="What is the population of TG-001?", session_id=DEFAULT_TEST_SESSION_ID))
        assert res.sentiment == "NEUTRAL"
        assert res.sentiment_score is not None, "Expected sentiment_score to be computed"
        assert res.sentiment_score == 0.0
        assert "117,305" in res.answer or "117305" in res.answer
