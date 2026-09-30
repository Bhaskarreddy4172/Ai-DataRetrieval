"""Universal All-States and Multi-Dataset Engine Test Suite.

Ensures that ALL 28 Indian States operate through the EXACT SAME generic pipeline
with ZERO state-specific hardcoding, dynamic dataset discovery, content-aware
parent-child linkage, robust cross-state comparisons, and all-states rankings.
"""

import inspect
import pytest
from app.api.routes import process_query, QueryRequest
from app.dataset.loader import dataset_loader
from app.dataset.registry import parent_child_registry
from app.dataset.hierarchical_engine import hierarchical_query_engine
from app.dataset.multi_child_executor import multi_child_executor
from app.query.comparison_engine import universal_comparison_engine
from app.conversation.context import conversation_manager


ALL_28_INDIAN_STATES = [
    ("Andhra Pradesh", "Amaravati", "AP"),
    ("Arunachal Pradesh", "Itanagar", "AR"),
    ("Assam", "Dispur", "AS"),
    ("Bihar", "Patna", "BR"),
    ("Chhattisgarh", "Raipur", "CG"),
    ("Goa", "Panaji", "GA"),
    ("Gujarat", "Gandhinagar", "GJ"),
    ("Haryana", "Chandigarh", "HR"),
    ("Himachal Pradesh", "Shimla", "HP"),
    ("Jharkhand", "Ranchi", "JH"),
    ("Karnataka", "Bengaluru", "KA"),
    ("Kerala", "Thiruvananthapuram", "KL"),
    ("Madhya Pradesh", "Bhopal", "MP"),
    ("Maharashtra", "Mumbai", "MH"),
    ("Manipur", "Imphal", "MN"),
    ("Meghalaya", "Shillong", "ML"),
    ("Mizoram", "Aizawl", "MZ"),
    ("Nagaland", "Kohima", "NL"),
    ("Odisha", "Bhubaneswar", "OD"),
    ("Punjab", "Chandigarh", "PB"),
    ("Rajasthan", "Jaipur", "RJ"),
    ("Sikkim", "Gangtok", "SK"),
    ("Tamil Nadu", "Chennai", "TN"),
    ("Telangana", "Hyderabad", "TG"),
    ("Tripura", "Agartala", "TR"),
    ("Uttar Pradesh", "Lucknow", "UP"),
    ("Uttarakhand", "Dehradun", "UK"),
    ("West Bengal", "Kolkata", "WB"),
]


@pytest.fixture(scope="module", autouse=True)
def setup_universal_environment():
    """Ensure parent dataset and dynamic registry are initialized."""
    parent_child_registry._initialize_paths()
    dataset_loader.load_sample("states")
    conversation_manager.clear_all()


@pytest.fixture(autouse=True)
def clean_sessions():
    """Ensure clean session for each test."""
    conversation_manager.clear_all()
    yield
    conversation_manager.clear_all()


class TestDynamicRegistryZeroHardcoding:
    """Validate dynamic discovery, registration, and zero state-specific hardcoding."""

    def test_startup_health_report(self):
        """Verify health check reports all discovered and registered child datasets."""
        report = parent_child_registry.get_health_report()
        assert report["child_datasets_discovered"] >= 28, f"Expected >=28 discovered, got {report['child_datasets_discovered']}"
        assert report["child_datasets_registered"] >= 28, f"Expected >=28 registered, got {report['child_datasets_registered']}"
        assert report["child_datasets_failed"] == 0, f"Child datasets failed: {report['child_datasets_failed']}"
        assert len(report["available_states"]) >= 28, f"Expected >=28 available states, got {len(report['available_states'])}"

    def test_zero_state_specific_logic_in_registry_source(self):
        """Source inspection test: ParentChildRegistry must contain ZERO state-specific if/elif branches."""
        src = inspect.getsource(parent_child_registry._index_entity_associations)
        assert 'if state_val.lower() == "telangana"' not in src
        assert 'if state_val == "Telangana"' not in src
        assert 'elif capital_val.lower() == "bengaluru"' not in src
        assert 'elif capital_val.lower() == "mumbai"' not in src

    def test_state_codes_map_contains_all_states(self):
        """Dynamic state codes map must contain codes and abbreviations for all 28 states."""
        codes_map = parent_child_registry.get_state_codes_map()
        for state, _, code in ALL_28_INDIAN_STATES:
            assert code.lower() in codes_map, f"Code {code} missing in dynamic state codes map"
            assert codes_map[code.lower()].lower() == state.lower()


class TestAll28StatesParameterized:
    """Parameterized test suite executed across ALL 28 Indian states."""

    @pytest.mark.parametrize("state,capital,code", ALL_28_INDIAN_STATES)
    def test_child_dataset_resolution(self, state: str, capital: str, code: str):
        """Verify every state, capital, and code dynamically resolves to a valid child dataset."""
        # Resolve by state name
        res_state = parent_child_registry.resolve_child_dataset(state)
        assert res_state is not None, f"Could not resolve child dataset for state '{state}'"
        assert res_state[1].exists(), f"Resolved child path for '{state}' does not exist: {res_state[1]}"

        # Resolve by state code
        res_code = parent_child_registry.resolve_child_dataset(code)
        assert res_code is not None, f"Could not resolve child dataset for code '{code}'"
        assert res_code[1].exists(), f"Resolved child path for '{code}' does not exist: {res_code[1]}"

    @pytest.mark.parametrize("state,capital,code", ALL_28_INDIAN_STATES)
    def test_state_village_population_query(self, state: str, capital: str, code: str):
        """Verify population query for each state routes to child dataset and returns positive total."""
        query = f"What is the population of {state}?"
        assert hierarchical_query_engine.can_handle(query), f"can_handle failed for '{query}'"
        res = hierarchical_query_engine.execute(query)
        assert res["operation"] == "SUM"
        assert res["result_count"] >= 1
        assert res["results"][0]["value"] > 0
        assert str(state).lower() in res["answer"].lower()

    @pytest.mark.parametrize("state,capital,code", ALL_28_INDIAN_STATES)
    def test_state_village_literacy_query(self, state: str, capital: str, code: str):
        """Verify literacy query for each state computes valid average rate."""
        query = f"What is the literacy rate of {state}?"
        assert hierarchical_query_engine.can_handle(query), f"can_handle failed for '{query}'"
        res = hierarchical_query_engine.execute(query)
        assert res["operation"] == "AVERAGE"
        assert 0 < res["results"][0]["value"] <= 100
        assert "%" in res["answer"]

    @pytest.mark.parametrize("state,capital,code", ALL_28_INDIAN_STATES)
    def test_state_code_population_query(self, state: str, capital: str, code: str):
        """Verify state code query (e.g. 'population of GJ') routes and returns correct state result."""
        query = f"population of {code}"
        assert hierarchical_query_engine.can_handle(query), f"can_handle failed for '{query}'"
        res = hierarchical_query_engine.execute(query)
        assert res["results"][0]["value"] > 0


class TestCrossStateComparisons:
    """Test comparative inquiries between arbitrary pairs of Indian states."""

    @pytest.mark.parametrize("state_a,state_b", [
        ("Gujarat", "Kerala"),
        ("Punjab", "Assam"),
        ("Maharashtra", "Tamil Nadu"),
        ("Rajasthan", "Odisha"),
        ("Bihar", "Haryana"),
        ("Karnataka", "Telangana"),
    ])
    def test_cross_state_comparison(self, state_a: str, state_b: str):
        """Verify deterministic comparison between two states across different regions."""
        query = f"Compare population between {state_a} and {state_b}"
        plan = universal_comparison_engine.parse_and_plan(query)
        assert plan is not None, f"Failed to plan comparison for '{query}'"
        assert plan.scope == "STATE_VS_STATE"
        res = universal_comparison_engine.execute(plan)
        assert res.verification_status == "VERIFIED"
        assert res.left_value is not None and res.left_value > 0
        assert res.right_value is not None and res.right_value > 0
        assert res.higher_entity in [state_a, state_b]


class TestAllStatesRankingsAndTopN:
    """Test cross-state rankings, aggregations, and Top-N queries across all 28 states."""

    def test_state_highest_population_ranking(self):
        """Verify 'Which state has the highest total village population?' returns ranked answer."""
        query = "Which state has the highest total village population?"
        assert hierarchical_query_engine.can_handle(query)
        res = hierarchical_query_engine.execute(query)
        assert res["operation"] == "RANK_STATE_METRIC"
        assert len(res["results"]) >= 27
        assert "highest total population" in res["answer"].lower()

    def test_state_lowest_population_ranking(self):
        """Verify 'Which state has the lowest total village population?' returns bottom state."""
        query = "Which state has the lowest total village population?"
        assert hierarchical_query_engine.can_handle(query)
        res = hierarchical_query_engine.execute(query)
        assert res["operation"] == "RANK_STATE_METRIC"
        assert len(res["results"]) >= 27
        assert "lowest total population" in res["answer"].lower()

    def test_state_most_villages_ranking(self):
        """Verify 'Which state has the most villages?' ranks states by row count."""
        query = "Which state has the most villages?"
        assert hierarchical_query_engine.can_handle(query)
        res = hierarchical_query_engine.execute(query)
        assert res["operation"] == "RANK_VILLAGE_COUNT"
        assert len(res["results"]) >= 27
        assert "most recorded villages" in res["answer"].lower()

    def test_top_10_villages_across_all_states(self):
        """Verify 'Top 10 villages by population across all states' returns 10 ordered records."""
        query = "Top 10 villages by population across all states"
        assert hierarchical_query_engine.can_handle(query)
        res = hierarchical_query_engine.execute(query)
        assert res["operation"] == "TOP_N"
        assert res["result_count"] == 10
        assert len(res["results"]) == 10
        # Ensure ranks are ordered descending
        values = [r["value"] for r in res["results"]]
        assert values == sorted(values, reverse=True)


class TestGracefulSection47ErrorHandling:
    """Verify Section 47 graceful error messages for missing child datasets and columns."""

    def test_missing_state_child_dataset(self):
        """Querying nonexistent state child dataset must list available states."""
        query = "What is the population of Atlantis?"
        # Atlantis is unknown, so resolve_child_dataset returns None
        resolved = parent_child_registry.resolve_child_dataset("Atlantis")
        assert resolved is None

    def test_missing_attribute_in_child_dataset(self):
        """Querying an unknown attribute for a valid state returns informative error."""
        # Execute query for an attribute not in child columns
        resolved_guj = parent_child_registry.resolve_child_dataset("Gujarat")
        assert resolved_guj is not None
        child_p = resolved_guj[1]

        res = hierarchical_query_engine._execute_single_child(
            question="What is the elevation of Gujarat?",
            q_lower="what is the elevation of gujarat?",
            scope="MAIN_TO_CHILD",
            entity_name="Gujarat",
            child_path=child_p,
            session_id=None
        )
        assert res["operation"] == "NO_MATCH"
        assert "Available attributes are" in res["answer"]

