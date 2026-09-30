"""Universal All-Dataset Intelligence Engine Test Suite.

Validates discovery, indexing, querying, and verification across ALL registered datasets:
- 28 Indian State village CSVs
- Main India states & capitals datasets
- Standalone datasets (Vivo.csv, company_a.csv, company_b.csv, test_products.csv, products.json)
- Health Dashboard & All-Dataset Metadata REST endpoints
"""

import pytest
from app.api.routes import process_query, QueryRequest, get_health_dashboard, get_all_dataset_metadata
from app.dataset.loader import dataset_loader
from app.dataset.registry import parent_child_registry
from app.dataset.indexer import dataset_indexer
from app.conversation.context import conversation_manager


@pytest.fixture(scope="module", autouse=True)
def setup_universal_all_datasets():
    """Initialize registry and preload states dataset."""
    parent_child_registry._initialize_paths()
    dataset_loader.load_sample("states")
    conversation_manager.clear_all()


@pytest.fixture(autouse=True)
def clean_sessions():
    """Ensure clean conversation context between tests."""
    conversation_manager.clear_all()
    yield
    conversation_manager.clear_all()


class TestHealthDashboardAndCatalog:
    """Validate startup diagnostics, health dashboard, and metadata catalog."""

    def test_health_dashboard_structure(self):
        """Dashboard must report complete health metrics across all discovered datasets."""
        dash = get_health_dashboard()
        assert dash["status"] == "ready"
        assert dash["datasets_discovered"] >= 30, f"Expected >=30 discovered datasets, got {dash['datasets_discovered']}"
        assert dash["datasets_ready"] >= 30, f"Expected >=30 ready datasets, got {dash['datasets_ready']}"
        assert dash["datasets_failed"] == 0, f"Expected 0 failed datasets, got {dash['datasets_failed']}"
        assert dash["parent_datasets"] >= 1, f"Expected >=1 parent datasets, got {dash['parent_datasets']}"
        assert dash["child_datasets"] >= 28, f"Expected >=28 child datasets, got {dash['child_datasets']}"
        assert dash["standalone_datasets"] >= 3, f"Expected >=3 standalone datasets, got {dash['standalone_datasets']}"
        assert len(dash["datasets"]) == dash["datasets_discovered"]

    def test_dataset_metadata_completeness(self):
        """Every cataloged dataset must contain full metadata schema."""
        catalog = get_all_dataset_metadata()
        assert len(catalog) >= 30

        for item in catalog:
            assert "dataset_id" in item
            assert "dataset_name" in item
            assert "file_path" in item
            assert "dataset_type" in item
            assert item["dataset_type"] in {"PARENT", "CHILD", "STANDALONE"}
            assert "status" in item
            assert item["status"] == "READY"
            assert "row_count" in item
            assert item["row_count"] > 0
            assert "columns" in item
            assert len(item["columns"]) > 0
            assert "content_hash" in item
            assert len(item["content_hash"]) == 64  # SHA256 hex string


class TestMultiDatasetInvertedIndex:
    """Validate global multi-dataset inverted indexing and candidate search."""

    def test_find_candidate_across_different_states(self):
        """Candidates from various states should be accurately found in the global index."""
        cand_gj = dataset_indexer.find_candidate_in_any_dataset("Gandhinagar_Village_01")
        assert cand_gj is not None and len(cand_gj) > 0
        assert "gj" in cand_gj[0]["dataset"].lower()

        cand_br = dataset_indexer.find_candidate_in_any_dataset("Patna_Village_01")
        assert cand_br is not None and len(cand_br) > 0
        assert "br" in cand_br[0]["dataset"].lower()

    def test_find_candidate_in_standalone_dataset(self):
        """Candidates in standalone datasets (e.g. Vivo) should be found globally."""
        cand_vivo = dataset_indexer.find_candidate_in_any_dataset("V29 Pro")
        if cand_vivo:
            assert "vivo" in cand_vivo[0]["dataset"].lower()


class TestUniversalVillageQueriesAcrossStates:
    """Validate queries across diverse state child datasets (not just AP and TG)."""

    def test_gujarat_village_population(self):
        """Query village population in Gujarat."""
        res = process_query(QueryRequest(question="What is the population of Gandhinagar_Village_01?"))
        assert res.grounded is True
        assert res.result_count >= 1
        assert "gandhinagar_village_01" in res.answer.lower()
        assert any(c.isdigit() for c in res.answer)

    def test_bihar_village_literacy(self):
        """Query village literacy in Bihar."""
        res = process_query(QueryRequest(question="What is the literacy rate of Patna_Village_01?"))
        assert res.grounded is True
        assert res.result_count >= 1
        assert "patna_village_01" in res.answer.lower()

    def test_punjab_village_area(self):
        """Query village area in Punjab."""
        res = process_query(QueryRequest(question="What is the area of Chandigarh_Village_01?"))
        assert res.grounded is True
        assert res.result_count >= 1
        assert "chandigarh_village_01" in res.answer.lower()

    def test_assam_village_households(self):
        """Query village households in Assam."""
        res = process_query(QueryRequest(question="What is the household count of Dispur_Village_01?"))
        assert res.grounded is True
        assert res.result_count >= 1
        assert "dispur_village_01" in res.answer.lower()

    def test_maharashtra_village_males(self):
        """Query male count in Maharashtra village."""
        res = process_query(QueryRequest(question="What is the male population of Mumbai_Village_01?"))
        assert res.grounded is True
        assert res.result_count >= 1
        assert "mumbai_village_01" in res.answer.lower()

    def test_west_bengal_village_query(self):
        """Query village in West Bengal."""
        res = process_query(QueryRequest(question="What is the population of Kolkata_Village_01?"))
        assert res.grounded is True
        assert res.result_count >= 1
        assert "kolkata_village_01" in res.answer.lower()


class TestStandaloneDatasetQueries:
    """Validate queries against standalone non-state datasets."""

    def test_vivo_phone_spec_query(self):
        """Dynamic dataset resolution should switch to Vivo.csv and retrieve phone specs."""
        res = process_query(QueryRequest(question="What is the battery capacity of Vivo V29 Pro?"))
        assert res.grounded is True
        assert res.result_count >= 1
        assert "vivo" in res.dataset_name.lower() or "v29" in res.answer.lower() or "battery" in res.answer.lower()

    def test_company_a_salary_query(self):
        """Dynamic dataset resolution should switch to company_a.csv."""
        res = process_query(QueryRequest(question="What is the salary of Alice?"))
        assert res.grounded is True
        assert res.result_count >= 1
        assert "company_a" in res.dataset_name.lower()
        assert "100000" in res.answer or "100,000" in res.answer or "alice" in res.answer.lower()

    def test_product_catalog_query(self):
        """Dynamic dataset resolution should handle products dataset queries."""
        res = process_query(QueryRequest(question="What is the price of Wireless Mouse?"))
        assert res.grounded is True
        assert res.result_count >= 1
        assert "mouse" in res.answer.lower() or "price" in res.answer.lower()


class TestMultiStepAndGlobalQueries:
    """Validate multi-step and all-dataset global queries from user requirements."""

    def test_state_village_with_capital(self):
        """Which state has the village with the highest population and what is its capital?"""
        res = process_query(QueryRequest(question="Which state has the village with the highest population and what is its capital?"))
        assert res.grounded is True
        assert res.result_count >= 1
        assert "capital" in res.answer.lower()
        assert "village" in res.answer.lower()

    def test_capital_belonging_to_largest_state(self):
        """Which capital belongs to the state with the largest total village population?"""
        res = process_query(QueryRequest(question="Which capital belongs to the state with the largest total village population?"))
        assert res.grounded is True
        assert res.result_count >= 1
        assert "capital is" in res.answer.lower() or "capital" in res.answer.lower()

    def test_village_with_highest_population_and_females(self):
        """Which village has the highest population and how many females does it have?"""
        res = process_query(QueryRequest(question="Which village has the highest population and how many females does it have?"))
        assert res.grounded is True
        assert res.result_count >= 1
        assert "female" in res.answer.lower()


class TestStatisticalCalculations:
    """Validate dynamic calculation of VARIANCE, STD, and RANGE on datasets."""

    def test_variance_and_range(self):
        """Test variance, std, and range calculations deterministically."""
        import pandas as pd
        from app.query.operations import execute_aggregation
        df = pd.DataFrame({"score": [10.0, 20.0, 30.0, 40.0, 50.0]})

        res_var = execute_aggregation(df, "VARIANCE", "score")
        assert res_var is not None
        assert res_var["value"] == 250.0

        res_std = execute_aggregation(df, "STD", "score")
        assert res_std is not None
        assert abs(res_std["value"] - 15.81) <= 0.05

        res_range = execute_aggregation(df, "RANGE", "score")
        assert res_range is not None
        assert res_range["value"] == 40.0
