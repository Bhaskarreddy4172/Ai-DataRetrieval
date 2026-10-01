"""Integration Test Suite: Database, RAG, Ingestion, and SQL Execution across 28 States."""

import pytest
from app.database.connection import db_manager
from app.database.repositories import (
    dataset_repo, entity_repo, state_village_repo, rag_doc_repo
)
from app.rag.vector_store import database_vector_store
from app.rag.hybrid_retriever import hybrid_retriever
from app.execution.sql_executor import sql_executor
from app.ingestion.importer import dataset_importer


@pytest.fixture(scope="module", autouse=True)
def setup_integration():
    """Ensure database is initialized and data is ingested before running tests."""
    db_manager.init_database()
    states = state_village_repo.get_all_states()
    if len(states) < 28:
        dataset_importer.import_all()
    if database_vector_store.count_documents() == 0:
        from app.rag.document_builder import rag_document_builder
        docs = rag_document_builder.build_all_documents()
        database_vector_store.add_documents(docs)


def test_database_health():
    health = db_manager.check_health()
    assert health.get("status") == "READY"
    assert health.get("healthy") is True
    assert health.get("backend") in ("PostgreSQL", "SQLite")


def test_all_28_states_in_database():
    states = state_village_repo.get_all_states()
    assert len(states) >= 28
    state_names = [s["state"] for s in states]
    assert "Andhra Pradesh" in state_names
    assert "Telangana" in state_names
    assert "Maharashtra" in state_names
    assert "West Bengal" in state_names
    assert "Gujarat" in state_names


def test_village_data_across_states():
    # Test multiple distinct states to verify child village records
    test_states = ["Andhra Pradesh", "Telangana", "Goa", "Punjab", "Kerala", "West Bengal"]
    for st in test_states:
        df = sql_executor.get_all_villages_df(state_filter=st)
        assert not df.empty, f"No villages found in database for state {st}"
        assert len(df) == 10, f"Expected 10 villages for state {st}, got {len(df)}"
        assert "village" in df.columns or "Village" in df.columns
        assert (df["population"] > 0).all()


def test_rag_vector_store_persistence():
    doc_count = database_vector_store.count_documents()
    assert doc_count > 0, "RAG vector store should contain indexed documents"


def test_hybrid_retriever_query():
    hits = hybrid_retriever.retrieve("which state has the highest population", k=3)
    assert len(hits) > 0
    assert "score" in hits[0]
    assert hits[0]["score"] > 0.0


def test_sql_executor_extreme_filtered():
    # AP highest village
    ap_max = sql_executor.get_village_extreme("population", "MAX", state_filter="Andhra Pradesh")
    assert ap_max is not None
    assert ap_max["village"] == "Amaravati_Village_06"
    assert ap_max["value"] == 60805.0

    # TG lowest village
    tg_min = sql_executor.get_village_extreme("population", "MIN", state_filter="Telangana")
    assert tg_min is not None
    assert tg_min["village"] == "Hyderabad_Village_10"
    assert tg_min["value"] == 43527.0


def test_sql_executor_state_aggregations():
    # Aggregation for Telangana
    tg_agg = sql_executor.get_state_aggregation("population", "SUM", state_filter="Telangana")
    assert tg_agg["aggregate_value"] == 919813.0
    assert tg_agg["record_count"] == 10

    # Aggregation for Andhra Pradesh
    ap_agg = sql_executor.get_state_aggregation("population", "SUM", state_filter="Andhra Pradesh")
    assert ap_agg["aggregate_value"] == 376000.0
    assert ap_agg["record_count"] == 10


def test_entity_alias_resolution():
    assert entity_repo.resolve_alias("tg") == "Telangana"
    assert entity_repo.resolve_alias("ap") == "Andhra Pradesh"
    assert entity_repo.resolve_alias("mh") == "Maharashtra"
    assert entity_repo.resolve_alias("wb") == "West Bengal"
    assert entity_repo.resolve_alias("gj") == "Gujarat"


def test_global_state_ranking():
    res = sql_executor.get_state_aggregation("population", "SUM", state_filter=None)
    assert res["success"] is True
    rows = res["rows"]
    assert len(rows) >= 28
    # Top state should have highest population
    top_state = rows[0]
    assert top_state["state"] in ("West Bengal", "Maharashtra", "Uttar Pradesh")
    assert top_state["aggregate_value"] > 900000.0
