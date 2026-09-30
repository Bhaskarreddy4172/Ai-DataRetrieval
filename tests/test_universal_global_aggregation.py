"""Comprehensive Automated Tests for Universal Global Aggregation,
Performance Optimization, Multi-State Operations, and Fast Routing.

Covers all 22+ requirements from Section 25 and acceptance tests from Section 27.
"""

import time
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.dataset.registry import parent_child_registry
from app.query.global_aggregation_engine import universal_global_aggregation_engine
from app.query.fast_classifier import fast_query_classifier


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient with initialized database and preloaded registry."""
    return TestClient(app)


class TestUniversalGlobalAggregationEngine:
    """Direct engine verification tests."""

    def test_total_population_all_28_states(self):
        """1. Verify total population of all 28 states matches exactly 12,543,915."""
        res = universal_global_aggregation_engine.execute("What is the total population of all states?")
        assert res["operation"] == "SUM"
        assert res["scope"] == "ALL_STATES"
        assert res["total"] == 12543915.0
        assert res["verification_status"] == "PASS"
        assert res["expected_state_count"] == 28
        assert res["resolved_state_count"] == 28
        assert len(res["missing_states"]) == 0
        assert "12,543,915" in res["answer"]

    def test_total_population_of_tg(self):
        """2. Verify total population of Telangana (TG) is 919,813."""
        res = universal_global_aggregation_engine.execute("Total population of Telangana")
        assert res["operation"] == "SUM"
        assert res["total"] == 919813.0
        assert "919,813" in res["answer"]

    def test_total_population_of_ap(self):
        """3. Verify total population of Andhra Pradesh (AP) is 376,000."""
        res = universal_global_aggregation_engine.execute("Total population of AP")
        assert res["operation"] == "SUM"
        assert res["total"] == 376000.0
        assert "376,000" in res["answer"]

    def test_total_population_of_tg_and_ap(self):
        """4. Verify total population of TG and AP is exactly 1,295,813 (919,813 + 376,000)."""
        res = universal_global_aggregation_engine.execute("Total population of TG and AP")
        assert res["operation"] == "SUM"
        assert res["scope"] == "SELECTED_STATES"
        assert res["total"] == 1295813.0
        assert "1,295,813" in res["answer"]

    def test_total_population_of_5_selected_states(self):
        """5. Verify total population of 5 selected states (TG, AP, Karnataka, Maharashtra, Tamil Nadu)."""
        res = universal_global_aggregation_engine.execute(
            "How many people are there in Telangana, Andhra Pradesh, Karnataka, Maharashtra and Tamil Nadu?"
        )
        assert res["operation"] == "SUM"
        assert res["scope"] == "SELECTED_STATES"
        assert len(res["results"]) == 5
        # TG: 919813, AP: 376000, KA: 728095, MH: 965353, TN: 684041 -> Total: 3,673,302
        assert res["total"] == 3673302.0
        assert "3,673,302" in res["answer"]

    def test_total_population_of_every_state(self):
        """6. Verify all-states breakdown returns 28 states sorted descending."""
        res = universal_global_aggregation_engine.execute("Total population of every state")
        assert res["scope"] == "ALL_STATES_BREAKDOWN"
        assert len(res["results"]) == 28
        assert res["results"][0]["state"] == "West Bengal"
        assert res["results"][0]["value"] == 972779.0

    def test_highest_population_state(self):
        """7. Verify highest population state across all registered states is West Bengal (972,779)."""
        res = universal_global_aggregation_engine.execute("Which state has the highest population?")
        assert res["operation"] == "MAX"
        assert res["scope"] == "STATE_RANKING"
        assert res["results"][0]["state"] == "West Bengal"
        assert res["results"][0]["value"] == 972779.0
        assert "West Bengal" in res["answer"]
        assert "972,779" in res["answer"]

    def test_lowest_population_state(self):
        """8. Verify lowest population state across all registered states is Goa (104,044)."""
        res = universal_global_aggregation_engine.execute("Which state has the lowest population?")
        assert res["operation"] == "MIN"
        assert res["scope"] == "STATE_RANKING"
        assert res["results"][0]["state"] == "Goa"
        assert res["results"][0]["value"] == 104044.0
        assert "Goa" in res["answer"]

    def test_average_state_population(self):
        """9. Verify average state population is 12,543,915 / 28 = 447,996.96."""
        res = universal_global_aggregation_engine.execute("Average population of all states")
        assert res["operation"] == "AVERAGE"
        assert res["scope"] == "ALL_STATES"
        assert abs(res["average"] - 447996.96) < 1.0
        assert "447,997" in res["answer"]

    def test_top_5_states_by_population(self):
        """10. Verify Top 5 states ranking."""
        res = universal_global_aggregation_engine.execute("Top 5 states by population")
        assert len(res["results"]) == 5
        top_states = [r["state"] for r in res["results"]]
        assert top_states == ["West Bengal", "Maharashtra", "Uttar Pradesh", "Telangana", "Tamil Nadu"]

    def test_bottom_5_states_by_population(self):
        """11. Verify Bottom 5 states ranking."""
        res = universal_global_aggregation_engine.execute("Bottom 5 states by population")
        assert len(res["results"]) == 5
        bottom_states = [r["state"] for r in res["results"]]
        assert bottom_states == ["Goa", "Sikkim", "Himachal Pradesh", "Nagaland", "Mizoram"]

    def test_total_population_all_villages(self):
        """12. Verify total population of all villages is 12,543,915."""
        res = universal_global_aggregation_engine.execute("Total population of all villages")
        assert res["total"] == 12543915.0
        assert "12,543,915" in res["answer"]


class TestUniversalComparisonsAndDifferences:
    """Comparison and difference query verification."""

    def test_state_comparison_telangana_vs_andhra(self, client: TestClient):
        """13. Compare Telangana and Andhra Pradesh population."""
        resp = client.post("/chat", json={"question": "Compare Telangana and Andhra Pradesh population"})
        assert resp.status_code == 200
        data = resp.json()
        assert "919,813" in data["answer"]
        assert "376,000" in data["answer"]
        assert "543,813" in data["answer"]

    def test_how_many_more_people(self, client: TestClient):
        """14. How many more people does Telangana have than Andhra Pradesh?"""
        resp = client.post("/chat", json={"question": "How many more people does Telangana have than Andhra Pradesh?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "543,813 more people" in data["answer"]
        assert "Telangana" in data["answer"]

    def test_how_many_fewer_people(self, client: TestClient):
        """15. How many fewer people does Andhra Pradesh have than Telangana?"""
        resp = client.post("/chat", json={"question": "How many fewer people does Andhra Pradesh have than Telangana?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "543,813 fewer people" in data["answer"]
        assert "Andhra Pradesh" in data["answer"]

    def test_cross_state_village_comparison(self, client: TestClient):
        """16. Compare village 1 in Maharashtra and village 2 in Karnataka."""
        resp = client.post("/chat", json={"question": "Compare village 1 in Maharashtra and village 2 in Karnataka"})
        assert resp.status_code == 200
        data = resp.json()
        assert "Village 1 in Maharashtra" in data["answer"]
        assert "Village 2 in Karnataka" in data["answer"]
        assert "25,289" in data["answer"]

    def test_same_state_village_comparison(self, client: TestClient):
        """17. Compare village 1 and village 2 in Gujarat."""
        resp = client.post("/chat", json={"question": "Compare village 1 and village 2 in Gujarat"})
        assert resp.status_code == 200
        data = resp.json()
        assert "Gandhinagar_Village_01" in data["answer"] or "Village 1" in data["answer"]
        assert "Gandhinagar_Village_02" in data["answer"] or "Village 2" in data["answer"]


class TestGlobalQueriesWhileAnotherDatasetSelected:
    """Verify that global queries search the catalog regardless of active UI dataset."""

    def test_global_query_when_standalone_dataset_loaded(self, client: TestClient):
        """18. Global query must work even if Vivo.csv is currently loaded in dataset_loader."""
        from app.dataset.loader import dataset_loader
        from app.config import settings
        vivo_file = settings.UPLOADS_DIR / "Vivo.csv"
        if vivo_file.exists():
            dataset_loader.load_dataset(vivo_file, custom_name="Vivo.csv")

        resp = client.post("/chat", json={"question": "What is the total population of all states?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "12,543,915" in data["answer"]
        assert data["confidence"] == 1.0


class TestSection27AcceptanceSuite:
    """Verify the exact 17 queries from prompt section 27."""

    def test_q1_total_population_of_all_states(self, client: TestClient):
        resp = client.post("/chat", json={"question": "What is the total population of all states?"})
        assert "12,543,915" in resp.json()["answer"]

    def test_q2_total_population_of_telangana(self, client: TestClient):
        resp = client.post("/chat", json={"question": "Total population of Telangana"})
        assert "919,813" in resp.json()["answer"]

    def test_q3_total_population_of_tg_and_ap(self, client: TestClient):
        resp = client.post("/chat", json={"question": "Total population of TG and AP"})
        assert "1,295,813" in resp.json()["answer"]

    def test_q4_how_many_people_live_in_all_states(self, client: TestClient):
        resp = client.post("/chat", json={"question": "How many people live in all states?"})
        assert "12,543,915" in resp.json()["answer"]

    def test_q5_which_state_has_the_highest_population(self, client: TestClient):
        resp = client.post("/chat", json={"question": "Which state has the highest population?"})
        assert "West Bengal" in resp.json()["answer"]

    def test_q6_which_state_has_the_lowest_population(self, client: TestClient):
        resp = client.post("/chat", json={"question": "Which state has the lowest population?"})
        assert "Goa" in resp.json()["answer"]

    def test_q7_average_population_of_all_states(self, client: TestClient):
        resp = client.post("/chat", json={"question": "Average population of all states"})
        assert "447,997" in resp.json()["answer"]

    def test_q8_top_5_states_by_population(self, client: TestClient):
        resp = client.post("/chat", json={"question": "Top 5 states by population"})
        assert "West Bengal" in resp.json()["answer"]
        assert "Maharashtra" in resp.json()["answer"]

    def test_q9_compare_telangana_and_andhra_pradesh(self, client: TestClient):
        resp = client.post("/chat", json={"question": "Compare Telangana and Andhra Pradesh population"})
        assert "919,813" in resp.json()["answer"]
        assert "376,000" in resp.json()["answer"]

    def test_q10_how_many_more_people_tg_than_ap(self, client: TestClient):
        resp = client.post("/chat", json={"question": "How many more people does Telangana have than Andhra Pradesh?"})
        assert "543,813 more people" in resp.json()["answer"]

    def test_q11_how_many_fewer_people_ap_than_tg(self, client: TestClient):
        resp = client.post("/chat", json={"question": "How many fewer people does Andhra Pradesh have than Telangana?"})
        assert "543,813 fewer people" in resp.json()["answer"]

    def test_q12_compare_village_in_tg_with_village_in_ap(self, client: TestClient):
        resp = client.post("/chat", json={"question": "Compare village 1 in Telangana with village 2 in Andhra Pradesh"})
        assert "Village 1 in Telangana" in resp.json()["answer"] or "Hyderabad_Village_01" in resp.json()["answer"]
        assert "Village 2 in Andhra Pradesh" in resp.json()["answer"] or "Amaravati_Village_02" in resp.json()["answer"]

    def test_q13_which_village_has_more_population_followup(self, client: TestClient):
        session_id = f"test-comp-{int(time.time())}"
        client.post("/chat", json={
            "question": "Compare village 1 in Maharashtra and village 2 in Karnataka",
            "session_id": session_id
        })
        resp = client.post("/chat", json={
            "question": "Which village has more population?",
            "session_id": session_id
        })
        assert resp.status_code == 200
        assert "Karnataka" in resp.json()["answer"]

    def test_q14_how_many_more_people_are_there_followup(self, client: TestClient):
        session_id = f"test-more-{int(time.time())}"
        client.post("/chat", json={
            "question": "Compare village 1 in Maharashtra and village 2 in Karnataka",
            "session_id": session_id
        })
        resp = client.post("/chat", json={
            "question": "How many more people are there?",
            "session_id": session_id
        })
        assert resp.status_code == 200
        assert "25,289" in resp.json()["answer"]

    def test_q15_what_is_the_population_difference_followup(self, client: TestClient):
        session_id = f"test-diff-{int(time.time())}"
        client.post("/chat", json={
            "question": "Compare village 1 in Maharashtra and village 2 in Karnataka",
            "session_id": session_id
        })
        resp = client.post("/chat", json={
            "question": "What is the population difference?",
            "session_id": session_id
        })
        assert resp.status_code == 200
        assert "25,289" in resp.json()["answer"]

    def test_q16_total_population_of_these_5_states(self, client: TestClient):
        session_id = f"test-5states-{int(time.time())}"
        # Seed session with 5 states
        client.post("/chat", json={
            "question": "Top 5 states by population",
            "session_id": session_id
        })
        resp = client.post("/chat", json={
            "question": "Total population of these 5 states",
            "session_id": session_id
        })
        assert resp.status_code == 200
        # West Bengal (972779) + Maharashtra (965353) + UP (935047) + TG (919813) + TN (731621) = 4,524,613
        assert "4,524,613" in resp.json()["answer"]

    def test_q17_total_population_of_all_villages(self, client: TestClient):
        resp = client.post("/chat", json={"question": "Total population of all villages"})
        assert "12,543,915" in resp.json()["answer"]

