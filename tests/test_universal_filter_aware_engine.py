"""Comprehensive Verification Test Suite for Universal Filter-Aware Engine.

Covers:
1. Filtered MAX/MIN queries (AP, Telangana, Karnataka, West Bengal, Goa)
2. Global MAX/MIN queries (all 280 villages)
3. Multi-hop capital filters (whose capital is Hyderabad -> Telangana)
4. State code resolution (in ka -> Karnataka)
5. Top-N filtered village queries
6. Multi-turn session context independence (no bleed-over across turns)
7. AnswerConsistencyValidator rejection of artificially corrupted results
8. FastClassificationResult plan dictionary format compliance
"""

import pytest
from app.query.fast_classifier import fast_query_classifier
from app.query.global_aggregation_engine import universal_global_aggregation_engine
from app.query.validator import answer_consistency_validator
from app.conversation.context import conversation_manager


class TestUniversalFilterAwareEngine:
    """Rigorous end-to-end tests for filter-aware dataset retrieval."""

    def test_filtered_ap_max_village(self):
        """'which village has more population in ap' must return Amaravati_Village_06 (60,805)."""
        question = "which village has more population in ap"
        cls_res = fast_query_classifier.classify(question)
        assert cls_res.intent == "MAX"
        assert cls_res.scope == "FILTERED"
        assert "Andhra Pradesh" in cls_res.entities

        res = universal_global_aggregation_engine.execute(question)
        assert res.get("verification_status") == "PASS"
        assert len(res.get("results", [])) == 1
        record = res["results"][0]
        assert record.get("village") == "Amaravati_Village_06"
        assert record.get("state") == "Andhra Pradesh"
        assert record.get("value") == 60805.0

    def test_filtered_ap_min_village(self):
        """'which village has less population in ap' must return Amaravati_Village_04 (19,825)."""
        question = "which village has less population in ap"
        cls_res = fast_query_classifier.classify(question)
        assert cls_res.intent == "MIN"
        assert cls_res.scope == "FILTERED"
        assert "Andhra Pradesh" in cls_res.entities

        res = universal_global_aggregation_engine.execute(question)
        assert res.get("verification_status") == "PASS"
        record = res["results"][0]
        assert record.get("village") == "Amaravati_Village_04"
        assert record.get("state") == "Andhra Pradesh"
        assert record.get("value") == 19825.0

    def test_filtered_telangana_max_village(self):
        """'which village has more population in telangana' must return Hyderabad_Village_01 (117,305)."""
        question = "which village has more population in telangana"
        cls_res = fast_query_classifier.classify(question)
        assert cls_res.intent == "MAX"
        assert cls_res.scope == "FILTERED"
        assert "Telangana" in cls_res.entities

        res = universal_global_aggregation_engine.execute(question)
        assert res.get("verification_status") == "PASS"
        record = res["results"][0]
        assert record.get("village") == "Hyderabad_Village_01"
        assert record.get("state") == "Telangana"
        assert record.get("value") == 117305.0

    def test_filtered_telangana_min_village(self):
        """'which village has less population in telangana' must return Hyderabad_Village_10 (43,527)."""
        question = "which village has less population in telangana"
        cls_res = fast_query_classifier.classify(question)
        assert cls_res.intent == "MIN"
        assert cls_res.scope == "FILTERED"
        assert "Telangana" in cls_res.entities

        res = universal_global_aggregation_engine.execute(question)
        assert res.get("verification_status") == "PASS"
        record = res["results"][0]
        assert record.get("village") == "Hyderabad_Village_10"
        assert record.get("state") == "Telangana"
        assert record.get("value") == 43527.0

    def test_filtered_karnataka_code_village(self):
        """'which village has more population in ka' must resolve 'ka' to Karnataka -> Bengaluru_Village_10 (106,986)."""
        question = "which village has more population in ka"
        cls_res = fast_query_classifier.classify(question)
        assert cls_res.intent == "MAX"
        assert cls_res.scope == "FILTERED"
        assert "Karnataka" in cls_res.entities

        res = universal_global_aggregation_engine.execute(question)
        assert res.get("verification_status") == "PASS"
        record = res["results"][0]
        assert record.get("village") == "Bengaluru_Village_10"
        assert record.get("state") == "Karnataka"
        assert record.get("value") == 106986.0

    def test_multi_hop_capital_filter(self):
        """'which village has highest population in the state whose capital is Hyderabad' -> Telangana -> Hyderabad_Village_01."""
        question = "which village has highest population in the state whose capital is Hyderabad"
        cls_res = fast_query_classifier.classify(question)
        assert cls_res.intent == "MAX"
        assert cls_res.scope == "FILTERED"
        assert "Telangana" in cls_res.entities

        res = universal_global_aggregation_engine.execute(question)
        assert res.get("verification_status") == "PASS"
        record = res["results"][0]
        assert record.get("village") == "Hyderabad_Village_01"
        assert record.get("state") == "Telangana"
        assert record.get("value") == 117305.0

    def test_global_village_extremes(self):
        """Global village extreme queries (no state filter) must evaluate across all 280 villages."""
        max_q = "which village has highest population across all states"
        cls_max = fast_query_classifier.classify(max_q)
        assert cls_max.scope == "ALL_VILLAGES"
        assert len(cls_max.entities) == 0

        res_max = universal_global_aggregation_engine.execute(max_q)
        assert res_max.get("verification_status") == "PASS"
        rec_max = res_max["results"][0]
        assert rec_max.get("Village") == "Kolkata_Village_05"
        assert rec_max.get("State") == "West Bengal"
        assert rec_max.get("Population") == 147029

        min_q = "which village has lowest population across all states"
        cls_min = fast_query_classifier.classify(min_q)
        assert cls_min.scope == "ALL_VILLAGES"
        assert len(cls_min.entities) == 0

        res_min = universal_global_aggregation_engine.execute(min_q)
        assert res_min.get("verification_status") == "PASS"
        rec_min = res_min["results"][0]
        assert rec_min.get("Village") == "Panaji_Village_08"
        assert rec_min.get("State") == "Goa"
        assert rec_min.get("Population") == 5368

    def test_top_3_villages_in_ap(self):
        """'top 3 villages in ap' must return top 3 villages exclusively from Andhra Pradesh."""
        question = "top 3 villages in ap"
        cls_res = fast_query_classifier.classify(question)
        assert cls_res.intent == "TOP_N"
        assert cls_res.scope == "FILTERED"
        assert cls_res.n_limit == 3

        res = universal_global_aggregation_engine.execute(question)
        assert res.get("verification_status") == "PASS"
        assert len(res.get("results", [])) == 3
        for r in res["results"]:
            assert r.get("state") == "Andhra Pradesh"
        assert res["results"][0]["village"] == "Amaravati_Village_06"
        assert res["results"][0]["value"] == 60805.0

    def test_multi_turn_context_isolation(self):
        """Ensure turn 1 filter does not pollute turn 2 filter or global turn 3."""
        session_id = "test_turn_isolation_session"
        conversation_manager.clear_session(session_id)

        # Turn 1: Telangana
        q1 = "which village has less population in telangana"
        res1 = universal_global_aggregation_engine.execute(q1, session_id=session_id)
        assert res1["results"][0]["state"] == "Telangana"

        # Turn 2: AP (Must NOT reuse Telangana)
        q2 = "which village has more population in ap"
        res2 = universal_global_aggregation_engine.execute(q2, session_id=session_id)
        assert res2["results"][0]["state"] == "Andhra Pradesh"
        assert res2["results"][0]["village"] == "Amaravati_Village_06"

        # Turn 3: Global village query (Must NOT inherit AP or Telangana)
        q3 = "which village has highest population"
        res3 = universal_global_aggregation_engine.execute(q3, session_id=session_id)
        assert res3["results"][0]["State"] == "West Bengal"

    def test_answer_consistency_validator_rejection(self):
        """Validator must reject artificially injected wrong state or missing entity results."""
        # 1. Reject West Bengal result for AP question
        valid, reason = answer_consistency_validator.validate_answer(
            question="which village has more population in ap",
            operation="MAX",
            scope="FILTERED",
            answer="The village is Kolkata_Village_05 in West Bengal.",
            results=[{"village": "Kolkata_Village_05", "state": "West Bengal", "value": 147029}]
        )
        assert valid is False
        assert "State constraint violation" in reason

        # 2. Reject Goa result for Telangana question
        valid, reason = answer_consistency_validator.validate_answer(
            question="which village has less population in telangana",
            operation="MIN",
            scope="FILTERED",
            answer="The village is Panaji_Village_08 in Goa.",
            results=[{"village": "Panaji_Village_08", "state": "Goa", "value": 5368}]
        )
        assert valid is False
        assert "State constraint violation" in reason

        # 3. Reject result with missing village identifier when village requested
        valid, reason = answer_consistency_validator.validate_answer(
            question="which village has more population in ap",
            operation="MAX",
            scope="FILTERED",
            answer="Total is 398,829.",
            results=[{"state": "Andhra Pradesh", "value": 398829}]
        )
        assert valid is False
        assert "Entity level violation" in reason

    def test_plan_dict_compliance(self):
        """Classification result must generate compliant query plan dictionary."""
        question = "which village has more population in ap"
        cls_res = fast_query_classifier.classify(question)
        plan = cls_res.to_plan_dict()
        assert plan["intent"] == "MAX"
        assert plan["return_entity"] == "Village"
        assert plan["scope"] == "FILTERED"
        assert plan["scope_type"] == "FILTERED"
        assert len(plan["filters"]) == 1
        assert plan["filters"][0]["column"] == "State"
        assert plan["filters"][0]["value"] == "Andhra Pradesh"
        assert plan["order"] == "DESC"
        assert plan["limit"] == 1

