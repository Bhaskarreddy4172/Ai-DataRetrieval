"""Comprehensive Universal Test Suite: 28 States + Global Aggregations + Filter-Aware Queries.

Validates Section 1-10 requirements:
- Startup discovery & health report verification for all 28 states
- Universal operations for EVERY state:
  * Capital lookup
  * Village count (always 10)
  * Total population
  * Average village population
  * Median village population
  * Population range
  * Highest population village (scoped to state)
  * Lowest population village (scoped to state)
  * Top-N villages (scoped to state)
  * Column-to-column comparison filter (scoped to state)
  * Numeric threshold filter (scoped to state)
  * Village ranking (scoped to state)
- Global 28-state queries:
  * Total population across all 28 states
  * Total village count (280)
  * State with highest population
  * State with lowest population
  * State with highest literacy rate (AVG, not SUM)
  * Top 5 states by population
  * Villages per state breakdown (all 28 states)
  * Average population by state breakdown (all 28 states)
  * Top 10 villages across all states
  * Extreme villages across all states
"""

import unittest
from typing import Dict, Any, List

from app.dataset.registry import parent_child_registry
from app.dataset.multi_child_executor import multi_child_executor
from app.query.global_aggregation_engine import universal_global_aggregation_engine
from app.query.fast_classifier import fast_query_classifier


ALL_28_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal"
]


class TestUniversalAll28StatesEngine(unittest.TestCase):
    """Test suite ensuring universal operation across all 28 Indian states without hardcoding."""

    @classmethod
    def setUpClass(cls):
        # Trigger initialization and health check
        parent_child_registry._initialize_paths()
        multi_child_executor.get_combined_dataframe()

    def test_01_all_28_states_discovered_and_ready(self):
        """Verify all 28 states are registered and loadable."""
        available_states = parent_child_registry.get_available_states()
        self.assertGreaterEqual(len(available_states), 28)
        for state in ALL_28_STATES:
            self.assertIn(state, available_states, f"State '{state}' missing from available states")
            resolved = parent_child_registry.resolve_child_dataset(state)
            self.assertIsNotNone(resolved, f"Failed to resolve child dataset for '{state}'")
            st_name, child_path = resolved
            df = parent_child_registry.load_child_dataframe(child_path)
            self.assertIsNotNone(df, f"Failed to load dataframe for '{state}'")
            self.assertEqual(len(df), 10, f"State '{state}' should have 10 rows, got {len(df)}")
            for col in ["Village", "Population", "No_of_Males", "No_of_Females", "Literacy_Rate_Percent"]:
                self.assertIn(col, df.columns, f"Normalized column '{col}' missing from '{state}' dataset")

    def test_02_all_28_states_individual_operations(self):
        """Verify each of the 28 states supports lookup, aggregation, extremes, ranking, and filtering."""
        for state in ALL_28_STATES:
            with self.subTest(state=state):
                # 1. Capital lookup
                q_cap = f"What is the capital of {state}?"
                res_cap = universal_global_aggregation_engine.execute(q_cap)
                self.assertEqual(res_cap.get("scope"), "STATE_CAPITAL")
                self.assertTrue(len(res_cap.get("answer", "")) > 0)

                # 2. Village count
                q_cnt = f"How many villages in {state}?"
                res_cnt = universal_global_aggregation_engine.execute(q_cnt)
                self.assertEqual(res_cnt.get("value"), 10.0)
                self.assertIn("10 villages", res_cnt.get("answer", ""))

                # 3. Total population
                q_sum = f"Total population of {state}"
                res_sum = universal_global_aggregation_engine.execute(q_sum)
                self.assertGreater(res_sum.get("value", 0), 0)

                # 4. Average population
                q_avg = f"Average population in {state}"
                res_avg = universal_global_aggregation_engine.execute(q_avg)
                self.assertGreater(res_avg.get("value", 0), 0)

                # 5. Median population
                q_med = f"Median population of {state}"
                res_med = universal_global_aggregation_engine.execute(q_med)
                self.assertGreater(res_med.get("value", 0), 0)

                # 6. Highest population village (Scoped to state!)
                q_max = f"Which village has the highest population in {state}?"
                res_max = universal_global_aggregation_engine.execute(q_max)
                self.assertEqual(res_max.get("operation"), "MAX")
                self.assertEqual(res_max.get("scope"), "FILTERED")
                self.assertEqual(res_max.get("results")[0]["state"].lower(), state.lower())
                self.assertIn(state, res_max.get("answer"))

                # 7. Lowest population village (Scoped to state!)
                q_min = f"Which village has the lowest population in {state}?"
                res_min = universal_global_aggregation_engine.execute(q_min)
                self.assertEqual(res_min.get("operation"), "MIN")
                self.assertEqual(res_min.get("scope"), "FILTERED")
                self.assertEqual(res_min.get("results")[0]["state"].lower(), state.lower())
                self.assertIn(state, res_min.get("answer"))

                # 8. Top 3 villages (Scoped to state!)
                q_top = f"Top 3 villages by population in {state}"
                res_top = universal_global_aggregation_engine.execute(q_top)
                self.assertEqual(res_top.get("operation"), "TOP_N")
                self.assertEqual(res_top.get("scope"), "FILTERED")
                self.assertEqual(len(res_top.get("results")), 3)
                for r in res_top.get("results"):
                    self.assertEqual(r["state"].lower(), state.lower())

                # 9. Column-to-column comparison filter
                q_comp = f"Which village has more males than females in {state}?"
                res_comp = universal_global_aggregation_engine.execute(q_comp)
                self.assertEqual(res_comp.get("operation"), "FILTER")
                self.assertEqual(res_comp.get("scope"), "FILTERED")
                for r in res_comp.get("results"):
                    self.assertGreater(r.get("No_of_Males", 0), r.get("No_of_Females", 0))

                # 10. Village ranking within state
                q_rank = f"Rank villages by population in {state}"
                res_rank = universal_global_aggregation_engine.execute(q_rank)
                self.assertEqual(res_rank.get("operation"), "RANKING")
                self.assertEqual(res_rank.get("scope"), "FILTERED")
                self.assertEqual(len(res_rank.get("results")), 10)
                # Verify descending order
                vals = [r["value"] for r in res_rank.get("results")]
                self.assertEqual(vals, sorted(vals, reverse=True))

    def test_03_global_total_population_across_all_28_states(self):
        """Verify global sum calculates across all 28 states and 280 villages."""
        res = universal_global_aggregation_engine.execute("Total population across all states")
        self.assertEqual(res.get("operation"), "SUM")
        self.assertEqual(res.get("scope"), "ALL_STATES")
        self.assertEqual(res.get("expected_state_count"), 28)
        self.assertEqual(res.get("resolved_state_count"), 28)
        self.assertEqual(res.get("verification_status"), "PASS")
        self.assertGreater(res.get("total"), 10_000_000)
        self.assertIn("across all 28 registered states (280 villages)", res.get("answer"))

    def test_04_global_village_count(self):
        """Verify total villages across all states is 280 without binder error."""
        res = universal_global_aggregation_engine.execute("Total villages across all states")
        self.assertEqual(res.get("value"), 280.0)
        self.assertIn("280", res.get("answer"))

    def test_05_state_extremes_population(self):
        """Verify state with highest and lowest population."""
        res_high = universal_global_aggregation_engine.execute("Which state has the highest population?")
        self.assertEqual(res_high.get("scope"), "STATE_RANKING")
        self.assertEqual(res_high.get("operation"), "MAX")
        self.assertIn("West Bengal", res_high.get("answer"))

        res_low = universal_global_aggregation_engine.execute("Which state has the lowest population?")
        self.assertEqual(res_low.get("scope"), "STATE_RANKING")
        self.assertEqual(res_low.get("operation"), "MIN")
        self.assertIn("Goa", res_low.get("answer"))

    def test_06_state_highest_literacy_rate_uses_average(self):
        """Verify state with highest literacy rate calculates average, not sum."""
        res = universal_global_aggregation_engine.execute("Which state has the highest literacy rate?")
        self.assertEqual(res_high := res.get("operation"), "MAX")
        self.assertEqual(res.get("scope"), "STATE_RANKING")
        # Value must be an average percentage between 70% and 100%, NEVER 800%+
        top_val = res.get("results")[0]["value"]
        self.assertGreaterEqual(top_val, 70.0)
        self.assertLessEqual(top_val, 100.0)
        self.assertIn("average Literacy Rate Percent", res.get("answer"))

    def test_07_all_states_breakdown_villages_per_state(self):
        """Verify breakdown of villages per state lists all 28 states."""
        res = universal_global_aggregation_engine.execute("Villages per state")
        self.assertEqual(res.get("scope"), "ALL_STATES_BREAKDOWN")
        self.assertEqual(res.get("result_count"), 28)
        self.assertIn("Village count breakdown for all 28 registered states", res.get("answer"))

    def test_08_all_states_breakdown_population_by_state(self):
        """Verify ranking/breakdown of all states by population."""
        res = universal_global_aggregation_engine.execute("Rank all states by population")
        self.assertEqual(res.get("scope"), "ALL_STATES_BREAKDOWN")
        self.assertEqual(res.get("result_count"), 28)
        self.assertIn("Total Population breakdown for all 28 registered states", res.get("answer"))

    def test_09_global_top_10_villages(self):
        """Verify top 10 villages across all 280 villages."""
        res = universal_global_aggregation_engine.execute("Top 10 villages across all states")
        self.assertEqual(res.get("scope"), "ALL_VILLAGES")
        self.assertEqual(len(res.get("results")), 10)
        self.assertIn("Top 10 villages by Population across all states", res.get("answer"))

    def test_10_filtered_numeric_range_query(self):
        """Verify range filtering within a state."""
        res = universal_global_aggregation_engine.execute("Villages with population between 50000 and 100000 in Telangana")
        self.assertEqual(res.get("scope"), "FILTERED")
        self.assertEqual(res.get("operation"), "FILTER")
        for r in res.get("results"):
            pop = r.get("Population")
            self.assertGreaterEqual(pop, 50000)
            self.assertLessEqual(pop, 100000)
            self.assertEqual(r.get("state"), "Telangana")

    def test_11_multi_turn_conversation_flow(self):
        """Verify 9-turn ChatGPT-style conversation sequence with state overrides, pivots, and follow-ups."""
        from app.conversation.context import conversation_manager
        from app.query.schema import StructuredQuery

        session_id = "test_eval_convo_9turns"
        conversation_manager.clear_session(session_id)

        # Turn 1: Gujarat Highest Area
        q1 = "Which village has the highest area in Gujarat?"
        cls1 = fast_query_classifier.classify(q1, session_id=session_id)
        res1 = universal_global_aggregation_engine.execute(q1, session_id=session_id)
        self.assertEqual(cls1.intent, "MAX")
        self.assertEqual(cls1.scope, "FILTERED")
        self.assertEqual(cls1.metric, "Area_Sq_Km")
        self.assertIn("Gujarat", res1.get("answer"))
        self.assertIn("Gandhinagar_Village_10", res1.get("answer"))
        conversation_manager.add_turn(
            session_id=session_id, question=q1, query=StructuredQuery(operation="MAX", limit=1),
            result_count=len(res1.get("results", [])), results=res1.get("results", []),
            answer=res1.get("answer", ""), metric=cls1.metric, intent=cls1.intent,
            return_entity=cls1.return_entity, scope=cls1.scope
        )

        # Turn 2: Follow-up lowest area in Gujarat
        q2 = "Which has less area?"
        cls2 = fast_query_classifier.classify(q2, session_id=session_id)
        res2 = universal_global_aggregation_engine.execute(q2, session_id=session_id)
        self.assertEqual(cls2.intent, "MIN")
        self.assertEqual(cls2.scope, "FILTERED")
        self.assertEqual(cls2.metric, "Area_Sq_Km")
        self.assertEqual(cls2.entities, ["Gujarat"])
        self.assertIn("Gandhinagar_Village_05", res2.get("answer"))
        self.assertIn("Gujarat", res2.get("answer"))
        conversation_manager.add_turn(
            session_id=session_id, question=q2, query=StructuredQuery(operation="MIN", limit=1),
            result_count=len(res2.get("results", [])), results=res2.get("results", []),
            answer=res2.get("answer", ""), metric=cls2.metric, intent=cls2.intent,
            return_entity=cls2.return_entity, scope=cls2.scope
        )

        # Turn 3: Explicit State Override to Karnataka
        q3 = "Which village has highest area in Karnataka?"
        cls3 = fast_query_classifier.classify(q3, session_id=session_id)
        res3 = universal_global_aggregation_engine.execute(q3, session_id=session_id)
        self.assertEqual(cls3.intent, "MAX")
        self.assertEqual(cls3.scope, "FILTERED")
        self.assertEqual(cls3.metric, "Area_Sq_Km")
        self.assertEqual(cls3.entities, ["Karnataka"])
        self.assertIn("Karnataka", res3.get("answer"))
        self.assertIn("Bengaluru_Village_09", res3.get("answer"))
        conversation_manager.add_turn(
            session_id=session_id, question=q3, query=StructuredQuery(operation="MAX", limit=1),
            result_count=len(res3.get("results", [])), results=res3.get("results", []),
            answer=res3.get("answer", ""), metric=cls3.metric, intent=cls3.intent,
            return_entity=cls3.return_entity, scope=cls3.scope
        )

        # Turn 4: Follow-up highest population in Karnataka
        q4 = "Which has more population?"
        cls4 = fast_query_classifier.classify(q4, session_id=session_id)
        res4 = universal_global_aggregation_engine.execute(q4, session_id=session_id)
        self.assertEqual(cls4.intent, "MAX")
        self.assertEqual(cls4.scope, "FILTERED")
        self.assertEqual(cls4.metric, "Population")
        self.assertEqual(cls4.entities, ["Karnataka"])
        self.assertIn("Karnataka", res4.get("answer"))
        self.assertIn("Bengaluru_Village_10", res4.get("answer"))
        conversation_manager.add_turn(
            session_id=session_id, question=q4, query=StructuredQuery(operation="MAX", limit=1),
            result_count=len(res4.get("results", [])), results=res4.get("results", []),
            answer=res4.get("answer", ""), metric=cls4.metric, intent=cls4.intent,
            return_entity=cls4.return_entity, scope=cls4.scope
        )

        # Turn 5: Pronoun capital lookup
        q5 = "What is its capital?"
        cls5 = fast_query_classifier.classify(q5, session_id=session_id)
        res5 = universal_global_aggregation_engine.execute(q5, session_id=session_id)
        self.assertEqual(cls5.intent, "CAPITAL_LOOKUP")
        self.assertEqual(cls5.scope, "STATE_CAPITAL")
        self.assertEqual(cls5.entities, ["Karnataka"])
        self.assertIn("Bengaluru", res5.get("answer"))
        conversation_manager.add_turn(
            session_id=session_id, question=q5, query=StructuredQuery(operation="CAPITAL_LOOKUP", limit=1),
            result_count=len(res5.get("results", [])), results=res5.get("results", []),
            answer=res5.get("answer", ""), metric=cls5.metric, intent=cls5.intent,
            return_entity=cls5.return_entity, scope=cls5.scope
        )

        # Turn 6: Pronoun village count
        q6 = "How many villages does it have?"
        cls6 = fast_query_classifier.classify(q6, session_id=session_id)
        res6 = universal_global_aggregation_engine.execute(q6, session_id=session_id)
        self.assertEqual(cls6.intent, "COUNT")
        self.assertEqual(cls6.metric, "VILLAGE_COUNT")
        self.assertEqual(cls6.entities, ["Karnataka"])
        self.assertIn("10 villages", res6.get("answer"))
        conversation_manager.add_turn(
            session_id=session_id, question=q6, query=StructuredQuery(operation="COUNT", limit=1),
            result_count=len(res6.get("results", [])), results=res6.get("results", []),
            answer=res6.get("answer", ""), metric=cls6.metric, intent=cls6.intent,
            return_entity=cls6.return_entity, scope=cls6.scope
        )

        # Turn 7: Elliptical pivot to Bihar preserving COUNT
        q7 = "And for Bihar?"
        cls7 = fast_query_classifier.classify(q7, session_id=session_id)
        res7 = universal_global_aggregation_engine.execute(q7, session_id=session_id)
        self.assertEqual(cls7.intent, "COUNT")
        self.assertEqual(cls7.metric, "VILLAGE_COUNT")
        self.assertEqual(cls7.entities, ["Bihar"])
        self.assertIn("10 villages", res7.get("answer"))
        conversation_manager.add_turn(
            session_id=session_id, question=q7, query=StructuredQuery(operation="COUNT", limit=1),
            result_count=len(res7.get("results", [])), results=res7.get("results", []),
            answer=res7.get("answer", ""), metric=cls7.metric, intent=cls7.intent,
            return_entity=cls7.return_entity, scope=cls7.scope
        )

        # Turn 8: Explicit global state ranking MAX
        q8 = "which state has more population?"
        cls8 = fast_query_classifier.classify(q8, session_id=session_id)
        res8 = universal_global_aggregation_engine.execute(q8, session_id=session_id)
        self.assertEqual(cls8.intent, "MAX")
        self.assertEqual(cls8.scope, "STATE_RANKING")
        self.assertIn("West Bengal", res8.get("answer"))

        # Turn 9: Explicit global state ranking MIN
        q9 = "which state has less population?"
        cls9 = fast_query_classifier.classify(q9, session_id=session_id)
        res9 = universal_global_aggregation_engine.execute(q9, session_id=session_id)
        self.assertEqual(cls9.intent, "MIN")
        self.assertEqual(cls9.scope, "STATE_RANKING")
        self.assertIn("Goa", res9.get("answer"))

    def test_12_all_columns_support_generic_operations(self):
        """Verify all columns (Area, Households, Males, Females, Literacy, Population) support operations."""
        cols = ["Area_Sq_Km", "Households", "No_of_Males", "No_of_Females", "Literacy_Rate_Percent", "Population"]
        ops = ["SUM", "AVERAGE", "MEDIAN", "MIN", "MAX", "RANGE"]

        for col in cols:
            for op in ops:
                with self.subTest(column=col, operation=op):
                    q = f"{op} of {col} in Telangana"
                    res = universal_global_aggregation_engine.execute(q)
                    self.assertTrue(len(res.get("answer", "")) > 0)
                    self.assertIn("Telangana", res.get("answer"))

    def test_13_consistency_validator_enforces_state_filters(self):
        """Verify AnswerConsistencyValidator catches state filter violations."""
        from app.query.validator import answer_consistency_validator

        # Valid result matching query state
        valid_res = [{"village": "Village_1", "state": "Gujarat", "Area_Sq_Km": 40.72}]
        is_ok, reason = answer_consistency_validator.validate_answer(
            question="Which village has highest area in Gujarat?",
            operation="MAX", scope="FILTERED",
            answer="The village is Village_1 in Gujarat.",
            results=valid_res
        )
        self.assertTrue(is_ok)
        self.assertIsNone(reason)

        # Invalid result from another state (e.g. Rajasthan returned for Gujarat query)
        invalid_res = [{"village": "Jaipur_Village_1", "state": "Rajasthan", "Area_Sq_Km": 50.0}]
        is_bad, reason = answer_consistency_validator.validate_answer(
            question="Which village has highest area in Gujarat?",
            operation="MAX", scope="FILTERED",
            answer="The village is Jaipur_Village_1 in Rajasthan.",
            results=invalid_res
        )
        self.assertFalse(is_bad)
        self.assertIn("State constraint violation", reason)


if __name__ == "__main__":
    unittest.main()
