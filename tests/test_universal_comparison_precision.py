"""Universal Dataset Comparison Precision, Verification & Accuracy Test Suite.

Adheres strictly to the 52 detailed specifications:
1. Absolute Source of Truth: Dataset is single source of truth; Ollama does not invent/alter numbers.
2. 13-Point Verification Suite: ENTITY, ATTRIBUTE, DATASET, ROW, TYPE, UNIT, VALUE,
   CALCULATION, DIRECTION, AGGREGATION, DUPLICATE, MISSING_VALUE, RESULT.
3. Section 14 Comparison Result Model: Standardized dictionary with directed/absolute differences,
   higher/lower entities, unit, source datasets, rows, versions, verification status/checks.
4. Section 44 Response Templates: Non-negative counts, '{higher}: {v1} | {lower}: {v2}',
   missing value explanation, division-by-zero protection.
5. Multi-Turn Follow-Ups: 'What about literacy?', 'How much more?', with explicit-entity override.
6. Metric & Unit Awareness: Auto-detection for people, males, females, households, sq km, %, currency.
7. Edge cases: Incompatible attributes, zero denominators, ties, missing values.
"""

import unittest
from pathlib import Path
import pandas as pd

from app.api.routes import process_query, QueryRequest
from app.dataset.loader import dataset_loader
from app.dataset.registry import parent_child_registry
from app.query.comparison_engine import (
    universal_comparison_engine,
    ComparisonEntity,
    ComparisonPlan,
    ComparisonResult
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_DATASET = PROJECT_ROOT / "universal_dataset_demo" / "main_dataset" / "india_states_capitals_main.csv"
SAMPLE_EMPLOYEES = PROJECT_ROOT / "data" / "sample_employees.xlsx"


class TestUniversalComparisonPrecision(unittest.TestCase):
    """Rigorous verification and precision test suite for comparison engine."""

    @classmethod
    def setUpClass(cls):
        if MAIN_DATASET.exists():
            dataset_loader.load_dataset(MAIN_DATASET)
            parent_child_registry._initialize_paths()

    def setUp(self):
        if MAIN_DATASET.exists():
            dataset_loader.load_dataset(MAIN_DATASET)
            parent_child_registry._initialize_paths()

    # -----------------------------------------------------------------
    # 1. 13-POINT VERIFICATION SUITE TESTS
    # -----------------------------------------------------------------

    def test_01_thirteen_point_verification_pass(self):
        """Verify that a valid comparison passes all 13 verification checks."""
        q = "Compare village 1 and village 2 in TG"
        res = process_query(QueryRequest(question=q, session_id="prec_verify_1"))

        # Section 14 result model in aggregation
        agg = res.aggregation
        self.assertIsNotNone(agg)
        self.assertEqual(agg.get("verification_status"), "VERIFIED")

        checks = agg.get("verification_checks", {})
        expected_checks = [
            "ENTITY_CHECK", "ATTRIBUTE_CHECK", "DATASET_CHECK", "ROW_CHECK",
            "TYPE_CHECK", "UNIT_CHECK", "VALUE_CHECK", "CALCULATION_CHECK",
            "DIRECTION_CHECK", "AGGREGATION_CHECK", "DUPLICATE_CHECK",
            "MISSING_VALUE_CHECK", "RESULT_CHECK"
        ]
        for c in expected_checks:
            self.assertIn(c, checks, f"Check {c} must be present")
            self.assertTrue(checks[c], f"Check {c} must pass")

    def test_02_secondary_calculation_check(self):
        """Verify independent secondary calculation matches reported values."""
        plan = ComparisonPlan(
            entities=[
                ComparisonEntity(name="Hyderabad_Village_01", entity_type="VILLAGE"),
                ComparisonEntity(name="Hyderabad_Village_02", entity_type="VILLAGE")
            ],
            attribute="Population",
            comparison_type="ABSOLUTE_DIFFERENCE",
            scope="SAME_STATE_VILLAGE"
        )
        # Manually test verify_comparison with known values
        val1, val2 = 117305.0, 79281.0
        abs_diff = 38024.0
        dir_diff = 38024.0
        ok, checks = universal_comparison_engine.verify_comparison(
            plan=plan,
            left_ent=plan.entities[0],
            right_ent=plan.entities[1],
            val1=val1,
            val2=val2,
            abs_diff=abs_diff,
            directed_diff=dir_diff,
            higher_ent="Hyderabad_Village_01",
            lower_ent="Hyderabad_Village_02",
            is_equal=False,
            unit="people",
            sources=["tg_hyderabad_villages.csv"],
            source_rows=[1, 2]
        )
        self.assertTrue(ok)
        self.assertTrue(checks["CALCULATION_CHECK"])
        self.assertTrue(checks["DIRECTION_CHECK"])

        # Test calculation failure detection: mismatched abs_diff
        bad_ok, bad_checks = universal_comparison_engine.verify_comparison(
            plan=plan,
            left_ent=plan.entities[0],
            right_ent=plan.entities[1],
            val1=val1,
            val2=val2,
            abs_diff=99999.0,  # Wrong calculation
            directed_diff=dir_diff,
            higher_ent="Hyderabad_Village_01",
            lower_ent="Hyderabad_Village_02",
            is_equal=False,
            unit="people",
            sources=["tg_hyderabad_villages.csv"],
            source_rows=[1, 2]
        )
        self.assertFalse(bad_ok)
        self.assertFalse(bad_checks["CALCULATION_CHECK"])

    # -----------------------------------------------------------------
    # 2. SECTION 14 RESULT MODEL CONFORMANCE
    # -----------------------------------------------------------------

    def test_03_section_14_result_model_conformance(self):
        """Verify the full dictionary structure required by Section 14."""
        q = "Compare village 1 and village 2 in Telangana"
        res = process_query(QueryRequest(question=q, session_id="prec_sec14"))
        agg = res.aggregation
        self.assertIsNotNone(agg)

        required_keys = [
            "left_entity", "right_entity", "left_value", "right_value",
            "attribute", "unit", "directed_difference", "absolute_difference",
            "higher_entity", "lower_entity", "comparison_operator",
            "percentage_difference", "percentage_change", "ratio",
            "source_datasets", "source_rows", "dataset_versions",
            "verification_status", "verification_checks", "answer"
        ]
        for key in required_keys:
            self.assertIn(key, agg, f"Section 14 requires key '{key}'")

        # Check values
        self.assertEqual(agg["left_value"], 117305)
        self.assertEqual(agg["right_value"], 79281)
        self.assertEqual(agg["absolute_difference"], 38024)
        self.assertEqual(agg["directed_difference"], 38024)
        self.assertEqual(agg["unit"], "people")
        self.assertEqual(agg["verification_status"], "VERIFIED")

    # -----------------------------------------------------------------
    # 3. SECTION 44 TEMPLATES & DIRECTION CONFORMANCE
    # -----------------------------------------------------------------

    def test_04_section_44_format_more(self):
        """Format: '{higher} has {diff} more {unit} than {lower}. {higher}: {v1} | {lower}: {v2}.'"""
        q = "How many people are more in village 1 than village 2 in TG?"
        res = process_query(QueryRequest(question=q, session_id="prec_sec44_more"))
        ans = res.answer

        self.assertIn("more people than", ans.lower())
        self.assertIn("38,024", ans)
        self.assertIn("117,305", ans)
        self.assertIn("79,281", ans)
        # Must have separator '|'
        self.assertIn("|", ans)
        # Must NEVER have negative signs
        self.assertNotIn("-38024", ans)
        self.assertNotIn("-38,024", ans)

    def test_05_section_44_format_fewer(self):
        """Format: '{lower} has {diff} fewer {unit} than {higher}. {lower}: {v1} | {higher}: {v2}.'"""
        q = "How much population is more in village 2 compared to village 1 in TG?"
        res = process_query(QueryRequest(question=q, session_id="prec_sec44_fewer"))
        ans = res.answer

        self.assertIn("fewer people than", ans.lower())
        self.assertIn("38,024", ans)
        self.assertIn("79,281", ans)
        self.assertIn("117,305", ans)
        self.assertIn("|", ans)
        self.assertNotIn("-38024", ans)
        self.assertNotIn("-38,024", ans)

    def test_06_equal_values_formatting(self):
        """Format: 'Both {entities} have the same {attribute}: {val}.'"""
        # Telangana state vs Hyderabad capital aggregation (both have 919,813 population)
        q = "Compare population of Telangana and Hyderabad"
        res = process_query(QueryRequest(question=q, session_id="prec_equal"))
        ans = res.answer

        self.assertIn("same population", ans.lower())
        self.assertIn("919,813", ans)

    # -----------------------------------------------------------------
    # 4. CROSS-STATE & SAME-STATE VILLAGES
    # -----------------------------------------------------------------

    def test_07_cross_state_village_precision(self):
        """TG Village 1 (117,305) vs AP Village 2 (42,844). Difference = 74,461."""
        q = "Compare village 1 in TG and village 2 in AP"
        res = process_query(QueryRequest(question=q, session_id="prec_cross_1"))
        agg = res.aggregation
        self.assertIsNotNone(agg)

        self.assertEqual(agg["left_value"], 117305)
        self.assertEqual(agg["right_value"], 42844)
        self.assertEqual(agg["absolute_difference"], 74461)
        self.assertEqual(agg["directed_difference"], 74461)
        self.assertIn("Telangana", res.answer)
        self.assertIn("Andhra Pradesh", res.answer)
        self.assertIn("74,461", res.answer)

    def test_08_cross_state_village_reverse(self):
        """AP Village 2 (42,844) vs TG Village 1 (117,305). Diff = 74,461 fewer."""
        q = "Compare village 2 in AP and village 1 in TG"
        res = process_query(QueryRequest(question=q, session_id="prec_cross_2"))
        agg = res.aggregation
        self.assertIsNotNone(agg)

        self.assertEqual(agg["left_value"], 42844)
        self.assertEqual(agg["right_value"], 117305)
        self.assertEqual(agg["absolute_difference"], 74461)
        self.assertEqual(agg["directed_difference"], -74461)
        self.assertIn("fewer", res.answer.lower())
        self.assertNotIn("-74461", res.answer)

    # -----------------------------------------------------------------
    # 5. STATE VS STATE & CAPITAL VS CAPITAL
    # -----------------------------------------------------------------

    def test_09_state_vs_state_population_and_ratio(self):
        """TG (919,813) vs AP (376,000). Diff = 543,813."""
        q = "Compare population between TG and AP"
        res = process_query(QueryRequest(question=q, session_id="prec_state_state"))
        agg = res.aggregation
        self.assertIsNotNone(agg)

        self.assertEqual(agg["left_value"], 919813)
        self.assertEqual(agg["right_value"], 376000)
        self.assertEqual(agg["absolute_difference"], 543813)
        self.assertIn("543,813", res.answer)
        self.assertEqual(agg["verification_status"], "VERIFIED")

    def test_10_capital_vs_capital(self):
        """Hyderabad vs Amaravati population comparison."""
        q = "Compare population of Hyderabad and Amaravati"
        res = process_query(QueryRequest(question=q, session_id="prec_cap_cap"))
        agg = res.aggregation
        self.assertIsNotNone(agg)

        self.assertEqual(agg["left_value"], 919813)
        self.assertEqual(agg["right_value"], 376000)
        self.assertEqual(agg["absolute_difference"], 543813)
        self.assertIn("543,813", res.answer)

    # -----------------------------------------------------------------
    # 6. ALL METRICS & DYNAMIC UNITS
    # -----------------------------------------------------------------

    def test_11_males_comparison_unit(self):
        """TG Village 1 (58,045) vs Village 2 (40,488) males. Diff = 17,557 males."""
        q = "Compare males in village 1 and village 2 in TG"
        res = process_query(QueryRequest(question=q, session_id="prec_males"))
        agg = res.aggregation
        self.assertIsNotNone(agg)

        self.assertEqual(agg["unit"], "males")
        self.assertEqual(agg["left_value"], 58045)
        self.assertEqual(agg["right_value"], 40488)
        self.assertEqual(agg["absolute_difference"], 17557)
        self.assertIn("17,557", res.answer)
        self.assertIn("males", res.answer.lower())

    def test_12_females_comparison_unit(self):
        """TG Village 1 (59,260) vs Village 2 (38,793) females. Diff = 20,467 females."""
        q = "Compare females in village 1 and village 2 in TG"
        res = process_query(QueryRequest(question=q, session_id="prec_females"))
        agg = res.aggregation
        self.assertIsNotNone(agg)

        self.assertEqual(agg["unit"], "females")
        self.assertEqual(agg["left_value"], 59260)
        self.assertEqual(agg["right_value"], 38793)
        self.assertEqual(agg["absolute_difference"], 20467)
        self.assertIn("20,467", res.answer)
        self.assertIn("females", res.answer.lower())

    def test_13_households_comparison_unit(self):
        """TG Village 1 (26,218) vs Village 2 (22,551) households. Diff = 3,667 households."""
        q = "Compare households between village 1 and village 2 in TG"
        res = process_query(QueryRequest(question=q, session_id="prec_hh"))
        agg = res.aggregation
        self.assertIsNotNone(agg)

        self.assertEqual(agg["unit"], "households")
        self.assertEqual(agg["left_value"], 26218)
        self.assertEqual(agg["right_value"], 22551)
        self.assertEqual(agg["absolute_difference"], 3667)
        self.assertIn("3,667", res.answer)
        self.assertIn("households", res.answer.lower())

    def test_14_area_comparison_unit(self):
        """TG Village 1 (7.7) vs Village 2 (10.67) sq km. Village 1 has 2.97 fewer sq km."""
        q = "Compare area of village 1 and village 2 in TG"
        res = process_query(QueryRequest(question=q, session_id="prec_area"))
        agg = res.aggregation
        self.assertIsNotNone(agg)

        self.assertEqual(agg["unit"], "sq km")
        self.assertEqual(agg["left_value"], 7.7)
        self.assertEqual(agg["right_value"], 10.67)
        self.assertAlmostEqual(agg["absolute_difference"], 2.97, places=2)
        self.assertIn("2.97", res.answer)
        self.assertIn("sq km", res.answer.lower())

    def test_15_literacy_rate_unit(self):
        """TG Village 1 (96.9) vs Village 2 (94.3) literacy. Diff = 2.6 %."""
        q = "Compare literacy between village 1 and village 2 in TG"
        res = process_query(QueryRequest(question=q, session_id="prec_lit"))
        agg = res.aggregation
        self.assertIsNotNone(agg)

        self.assertEqual(agg["unit"], "%")
        self.assertEqual(agg["left_value"], 96.9)
        self.assertEqual(agg["right_value"], 94.3)
        self.assertAlmostEqual(agg["absolute_difference"], 2.6, places=1)
        self.assertIn("2.6", res.answer)

    def test_16_village_count_unit(self):
        """Compare village counts between states."""
        q = "Which state has more villages, TG or AP?"
        res = process_query(QueryRequest(question=q, session_id="prec_v_count"))
        agg = res.aggregation
        self.assertIsNotNone(agg)

        self.assertEqual(agg["unit"], "villages")
        self.assertEqual(agg["left_value"], 10)
        self.assertEqual(agg["right_value"], 10)
        self.assertEqual(agg["absolute_difference"], 0)
        self.assertIn("same", res.answer.lower())

    # -----------------------------------------------------------------
    # 7. PERCENTAGE, RATIO, & BOOLEAN COMPARISONS
    # -----------------------------------------------------------------

    def test_17_percentage_more_calculation(self):
        """Village 1 (117,305) vs Village 2 (79,281) -> (38,024 / 79,281) * 100 = 47.96%."""
        q = "What percentage more population is in village 1 compared to village 2 in TG?"
        res = process_query(QueryRequest(question=q, session_id="prec_pct_more"))
        agg = res.aggregation
        self.assertIsNotNone(agg)

        self.assertAlmostEqual(agg["percentage_difference"], 47.96, places=1)
        self.assertIn("47.96%", res.answer)

    def test_18_ratio_calculation_and_formatting(self):
        """Ratio between TG and AP total village population."""
        q = "What is the ratio of population between TG and AP?"
        res = process_query(QueryRequest(question=q, session_id="prec_ratio"))
        agg = res.aggregation
        self.assertIsNotNone(agg)

        self.assertAlmostEqual(agg["ratio"], round(919813 / 376000, 3), places=2)
        self.assertIn("49:20", res.answer)

    def test_19_boolean_inquiries(self):
        """Is Hyderabad more populated than Amaravati? -> Yes."""
        q = "Is Hyderabad more populated than Amaravati?"
        res = process_query(QueryRequest(question=q, session_id="prec_bool_1"))
        self.assertEqual(res.operation, "BOOLEAN_CHECK")
        self.assertIn("Yes", res.answer)

        q2 = "Is Amaravati larger in population than Hyderabad?"
        res2 = process_query(QueryRequest(question=q2, session_id="prec_bool_2"))
        self.assertEqual(res2.operation, "BOOLEAN_CHECK")
        self.assertIn("No", res2.answer)

    # -----------------------------------------------------------------
    # 8. MULTI-ENTITY & RANKING WITH TIES
    # -----------------------------------------------------------------

    def test_20_ranking_top_two(self):
        """Compare top two villages across India or within state."""
        q = "Compare top two villages by population"
        res = process_query(QueryRequest(question=q, session_id="prec_top2"))
        self.assertIn(res.operation, ["COMPARE", "LOOKUP", "RANKING"])
        self.assertIn("more", res.answer.lower())

    def test_21_ranking_tie_handling(self):
        """Verify tie formatting when top two entities have identical metric values."""
        plan = ComparisonPlan(
            entities=[
                ComparisonEntity(name="Entity A", entity_type="GENERIC"),
                ComparisonEntity(name="Entity B", entity_type="GENERIC")
            ],
            attribute="Score",
            comparison_type="RANKING",
            scope="RANKING"
        )
        ans = universal_comparison_engine._format_answer(
            plan=plan,
            left_ent=plan.entities[0],
            right_ent=plan.entities[1],
            val1=100.0,
            val2=100.0,
            directed_diff=0.0,
            abs_diff=0.0,
            higher_ent="Equal",
            lower_ent="Equal",
            is_equal=True,
            pct_diff=None,
            ratio_val=None,
            bool_result=None,
            unit="points"
        )
        self.assertIn("tied", ans.lower())
        self.assertIn("100", ans)

    # -----------------------------------------------------------------
    # 9. ATTRIBUTE COMPATIBILITY & ERROR HANDLING
    # -----------------------------------------------------------------

    def test_22_attribute_incompatibility_rejection(self):
        """Semantic check: Population vs Area must be rejected."""
        compat, msg = universal_comparison_engine.check_attribute_compatibility("Population", "Area_Sq_Km")
        self.assertFalse(compat)
        self.assertIn("only using the same compatible attribute", msg)

        compat2, _ = universal_comparison_engine.check_attribute_compatibility("Population", "No_of_Males")
        # Incompatible units (people vs males)
        self.assertFalse(compat2)

        compat3, _ = universal_comparison_engine.check_attribute_compatibility("Population", "Population")
        self.assertTrue(compat3)

    def test_23_division_by_zero_protection(self):
        """Percentage comparison where reference value is zero."""
        plan = ComparisonPlan(
            entities=[
                ComparisonEntity(name="Entity A", entity_type="GENERIC"),
                ComparisonEntity(name="Entity B", entity_type="GENERIC")
            ],
            attribute="Population",
            comparison_type="PERCENTAGE_DIFFERENCE"
        )
        # Mocking plan execution where val2 is 0
        df_zero = pd.DataFrame([
            {"Name": "Entity A", "Population": 50},
            {"Name": "Entity B", "Population": 0}
        ])
        res = universal_comparison_engine.execute(plan, df=df_zero)
        self.assertEqual(res.status, "DIVISION_BY_ZERO")
        self.assertIn("cannot be calculated because the reference value is zero", res.answer)

    def test_24_missing_value_explicit_explanation(self):
        """Missing value must explicitly report missing data and never silently assume 0."""
        plan = ComparisonPlan(
            entities=[
                ComparisonEntity(name="Entity A", entity_type="GENERIC"),
                ComparisonEntity(name="Entity B", entity_type="GENERIC")
            ],
            attribute="Population",
            comparison_type="ABSOLUTE_DIFFERENCE"
        )
        df_nan = pd.DataFrame([
            {"Name": "Entity A", "Population": 50},
            {"Name": "Entity B", "Population": None}
        ])
        res = universal_comparison_engine.execute(plan, df=df_nan)
        self.assertEqual(res.status, "MISSING_VALUE")
        self.assertIn("data is missing for Entity B", res.answer)
        self.assertEqual(res.verification_status, "MISSING_VALUE")

    # -----------------------------------------------------------------
    # 10. MULTI-TURN FOLLOW-UP CONTEXT & EXPLICIT OVERRIDE
    # -----------------------------------------------------------------

    def test_25_multi_turn_followup_and_explicit_override(self):
        """Test sequence:
        Turn 1: Compare village 1 and village 2 in TG
        Turn 2: What about literacy? (Follow-up inherits Village 1 and Village 2)
        Turn 3: How much more? (Follow-up computes difference for literacy)
        Turn 4: Compare Karnataka and AP (Explicit override discards previous villages)
        """
        session = "session_prec_multiturn_flow"

        # Turn 1: Compare village 1 and village 2 in TG
        res1 = process_query(QueryRequest(question="Compare village 1 and village 2 in TG", session_id=session))
        self.assertIn("38,024", res1.answer)
        self.assertIn("people", res1.answer.lower())

        # Turn 2: What about literacy?
        res2 = process_query(QueryRequest(question="What about literacy?", session_id=session))
        self.assertIn("2.6", res2.answer)
        self.assertIn("96.9", res2.answer)
        self.assertIn("94.3", res2.answer)

        # Turn 3: How much more?
        res3 = process_query(QueryRequest(question="How much more?", session_id=session))
        self.assertIn("2.6", res3.answer)

        # Turn 4: Explicit Override: Compare Karnataka and AP
        # Must compare Karnataka and Andhra Pradesh, NOT mixing Village 1/2 from TG
        res4 = process_query(QueryRequest(question="Compare Karnataka and AP", session_id=session))
        self.assertIn("Karnataka", res4.answer)
        self.assertIn("Andhra Pradesh", res4.answer)
        self.assertNotIn("Hyderabad_Village_01", res4.answer)
        self.assertNotIn("Village 1", res4.answer)

    # -----------------------------------------------------------------
    # 11. ROBUSTNESS, TYPOS, HINGLISH, & IDS
    # -----------------------------------------------------------------

    def test_26_typo_resilience_in_comparison(self):
        """Typo tolerance: 'comapre villag 1 and villag 2 in telengana'."""
        q = "comapre villag 1 and villag 2 in telengana"
        res = process_query(QueryRequest(question=q, session_id="prec_typo_1"))
        self.assertIn("38,024", res.answer)

    def test_27_hinglish_comparison(self):
        """Hinglish query: 'village 1 aur village 2 me se kiska population jyada hai in TG'."""
        q = "village 1 aur village 2 me se kiska population jyada hai in TG"
        res = process_query(QueryRequest(question=q, session_id="prec_hinglish"))
        self.assertIn("38,024", res.answer)
        self.assertIn("more", res.answer.lower())

    def test_28_cross_state_village_ids(self):
        """Explicit Village ID comparison: 'Compare TG-001 and AP-002'."""
        q = "Compare TG-001 and AP-002"
        res = process_query(QueryRequest(question=q, session_id="prec_ids"))
        agg = res.aggregation
        self.assertIsNotNone(agg)
        self.assertEqual(agg["left_value"], 117305)
        self.assertEqual(agg["right_value"], 42844)
        self.assertEqual(agg["absolute_difference"], 74461)
        self.assertIn("74,461", res.answer)

    def test_29_provenance_and_audit_trail(self):
        """Verify provenance structure and verification status."""
        q = "Compare village 1 and village 2 in TG"
        res = process_query(QueryRequest(question=q, session_id="prec_prov"))
        prov = res.provenance
        self.assertIsNotNone(prov)
        self.assertEqual(prov.get("verification_status"), "VERIFIED")
        self.assertIn("calculation_formula", prov)
        self.assertIn("columns_used", prov)

    def test_30_debug_trace_verification(self):
        """Verify debug_trace contains 13-point verification dictionary."""
        q = "Compare village 1 and village 2 in TG"
        res = process_query(QueryRequest(question=q, session_id="prec_trace"))
        trace = res.debug_trace
        self.assertIsNotNone(trace)
        verif = trace.get("verification")
        self.assertIsNotNone(verif)
        self.assertTrue(verif.get("CALCULATION_CHECK"))
        self.assertTrue(verif.get("DIRECTION_CHECK"))

    def test_31_multi_entity_comparison(self):
        """Compare 3 states: Telangana, Andhra Pradesh, and Karnataka."""
        q = "Compare population of Telangana, Andhra Pradesh, and Karnataka"
        res = process_query(QueryRequest(question=q, session_id="prec_multi_3"))
        self.assertIn("Telangana", res.answer)
        self.assertIn("Andhra Pradesh", res.answer)
        self.assertIn("Karnataka", res.answer)

    def test_32_generic_tabular_comparison(self):
        """Compare two employees in sample_employees dataset."""
        if SAMPLE_EMPLOYEES.exists():
            dataset_loader.load_dataset(SAMPLE_EMPLOYEES)
            q = "Compare salary of Rahul Sharma and Priya Patel"
            res = process_query(QueryRequest(question=q, session_id="prec_emp"))
            self.assertIn("Rahul Sharma", res.answer)
            self.assertIn("Priya Patel", res.answer)
            self.assertIn("currency", res.answer.lower())
            # Re-load main dataset after test
            dataset_loader.load_dataset(MAIN_DATASET)
            parent_child_registry._initialize_paths()

    def test_33_never_negative_response(self):
        """Verify that answers never contain negative numeric tokens like '-38,024' or '-38024'."""
        queries = [
            "How much population is more in village 2 compared to village 1 in TG?",
            "How much population is more in village 1 compared to village 2 in TG?",
            "Compare village 2 and village 1 in TG",
            "Difference between Amaravati and Hyderabad population"
        ]
        import re as regex
        for q in queries:
            res = process_query(QueryRequest(question=q, session_id="prec_noneg"))
            neg_matches = regex.findall(r"-\s*\d[\d,]*", res.answer)
            self.assertEqual(neg_matches, [], f"Found negative numbers {neg_matches} in answer: '{res.answer}'")

    def test_34_unknown_entity_guard(self):
        """Fictional entities must not hallucinate answers from LLM knowledge."""
        q = "Compare population of Atlantis and El Dorado"
        res = process_query(QueryRequest(question=q, session_id="prec_unknown"))
        # Should not claim a numeric difference between Atlantis and El Dorado
        self.assertNotIn("Atlantis has", res.answer)

    def test_35_attribute_alias_diversity(self):
        """Test aliases: 'size' -> Area, 'education' -> Literacy, 'residents' -> Population."""
        q1 = "Compare size of village 1 and village 2 in TG"
        res1 = process_query(QueryRequest(question=q1, session_id="prec_alias_1"))
        self.assertIn("sq km", res1.answer.lower())

        q2 = "Compare education between village 1 and village 2 in TG"
        res2 = process_query(QueryRequest(question=q2, session_id="prec_alias_2"))
        self.assertIn("2.6", res2.answer)


if __name__ == "__main__":
    unittest.main()
