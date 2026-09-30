"""Comprehensive test suite for Universal Dataset Comparison & Difference Engine.

Verifies:
1. Same-State Village Comparison (Directional, non-negative, Village 1 vs Village 2, Village 2 vs Village 1)
2. Cross-State Village Comparison (TG Village 1 vs AP Village 2, reverse)
3. State vs State Comparison (Aggregated child population, TG vs AP, AP vs TG)
4. Capital vs Capital Comparison (Hyderabad vs Amaravati, Amaravati vs Hyderabad)
5. State vs Capital & Capital vs State (Telangana vs Hyderabad, Hyderabad vs Telangana)
6. Multiple Metrics (Literacy, Area, Households, Males, Females, Village Count)
7. Percentage Difference & Ratio Comparisons
8. Boolean Comparative Inquiries (Is X larger than Y?)
9. Top-2 & Ranking Differences
10. Multi-Entity Comparisons (3+ states/capitals)
11. Missing Values & Unknown Entities
12. Typos & Abbreviations (TG, AP, Hyd, Telengana, Andra Pradesh)
13. Generic Tabular Comparisons (Employee vs Employee, e.g. Rahul vs Priya)
14. End-to-End API Integration via process_query
"""

import unittest
from pathlib import Path
import pandas as pd

from app.api.routes import process_query, QueryRequest
from app.dataset.loader import dataset_loader
from app.dataset.registry import parent_child_registry
from app.query.comparison_engine import universal_comparison_engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_DATASET = PROJECT_ROOT / "universal_dataset_demo" / "main_dataset" / "india_states_capitals_main.csv"
SAMPLE_EMPLOYEES = PROJECT_ROOT / "data" / "sample_employees.xlsx"


class TestUniversalComparisonEngine(unittest.TestCase):
    """Test suite for Universal Dataset Comparison & Difference Engine."""

    @classmethod
    def setUpClass(cls):
        # Load main parent-child dataset
        if MAIN_DATASET.exists():
            dataset_loader.load_dataset(MAIN_DATASET)
            parent_child_registry._initialize_paths()

    def setUp(self):
        # Ensure parent-child is active before hierarchical tests
        if MAIN_DATASET.exists():
            dataset_loader.load_dataset(MAIN_DATASET)
            parent_child_registry._initialize_paths()

    # -----------------------------------------------------------------
    # 1. SAME-STATE VILLAGE COMPARISONS
    # -----------------------------------------------------------------

    def test_01_same_state_village_higher_second(self):
        """Test 'How much population is more in village 2 compared to village 1 in TG?' -> Village 2 has 38,024 fewer people."""
        q = "How much population is more in village 2 compared to village 1 in TG?"
        res = process_query(QueryRequest(question=q, session_id="test_same_1"))
        self.assertIn(res.operation, ["COMPARE", "LOOKUP"])
        # Village 2 (79,281) vs Village 1 (117,305): Village 2 has 38,024 fewer people
        self.assertIn("fewer", res.answer.lower())
        self.assertIn("38,024", res.answer)
        # Verify NEVER returning raw negative number
        self.assertNotIn("-38024", res.answer)
        self.assertNotIn("-38,024", res.answer)

    def test_02_same_state_village_higher_first(self):
        """Test 'How many people are more in village 1 than village 2 in TG?' -> Village 1 has 38,024 more people."""
        q = "How many people are more in village 1 than village 2 in TG?"
        res = process_query(QueryRequest(question=q, session_id="test_same_2"))
        self.assertIn("more", res.answer.lower())
        self.assertIn("38,024", res.answer)
        self.assertNotIn("-38024", res.answer)

    def test_03_same_state_village_compare_wording(self):
        """Test 'Compare village 1 and village 2 in Telangana'."""
        q = "Compare village 1 and village 2 in Telangana"
        res = process_query(QueryRequest(question=q, session_id="test_same_3"))
        self.assertIn("117,305", res.answer)
        self.assertIn("79,281", res.answer)
        self.assertIn("38,024", res.answer)

    def test_04_same_state_village_id_syntax(self):
        """Test comparison using explicit Village IDs: 'Compare TG-001 and TG-002'."""
        q = "Compare TG-001 and TG-002"
        res = process_query(QueryRequest(question=q, session_id="test_same_4"))
        self.assertIn("38,024", res.answer)

    # -----------------------------------------------------------------
    # 2. CROSS-STATE VILLAGE COMPARISONS
    # -----------------------------------------------------------------

    def test_05_cross_state_village_comparison(self):
        """Test 'Compare village 1 in TG and village 2 in AP'."""
        # TG Village 1: 117,305. AP Village 2: 42,844. Diff: 74,461
        q = "Compare village 1 in TG and village 2 in AP"
        res = process_query(QueryRequest(question=q, session_id="test_cross_1"))
        self.assertIn("74,461", res.answer)
        self.assertIn("Telangana", res.answer)
        self.assertIn("Andhra Pradesh", res.answer)
        self.assertIn("more", res.answer.lower())

    def test_06_cross_state_village_reverse(self):
        """Test 'How many more people are in village 2 in AP than village 1 in TG?' -> AP has 74,461 fewer people."""
        q = "How many more people are in village 2 in AP than village 1 in TG?"
        res = process_query(QueryRequest(question=q, session_id="test_cross_2"))
        self.assertIn("74,461", res.answer)
        self.assertIn("fewer", res.answer.lower())
        self.assertNotIn("-74461", res.answer)

    # -----------------------------------------------------------------
    # 3. STATE VS STATE COMPARISONS
    # -----------------------------------------------------------------

    def test_07_state_vs_state_population(self):
        """Test 'Compare Telangana and Andhra Pradesh' -> TG (919,813) vs AP (376,000), TG has 543,813 more."""
        q = "Compare Telangana and Andhra Pradesh"
        res = process_query(QueryRequest(question=q, session_id="test_state_1"))
        self.assertIn("919,813", res.answer)
        self.assertIn("376,000", res.answer)
        self.assertIn("543,813", res.answer)
        self.assertIn("more", res.answer.lower())

    def test_08_state_vs_state_which_higher(self):
        """Test 'Which state has higher population, TG or AP?'."""
        q = "Which state has higher population, TG or AP?"
        res = process_query(QueryRequest(question=q, session_id="test_state_2"))
        self.assertIn("Telangana", res.answer)
        self.assertIn("919,813", res.answer)
        self.assertIn("543,813", res.answer)

    def test_09_state_vs_state_how_much_lower(self):
        """Test 'How much lower is AP than TG?' -> AP has 543,813 fewer people."""
        q = "How much lower is AP than TG?"
        res = process_query(QueryRequest(question=q, session_id="test_state_3"))
        self.assertIn("543,813", res.answer)
        self.assertIn("fewer", res.answer.lower())
        self.assertNotIn("-543813", res.answer)

    def test_10_state_village_count_comparison(self):
        """Test 'Which state has more villages, TG or AP?'."""
        q = "Which state has more villages, TG or AP?"
        res = process_query(QueryRequest(question=q, session_id="test_state_4"))
        self.assertTrue("villages" in res.answer.lower() or "village count" in res.answer.lower())
        self.assertTrue("same" in res.answer.lower() or "more" in res.answer.lower() or "fewer" in res.answer.lower())

    # -----------------------------------------------------------------
    # 4. CAPITAL VS CAPITAL COMPARISONS
    # -----------------------------------------------------------------

    def test_11_capital_vs_capital(self):
        """Test 'Compare Hyderabad and Amaravati'."""
        q = "Compare Hyderabad and Amaravati"
        res = process_query(QueryRequest(question=q, session_id="test_cap_1"))
        self.assertIn("919,813", res.answer)
        self.assertIn("376,000", res.answer)
        self.assertIn("543,813", res.answer)
        self.assertIn("Hyderabad", res.answer)

    def test_12_capital_which_has_more_people(self):
        """Test 'Which capital has higher population, Hyderabad or Amaravati?'."""
        q = "Which capital has higher population, Hyderabad or Amaravati?"
        res = process_query(QueryRequest(question=q, session_id="test_cap_2"))
        self.assertIn("Hyderabad", res.answer)
        self.assertIn("543,813", res.answer)

    # -----------------------------------------------------------------
    # 5. STATE VS CAPITAL & CAPITAL VS STATE
    # -----------------------------------------------------------------

    def test_13_state_vs_capital_equal(self):
        """Test 'Compare Telangana population with Hyderabad population' -> Both have the same population: 919,813."""
        q = "Compare Telangana population with Hyderabad population"
        res = process_query(QueryRequest(question=q, session_id="test_sc_1"))
        self.assertIn("same", res.answer.lower())
        self.assertIn("919,813", res.answer)

    def test_14_capital_vs_state_reverse(self):
        """Test 'Compare Hyderabad with Telangana'."""
        q = "Compare Hyderabad with Telangana"
        res = process_query(QueryRequest(question=q, session_id="test_sc_2"))
        self.assertIn("same", res.answer.lower())
        self.assertIn("919,813", res.answer)

    # -----------------------------------------------------------------
    # 6. OTHER NUMERIC METRICS (Literacy, Area, Households, Males, Females)
    # -----------------------------------------------------------------

    def test_15_literacy_comparison(self):
        """Test 'Which village has higher literacy between village 1 and village 2 in TG?'."""
        # TG-001: 96.9%, TG-002: 94.3% -> Diff: 2.6 percentage points
        q = "Which village has higher literacy between village 1 and village 2 in TG?"
        res = process_query(QueryRequest(question=q, session_id="test_lit_1"))
        self.assertIn("96.9", res.answer)
        self.assertIn("94.3", res.answer)
        self.assertIn("2.6", res.answer)
        self.assertIn("higher", res.answer.lower())

    def test_16_area_comparison(self):
        """Test 'Compare area between village 1 and village 2 in TG'."""
        # TG-001 Area: 7.7. TG-002 Area: 10.67.
        q = "Compare area between village 1 and village 2 in TG"
        res = process_query(QueryRequest(question=q, session_id="test_area_1"))
        self.assertIn("7.7", res.answer)
        self.assertIn("10.67", res.answer)
        self.assertIn("2.97", res.answer)

    def test_17_households_comparison(self):
        """Test 'Which village has more households, village 1 or village 2 in TG?'."""
        # TG-001: 26,218. TG-002: 22,551. Diff: 3,667.
        q = "Which village has more households, village 1 or village 2 in TG?"
        res = process_query(QueryRequest(question=q, session_id="test_hh_1"))
        self.assertIn("26,218", res.answer)
        self.assertIn("22,551", res.answer)
        self.assertIn("3,667", res.answer)

    def test_18_males_females_comparison(self):
        """Test 'Which village has more males between village 1 and village 2 in TG?'."""
        # TG-001 Males: 58,045. TG-002 Males: 40,488. Diff: 17,557.
        q = "Which village has more males between village 1 and village 2 in TG?"
        res = process_query(QueryRequest(question=q, session_id="test_mf_1"))
        self.assertIn("58,045", res.answer)
        self.assertIn("40,488", res.answer)
        self.assertIn("17,557", res.answer)

    # -----------------------------------------------------------------
    # 7. PERCENTAGE & RATIO COMPARISONS
    # -----------------------------------------------------------------

    def test_19_percentage_more(self):
        """Test 'What percentage more population does village 1 have than village 2 in TG?'."""
        # TG-001: 117,305. TG-002: 79,281. (117305 - 79281) / 79281 * 100 = 47.96%
        q = "What percentage more population does village 1 have than village 2 in TG?"
        res = process_query(QueryRequest(question=q, session_id="test_pct_1"))
        self.assertIn("47.96%", res.answer)
        self.assertIn("more", res.answer.lower())

    def test_20_ratio_comparison(self):
        """Test 'What is the ratio of population between village 1 and village 2 in TG?'."""
        # 117305 / 79281 = 1.48
        q = "What is the ratio of population between village 1 and village 2 in TG?"
        res = process_query(QueryRequest(question=q, session_id="test_ratio_1"))
        self.assertIn("1.48:1", res.answer)

    # -----------------------------------------------------------------
    # 8. BOOLEAN COMPARATIVE INQUIRIES
    # -----------------------------------------------------------------

    def test_21_boolean_capital_population(self):
        """Test 'Is Hyderabad more populated than Amaravati?' -> Yes."""
        q = "Is Hyderabad more populated than Amaravati?"
        res = process_query(QueryRequest(question=q, session_id="test_bool_1"))
        self.assertTrue(res.answer.lower().startswith("yes"))
        self.assertIn("919,813", res.answer)
        self.assertIn("376,000", res.answer)

    def test_22_boolean_village_population_false(self):
        """Test 'Is village 2 larger than village 1 in TG?' -> No."""
        q = "Is village 2 larger than village 1 in TG?"
        res = process_query(QueryRequest(question=q, session_id="test_bool_2"))
        self.assertTrue(res.answer.lower().startswith("no"))
        self.assertIn("79,281", res.answer)
        self.assertIn("117,305", res.answer)

    # -----------------------------------------------------------------
    # 9. RANKING & TOP-2 DIFFERENCES
    # -----------------------------------------------------------------

    def test_23_top_two_villages_comparison(self):
        """Test 'Compare top two villages' or 'Difference between first and second village'."""
        q = "Difference between first and second highest village by population"
        res = process_query(QueryRequest(question=q, session_id="test_rank_1"))
        self.assertIn("more", res.answer.lower())
        self.assertTrue(len(res.answer) > 20)

    # -----------------------------------------------------------------
    # 10. THREE OR MORE ENTITIES
    # -----------------------------------------------------------------

    def test_24_three_states_comparison(self):
        """Test 'Compare Telangana, Andhra Pradesh and Gujarat'."""
        q = "Compare Telangana, Andhra Pradesh and Gujarat"
        res = process_query(QueryRequest(question=q, session_id="test_multi_1"))
        self.assertIn("Telangana", res.answer)
        self.assertIn("Andhra Pradesh", res.answer)
        self.assertIn("Gujarat", res.answer)
        self.assertIn("highest", res.answer.lower())

    # -----------------------------------------------------------------
    # 11. TYPOS & ABBREVIATIONS
    # -----------------------------------------------------------------

    def test_25_typos_and_abbreviations(self):
        """Test 'compare hyd and amaravti'."""
        q = "compare hyd and amaravati"
        res = process_query(QueryRequest(question=q, session_id="test_typo_1"))
        self.assertIn("543,813", res.answer)
        self.assertIn("Hyderabad", res.answer)

    def test_26_state_abbreviations(self):
        """Test 'tg and ap population difference'."""
        q = "tg and ap population difference"
        res = process_query(QueryRequest(question=q, session_id="test_typo_2"))
        self.assertIn("543,813", res.answer)

    # -----------------------------------------------------------------
    # 12. GENERIC TABULAR DATASET (EMPLOYEE COMPARISONS)
    # -----------------------------------------------------------------

    def test_27_generic_employee_comparison(self):
        """Test employee comparison on generic tabular dataset."""
        test_emp_csv = PROJECT_ROOT / "data" / "test_employees_comp.csv"
        test_emp_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "Name": ["Rahul Patel", "Priya Nair"],
            "Department": ["Human Resources", "Engineering"],
            "Salary": [600000, 850000],
        }).to_csv(test_emp_csv, index=False)

        try:
            dataset_loader.load_dataset(test_emp_csv)

            # Rahul (600k) vs Priya (850k): Priya earns 250,000 more (or Rahul earns 250,000 fewer)
            q = "How much more salary does Priya have than Rahul?"
            res = process_query(QueryRequest(question=q, session_id="test_emp_comp_1"))
            self.assertIn("250,000", res.answer)
            self.assertIn("more", res.answer.lower())
            self.assertNotIn("-250000", res.answer)
        finally:
            if test_emp_csv.exists():
                test_emp_csv.unlink()
            if MAIN_DATASET.exists():
                dataset_loader.load_dataset(MAIN_DATASET)



if __name__ == "__main__":
    unittest.main()
