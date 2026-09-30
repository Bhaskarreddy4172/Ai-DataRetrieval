"""Targeted test suite for Relationship, Comparison, Equality, Membership, and 3-state (TRUE/FALSE/UNKNOWN) assertions."""

import unittest
from pathlib import Path
import pandas as pd

from app.dataset.loader import dataset_loader
from app.api.routes import process_query, QueryRequest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATES_DATASET = PROJECT_ROOT / "data" / "indian_states_capitals.csv"
EMPLOYEES_DATASET = PROJECT_ROOT / "data" / "sample_employees.xlsx"


class TestRelationshipComparisonEngine(unittest.TestCase):
    """Test suite verifying generic relationship, membership, equality, comparison, and cross-dataset isolation."""

    @classmethod
    def setUpClass(cls):
        # Load Indian States & Capitals dataset for state/capital relationship tests
        dataset_loader.load_dataset(STATES_DATASET)

    def test_01_state_membership_assertion(self):
        """Test 'is up belongs to uk' against indian_states_capitals.csv -> False."""
        dataset_loader.load_dataset(STATES_DATASET)
        
        # Test variations of UP vs UK belonging
        questions = [
            "is up belongs to uk",
            "does up belong to uk",
            "up belongs uk?",
            "is UP part of UK?",
            "does UP come under UK?",
        ]
        for q in questions:
            res = process_query(QueryRequest(question=q, session_id="test_rel_1"))
            self.assertEqual(res.operation, "BOOLEAN_CHECK")
            self.assertIn("no", res.answer.lower())

    def test_02_city_state_membership(self):
        """Test 'does hyd belong to ts?' -> True."""
        dataset_loader.load_dataset(STATES_DATASET)

        questions = [
            "does hyd belong to ts?",
            "is Hyderabad in Telangana?",
            "is hyd in telangana?",
            "does hydrabad belong to telengana?",
        ]
        for q in questions:
            res = process_query(QueryRequest(question=q, session_id="test_rel_2"))
            self.assertEqual(res.operation, "BOOLEAN_CHECK")
            self.assertIn("yes", res.answer.lower())

    def test_03_same_and_different_states(self):
        """Test 'ap and ts are same?' -> False, 'are ap and ts different?' -> True."""
        dataset_loader.load_dataset(STATES_DATASET)

        # Same questions -> False / No
        same_qs = [
            "ap and ts are same?",
            "ap ts same?",
            "are ap and ts same?",
            "is ap equal to ts?",
        ]
        for q in same_qs:
            res = process_query(QueryRequest(question=q, session_id="test_rel_3a"))
            self.assertEqual(res.operation, "BOOLEAN_CHECK")
            self.assertIn("no", res.answer.lower())

        # Different questions -> True / Yes
        diff_qs = [
            "are ap and ts different?",
            "ap and ts are different?",
        ]
        for q in diff_qs:
            res = process_query(QueryRequest(question=q, session_id="test_rel_3b"))
            self.assertEqual(res.operation, "BOOLEAN_CHECK")
            self.assertIn("yes", res.answer.lower())

    def test_04_employee_relationships_and_numeric_comparisons(self):
        """Test employee numeric comparisons, membership, and cross-row equality against sample_employees.xlsx."""
        if not EMPLOYEES_DATASET.exists():
            self.skipTest("sample_employees.xlsx was removed per user dataset cleanup")
        dataset_loader.load_dataset(EMPLOYEES_DATASET)

        # 1. Salary comparison: Rahul (600k) vs Priya (850k) -> False ("No...")
        res_sal = process_query(QueryRequest(question="rahul salary greater priya?", session_id="test_emp_1"))
        self.assertEqual(res_sal.operation, "BOOLEAN_CHECK")
        self.assertIn("no", res_sal.answer.lower())

        # Priya (850k) vs Rahul (600k) -> True ("Yes...")
        res_sal2 = process_query(QueryRequest(question="is Priya salary higher than Rahul?", session_id="test_emp_2"))
        self.assertEqual(res_sal2.operation, "BOOLEAN_CHECK")
        self.assertIn("yes", res_sal2.answer.lower())

        # 2. Same city cross-row: Rahul (Hyderabad) vs Priya (Bengaluru) -> False ("No...")
        res_city = process_query(QueryRequest(question="rahul priya same city?", session_id="test_emp_3"))
        self.assertEqual(res_city.operation, "BOOLEAN_CHECK")
        self.assertIn("no", res_city.answer.lower())

        # 3. Membership in department: Rahul in Engineering -> True ("Yes...")
        res_dept = process_query(QueryRequest(question="does Rahul belong to Engineering?", session_id="test_emp_4"))
        self.assertEqual(res_dept.operation, "BOOLEAN_CHECK")
        self.assertIn("yes", res_dept.answer.lower())

        # 4. Date comparison: Rahul (2021) join before Priya (2022) -> True ("Yes...")
        res_date = process_query(QueryRequest(question="did Rahul join before Priya?", session_id="test_emp_5"))
        self.assertEqual(res_date.operation, "BOOLEAN_CHECK")
        self.assertIn("yes", res_date.answer.lower())

    def test_05_unknown_result_when_data_not_in_dataset(self):
        """Test Absolute Dataset Truth Rule (Requirement #8 & #18): Returns UNKNOWN when data not in active dataset."""
        if not EMPLOYEES_DATASET.exists():
            self.skipTest("sample_employees.xlsx was removed per user dataset cleanup")
        dataset_loader.load_dataset(EMPLOYEES_DATASET)

        # When sample_employees.xlsx is active, asking about 'UP' and 'UK' must return UNKNOWN
        res_unkn = process_query(QueryRequest(question="is up belongs to uk", session_id="test_unkn_1"))
        self.assertEqual(res_unkn.operation, "BOOLEAN_CHECK")
        self.assertIn("can't determine", res_unkn.answer.lower())

    def test_06_superlative_cross_row_comparison(self):
        """Test multi-step superlative comparison: highest salary vs highest performance same city."""
        if not EMPLOYEES_DATASET.exists():
            self.skipTest("sample_employees.xlsx was removed per user dataset cleanup")
        dataset_loader.load_dataset(EMPLOYEES_DATASET)

        res_sup = process_query(QueryRequest(
            question="is the employee with the highest salary from the same city as the employee with the highest performance score?",
            session_id="test_sup_1"
        ))
        self.assertEqual(res_sup.operation, "BOOLEAN_CHECK")
        self.assertTrue(res_sup.answer.lower().startswith("yes") or res_sup.answer.lower().startswith("no"))


if __name__ == "__main__":
    unittest.main()
