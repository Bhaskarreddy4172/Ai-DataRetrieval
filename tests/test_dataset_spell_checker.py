import pandas as pd
import pytest

from app.api.routes import process_query, QueryRequest
from app.dataset.loader import dataset_loader
from app.dataset.spell_checker import dataset_spell_checker, DatasetSpellChecker


@pytest.fixture(autouse=True)
def setup_dataset():
    dataset_loader.load_sample("states")


class TestDatasetSpellCheckerEngine:
    def test_vocabulary_built(self):
        df = dataset_loader.dataframe
        assert not df.empty
        res = dataset_spell_checker.resolve_candidate("telangana")
        assert res is not None
        assert res.canonical_value == "Telangana"

    def test_deletion_error(self):
        res1 = dataset_spell_checker.resolve_candidate("telngana")
        assert res1 is not None
        assert res1.canonical_value == "Telangana"
        assert res1.confidence >= 0.85

        res2 = dataset_spell_checker.resolve_candidate("telagana")
        assert res2 is not None
        assert res2.canonical_value == "Telangana"

    def test_insertion_error(self):
        res1 = dataset_spell_checker.resolve_candidate("tellangana")
        assert res1 is not None
        assert res1.canonical_value == "Telangana"

        res2 = dataset_spell_checker.resolve_candidate("sikkimm")
        assert res2 is not None
        assert res2.canonical_value == "Sikkim"

    def test_substitution_error(self):
        res1 = dataset_spell_checker.resolve_candidate("telengana")
        assert res1 is not None
        assert res1.canonical_value == "Telangana"

        res2 = dataset_spell_checker.resolve_candidate("gujrat")
        assert res2 is not None
        assert res2.canonical_value == "Gujarat"

    def test_transposition_error(self):
        res1 = dataset_spell_checker.resolve_candidate("telnagana")
        assert res1 is not None
        assert res1.canonical_value == "Telangana"

    def test_missing_space(self):
        res1 = dataset_spell_checker.resolve_candidate("westbengal")
        assert res1 is not None
        assert res1.canonical_value == "West Bengal"

        res2 = dataset_spell_checker.resolve_candidate("tamilnadu")
        assert res2 is not None
        assert res2.canonical_value == "Tamil Nadu"

        res3 = dataset_spell_checker.resolve_candidate("uttarpradesh")
        assert res3 is not None
        assert res3.canonical_value == "Uttar Pradesh"

    def test_extra_space_and_punctuation(self):
        res1 = dataset_spell_checker.resolve_candidate("west  bengal")
        assert res1 is not None
        assert res1.canonical_value == "West Bengal"

        res2 = dataset_spell_checker.resolve_candidate("west-bengal")
        assert res2 is not None
        assert res2.canonical_value == "West Bengal"

        res3 = dataset_spell_checker.resolve_candidate("andhra_pradesh")
        assert res3 is not None
        assert res3.canonical_value == "Andhra Pradesh"

    def test_phonetic_matching(self):
        res1 = dataset_spell_checker.resolve_candidate("sikim")
        assert res1 is not None
        assert res1.canonical_value == "Sikkim"

        res2 = dataset_spell_checker.resolve_candidate("cikkim")
        assert res2 is not None
        assert res2.canonical_value == "Sikkim"

        res3 = dataset_spell_checker.resolve_candidate("hydrabad")
        assert res3 is not None
        assert res3.canonical_value == "Hyderabad"

        res4 = dataset_spell_checker.resolve_candidate("karnatka")
        assert res4 is not None
        assert res4.canonical_value == "Karnataka"

    def test_column_name_typo(self):
        col1 = dataset_spell_checker.resolve_column("captial", ["state", "capital"])
        assert col1 == "capital"

        col2 = dataset_spell_checker.resolve_column("stae", ["state", "capital"])
        assert col2 == "state"

    def test_column_name_multiword_typo(self):
        test_cols = ["Employee ID", "Department", "Salary", "Performance Score"]
        checker = DatasetSpellChecker()
        sample_df = pd.DataFrame({
            "Employee ID": ["EMP-1001", "EMP-1002"],
            "Department": ["Engineering", "Sales"],
            "Salary": [80000, 75000],
            "Performance Score": [4.8, 4.2]
        })
        checker.build_vocabulary(sample_df, "test_employees", "hash_emp_1")

        assert checker.resolve_column("salry", test_cols) == "Salary"
        assert checker.resolve_column("dpartmnt", test_cols) == "Department"
        assert checker.resolve_column("performence score", test_cols) == "Performance Score"

    def test_id_protection(self):
        checker = DatasetSpellChecker()
        sample_df = pd.DataFrame({
            "Employee ID": ["EMP-1001", "EMP-1002", "EMP-2005"],
            "Name": ["John Doe", "Jane Smith", "Bob Jones"]
        })
        checker.build_vocabulary(sample_df, "id_test", "hash_id_1")

        res1 = checker.resolve_candidate("emp1001")
        assert res1 is not None
        assert res1.canonical_value == "EMP-1001"

        res2 = checker.resolve_candidate("EMP-1002")
        assert res2 is not None
        assert res2.canonical_value == "EMP-1002"

    def test_full_question_correction(self):
        meta = dataset_spell_checker.correct_full_question("what is the captial of westbengal")
        assert "West Bengal" in meta["corrected_question"]
        assert "capital" in meta["corrected_question"]
        assert len(meta["replacements"]) >= 1


class TestEndToEndTypoQueries:
    def test_e2e_capital_with_state_typo(self):
        req = QueryRequest(question="what is telengana capital", session_id="test_typo_tel")
        res = process_query(req)
        assert res.result_count >= 1
        assert "Hyderabad" in res.answer

    def test_e2e_both_column_and_entity_typo(self):
        req = QueryRequest(question="what is the captial of westbengal", session_id="test_typo_wb")
        res = process_query(req)
        assert res.result_count >= 1
        assert "Kolkata" in res.answer

    def test_e2e_reverse_lookup_with_city_typo(self):
        req = QueryRequest(question="which state is hydrabad in", session_id="test_typo_hyd")
        res = process_query(req)
        assert res.result_count >= 1
        assert "Telangana" in res.answer

    def test_e2e_possessive_with_typo(self):
        req = QueryRequest(question="karnatka's captial", session_id="test_typo_kar")
        res = process_query(req)
        assert res.result_count >= 1
        assert "Bengaluru" in res.answer

    def test_e2e_boolean_with_typos(self):
        req = QueryRequest(question="is hydrabad the captial of telengana?", session_id="test_typo_bool")
        res = process_query(req)
        assert "True" in res.answer
        assert "Hyderabad" in res.answer

    def test_e2e_zero_hallucination_on_unsupported_with_typo(self):
        req = QueryRequest(question="what is the population of telengana", session_id="test_typo_unsupp")
        res = process_query(req)
        assert res.operation in {"UNSUPPORTED_QUERY", "UNKNOWN"}
        assert res.result_count == 0
        assert "population" in res.answer.lower()
        assert "does not contain" in res.answer.lower()
