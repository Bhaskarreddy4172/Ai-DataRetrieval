"""Comprehensive test suite for strict No-Match, Unsupported, Entity-Not-Found,

Column-Not-Found, Null-Value, and Ambiguity Query Handling.
Enforces: CLIENT DATA = SOURCE OF TRUTH (Zero Hallucination).
"""

import pandas as pd
import pytest

from app.api.routes import process_query, QueryRequest
from app.dataset.loader import dataset_loader
from app.dataset.indexer import dataset_indexer


@pytest.fixture(autouse=True)
def setup_states_dataset():
    """Ensure Indian States dataset is loaded before each test."""
    dataset_loader.load_sample("states")


class TestNoMatchAndUnsupportedHandling:
    """Test suite covering all 6 mandatory failure conditions and zero hallucinations."""

    def test_entity_not_found(self):
        """Condition 1: ENTITY_NOT_FOUND (e.g. 'What is the capital of Wakanda?')."""
        req = QueryRequest(question="What is the capital of Wakanda?", session_id="test_enf")
        res = process_query(req)
        assert res.operation == "NO_MATCH"
        assert res.result_count == 0
        assert "Wakanda" in res.answer
        assert "No matching state or entity was found in the uploaded dataset" in res.answer

    def test_column_not_found_population(self):
        """Condition 2: COLUMN_NOT_FOUND (e.g. 'What is the population of Telangana?')."""
        req = QueryRequest(question="What is the population of Telangana?", session_id="test_cnf_pop")
        res = process_query(req)
        assert res.operation in {"UNSUPPORTED_QUERY", "UNKNOWN"}
        assert res.result_count == 0
        assert "population" in res.answer.lower()
        assert "does not contain" in res.answer.lower()

    def test_unsupported_chief_minister(self):
        """Condition 6: UNSUPPORTED attribute (e.g. 'Who is the Chief Minister of Telangana?')."""
        req = QueryRequest(question="Who is the Chief Minister of Telangana?", session_id="test_unsupp_cm")
        res = process_query(req)
        assert res.operation in {"UNSUPPORTED_QUERY", "UNKNOWN"}
        assert res.result_count == 0
        assert "chief minister" in res.answer.lower()
        assert "does not contain" in res.answer.lower()

    def test_no_matching_rows_exact(self):
        """Condition 3: NO_MATCHING_ROWS (e.g. 'Which state has XYZ as capital?')."""
        req = QueryRequest(question="Which state has XYZ as capital?", session_id="test_nmr_exact")
        res = process_query(req)
        assert res.operation == "NO_MATCH"
        assert res.result_count == 0
        assert "Xyz" in res.answer or "XYZ" in res.answer
        assert "No matching" in res.answer

    def test_no_matching_rows_prefix(self):
        """Condition 3: NO_MATCHING_ROWS prefix (e.g. 'Which states have a capital beginning with Z?')."""
        req = QueryRequest(question="Which states have a capital beginning with Z?", session_id="test_nmr_prefix")
        res = process_query(req)
        assert res.result_count == 0
        assert "Z" in res.answer
        assert "No matching records were found" in res.answer

    def test_ambiguous_bare_attribute(self):
        """Condition 5: AMBIGUOUS (e.g. 'What is the capital?')."""
        req = QueryRequest(question="What is the capital?", session_id="test_ambig")
        res = process_query(req)
        assert res.operation == "CLARIFICATION"
        assert res.is_clarification is True
        assert "Could you specify which state you mean?" in res.answer

    def test_null_value_handling(self):
        """Condition 4: NULL_VALUE (Record exists, column exists, but cell is empty/null)."""
        temp_df = pd.DataFrame([
            {"_internal_row_id": 0, "state": "Telangana", "capital": "Hyderabad"},
            {"_internal_row_id": 1, "state": "ExampleLand", "capital": None}
        ])
        dataset_loader.dataframe = temp_df
        dataset_loader.dataset_name = "test_null_dataset"
        dataset_indexer.build_index(temp_df, "test_null_dataset")

        req = QueryRequest(question="What is ExampleLand's capital?", session_id="test_null")
        res = process_query(req)
        assert res.operation == "NULL_VALUE"
        assert res.result_count == 0
        assert "ExampleLand was found in the dataset, but its capital value is not available." in res.answer

        # Reset back
        dataset_loader.load_sample("states")

    def test_boolean_unknown_entity(self):
        """Boolean Check: Subject entity does not exist in dataset -> UNKNOWN."""
        req = QueryRequest(question="Is Wakanda a state in India?", session_id="test_bool_enf")
        res = process_query(req)
        assert res.operation == "BOOLEAN_CHECK"
        assert "UNKNOWN" in res.answer
        assert "Wakanda" in res.answer

    def test_boolean_unknown_column(self):
        """Boolean Check: Attribute does not exist in dataset -> UNKNOWN."""
        req = QueryRequest(question="Does Telangana have a population of 50 million?", session_id="test_bool_cnf")
        res = process_query(req)
        assert res.operation == "BOOLEAN_CHECK"
        assert "UNKNOWN" in res.answer
        assert "population" in res.answer.lower()

    def test_boolean_false_verification(self):
        """Boolean Check: Factual false verification with grounded truth."""
        req = QueryRequest(question="Is XYZ the capital of Telangana?", session_id="test_bool_false")
        res = process_query(req)
        assert res.operation == "BOOLEAN_CHECK"
        assert "False." in res.answer
        assert "Hyderabad" in res.answer

    def test_boolean_true_verification(self):
        """Boolean Check: Factual true verification with grounded truth."""
        req = QueryRequest(question="Is Hyderabad the capital of Telangana?", session_id="test_bool_true")
        res = process_query(req)
        assert res.operation == "BOOLEAN_CHECK"
        assert "True." in res.answer
        assert "Hyderabad" in res.answer

    def test_valid_grounded_lookups(self):
        """Ensure valid grounded lookups continue to work flawlessly."""
        req_wb = QueryRequest(question="What is West Bengal's capital?", session_id="test_wb")
        res_wb = process_query(req_wb)
        assert res_wb.operation == "LOOKUP"
        assert "Kolkata" in res_wb.answer

        req_blr = QueryRequest(question="Which state does Bangalore belong to?", session_id="test_blr")
        res_blr = process_query(req_blr)
        assert res_blr.operation == "LOOKUP"
        assert "Karnataka" in res_blr.answer
