"""Comprehensive test suite for possessive, word-order, and natural-language entity-attribute inquiries."""

import pytest
from app.api.routes import QueryRequest, process_query
from app.dataset.loader import dataset_loader
from app.query.planner import query_planner
from app.query.semantic_parser import semantic_relationship_parser
from app.router.question_router import question_router


@pytest.fixture(autouse=True)
def ensure_clean_state():
    """Ensure dataset loader is ready."""
    pass


class TestPossessiveAndSemanticLookup:
    """Test suite covering direct lookups, reverse lookups, possessives, and word-order variations."""

    def test_semantic_parser_possessive_direct(self):
        """Test semantic relationship parser on direct possessive formulations."""
        dataset_loader.load_sample("states")
        cols = dataset_loader.get_columns()

        variations = [
            "What is West Bengal's capital?",
            "West Bengal's capital",
            "West Bengal’s capital",
            "West Bengals capital",
            "West Bengal capital",
            "capital of West Bengal",
            "Which city is West Bengal's capital?",
            "West Bengal has which capital?",
            "the capital belonging to West Bengal",
            "capital for West Bengal",
        ]

        for q in variations:
            rel = semantic_relationship_parser.parse(q, cols)
            assert rel is not None, f"Failed to parse semantic relationship for: '{q}'"
            assert rel.subject_val == "West Bengal", f"Expected West Bengal, got '{rel.subject_val}' for '{q}'"
            assert rel.subject_col == "state", f"Expected state column, got '{rel.subject_col}' for '{q}'"
            assert rel.target_col == "capital", f"Expected capital target, got '{rel.target_col}' for '{q}'"
            assert rel.direction == "DIRECT"

    def test_semantic_parser_reverse_lookup(self):
        """Test reverse lookup formulations (e.g. Kolkata -> West Bengal)."""
        dataset_loader.load_sample("states")
        cols = dataset_loader.get_columns()

        reverse_variations = [
            "Which state has Kolkata as capital?",
            "Which state has Kolkata as its capital?",
            "Kolkata is the capital of which state?",
            "Kolkata is capital of which state?",
        ]

        for q in reverse_variations:
            rel = semantic_relationship_parser.parse(q, cols)
            assert rel is not None, f"Failed to parse reverse relationship for: '{q}'"
            assert rel.subject_val == "Kolkata", f"Expected Kolkata, got '{rel.subject_val}' for '{q}'"
            assert rel.subject_col == "capital", f"Expected capital column, got '{rel.subject_col}' for '{q}'"
            assert rel.target_col == "state", f"Expected state target, got '{rel.target_col}' for '{q}'"
            assert rel.direction == "REVERSE"

    def test_query_planner_structured_query(self):
        """Verify query planner produces clean LOOKUP StructuredQuery."""
        dataset_loader.load_sample("states")
        cols = dataset_loader.get_columns()

        sq = query_planner.plan_query("What is West Bengal's capital?", cols)
        assert sq.operation == "LOOKUP"
        assert sq.target_column == "capital"
        assert len(sq.conditions) == 1
        assert sq.conditions[0].column == "state"
        assert sq.conditions[0].value == "West Bengal"

        # Reverse query
        sq_rev = query_planner.plan_query("Which state has Kolkata as capital?", cols)
        assert sq_rev.operation == "LOOKUP"
        assert sq_rev.target_column == "state"
        assert len(sq_rev.conditions) == 1
        assert sq_rev.conditions[0].column == "capital"
        assert sq_rev.conditions[0].value == "Kolkata"

    def test_end_to_end_west_bengal_variations(self):
        """End-to-end execution of all 10+ prompt variations for West Bengal capital."""
        dataset_loader.load_sample("states")

        queries = [
            "What is West Bengal's capital?",
            "West Bengal capital?",
            "capital of West Bengal",
            "Which city is West Bengal's capital?",
            "West Bengal has which capital?",
            "Tell me West Bengal capital",
            "What about West Bengal's capital?",
        ]

        for idx, q in enumerate(queries):
            res = process_query(QueryRequest(question=q, session_id=f"test_wb_{idx}"))
            assert res.operation == "LOOKUP", f"Expected LOOKUP operation, got {res.operation} for '{q}'"
            assert res.source_type == "DATASET"
            assert len(res.results) == 1
            assert res.results[0]["capital"] == "Kolkata"
            assert "Kolkata" in res.answer, f"Expected 'Kolkata' in answer for '{q}', got: '{res.answer}'"

    def test_phonetic_and_typo_handling(self):
        """Test typo and phonetic variations resolve to correct states and capitals."""
        dataset_loader.load_sample("states")

        res_typo = process_query(QueryRequest(question="West Bengel's capital"))
        assert "Kolkata" in res_typo.answer
        assert res_typo.results[0]["state"] == "West Bengal"

        res_phonetic = process_query(QueryRequest(question="Cikkim's capital"))
        assert "Gangtok" in res_phonetic.answer
        assert res_phonetic.results[0]["state"] == "Sikkim"

    def test_other_indian_states(self):
        """Test across multiple Indian states."""
        dataset_loader.load_sample("states")

        states_test = [
            ("Telangana's capital", "Hyderabad"),
            ("Karnataka's capital", "Bengaluru"),
            ("What is Maharashtra's capital?", "Mumbai"),
            ("capital of Bihar", "Patna"),
            ("Goa's capital", "Panaji"),
        ]

        for q, expected_cap in states_test:
            res = process_query(QueryRequest(question=q))
            assert expected_cap in res.answer, f"Expected '{expected_cap}' for '{q}', got: '{res.answer}'"

    def test_reverse_lookup_end_to_end(self):
        """Test reverse lookup execution (Which state has Kolkata as capital?)."""
        dataset_loader.load_sample("states")

        queries = [
            "Which state has Kolkata as capital?",
            "Which state has Kolkata as its capital?",
            "Kolkata is the capital of which state?",
        ]

        for idx, q in enumerate(queries):
            res = process_query(QueryRequest(question=q, session_id=f"test_rev_{idx}"))
            assert res.operation == "LOOKUP"
            assert len(res.results) == 1
            assert res.results[0]["state"] == "West Bengal"
            assert "West Bengal" in res.answer

    def test_cross_domain_employee_dataset(self):
        """Test possessives and attributes on employee directory dataset."""
        ok, _ = dataset_loader.load_sample("employees")
        if not ok:
            pytest.skip("employees sample was removed per user dataset cleanup")

        try:
            # Salary lookup
            res_sal = process_query(QueryRequest(question="Rahul Dravid's salary", session_id="emp_sal"))
            assert res_sal.operation == "LOOKUP"
            assert len(res_sal.results) == 1
            assert res_sal.results[0]["Employee Name"] == "Rahul Dravid"
            assert res_sal.results[0]["Salary"] == 600000
            assert "600,000" in res_sal.answer

            # Department lookup
            res_dept = process_query(QueryRequest(question="Rahul Dravid's department", session_id="emp_dept"))
            assert res_dept.operation == "LOOKUP"
            assert res_dept.results[0]["Department"] == "Finance"
            assert "Finance" in res_dept.answer

            # Designation lookup
            res_desig = process_query(QueryRequest(question="What is Rahul Dravid's designation?", session_id="emp_desig"))
            assert res_desig.operation == "LOOKUP"
            assert res_desig.results[0]["Designation"] == "Financial Analyst"
            assert "Financial Analyst" in res_desig.answer
        finally:
            dataset_loader.load_sample("states")

    def test_cross_domain_product_dataset(self):
        """Test possessives on e-commerce products dataset."""
        ok, _ = dataset_loader.load_sample("products")
        if not ok:
            pytest.skip("products sample was removed per user dataset cleanup")

        res_prod = process_query(QueryRequest(question="Dell XPS 13's price", session_id="prod_price"))
        assert res_prod.operation == "LOOKUP"
        assert len(res_prod.results) == 1
        assert res_prod.results[0]["Product Name"] == "Dell XPS 13"
        assert "91,667" in res_prod.answer

    def test_general_knowledge_fallback_when_employees_loaded(self):
        """Verify that when employee dataset is loaded, geography questions route to General Knowledge."""
        ok, _ = dataset_loader.load_sample("employees")
        if not ok:
            pytest.skip("employees sample was removed per user dataset cleanup")

        res = process_query(QueryRequest(question="What is West Bengal's capital?", session_id="gk_wb_cap"))
        assert res.operation == "GENERAL_KNOWLEDGE"
        assert res.source_type == "GENERAL_KNOWLEDGE"
        assert "Kolkata" in res.answer

        res_rev = process_query(QueryRequest(question="Which state has Kolkata as capital?", session_id="gk_rev_cap"))
        assert res_rev.operation == "GENERAL_KNOWLEDGE"
        assert res_rev.source_type == "GENERAL_KNOWLEDGE"
        assert "West Bengal" in res_rev.answer
