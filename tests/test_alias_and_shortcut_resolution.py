"""Unit and integration tests for universal Entity Alias & Abbreviation Resolution Engine."""

import pytest
import pandas as pd
from app.dataset.alias_resolver import entity_alias_resolver
from app.dataset.indexer import dataset_indexer
from app.dataset.value_resolver import dataset_value_resolver
from app.query.semantic_parser import semantic_relationship_parser
from app.knowledge.general_knowledge import general_knowledge_engine
from app.router.question_router import QuestionRouter


@pytest.fixture
def router():
    return QuestionRouter()


@pytest.fixture
def sample_employee_df():
    return pd.DataFrame({
        "Employee ID": ["EMP-1001", "EMP-1002", "EMP-1003", "EMP-1004"],
        "Name": ["Rahul Patel", "Ananya Sharma", "Arjun Reddy", "Priya Nair"],
        "Department": ["Human Resources", "Information Technology", "Engineering", "Quality Assurance"],
        "City": ["Hyderabad", "Bengaluru", "Mumbai", "Kolkata"],
        "Salary": [85000, 95000, 110000, 75000],
    })


@pytest.fixture
def states_capitals_df():
    return pd.DataFrame({
        "State": ["Telangana", "Andhra Pradesh", "West Bengal", "Karnataka", "Maharashtra"],
        "Capital": ["Hyderabad", "Amaravati", "Kolkata", "Bengaluru", "Mumbai"],
        "Official Language": ["Telugu", "Telugu", "Bengali", "Kannada", "Marathi"]
    })


class TestEntityAliasResolverDirect:
    """Test direct alias resolution across categories."""

    def test_state_codes(self):
        cases = [
            ("AP", "Andhra Pradesh"),
            ("TG", "Telangana"),
            ("TS", "Telangana"),
            ("WB", "West Bengal"),
            ("MH", "Maharashtra"),
            ("DL", "Delhi"),
            ("HR", "Haryana"),
            ("KA", "Karnataka"),
            ("GJ", "Gujarat"),
            ("TN", "Tamil Nadu"),
            ("UP", "Uttar Pradesh"),
        ]
        for code, expected_name in cases:
            res = entity_alias_resolver.resolve_alias(code, expected_type="STATE")
            assert res is not None, f"Failed to resolve {code}"
            assert res["canonical_name"] == expected_name
            assert res["confidence"] >= 0.95

    def test_city_airport_codes(self):
        cases = [
            ("HYD", "Hyderabad"),
            ("BLR", "Bengaluru"),
            ("BOM", "Mumbai"),
            ("CCU", "Kolkata"),
            ("MAA", "Chennai"),
            ("DEL", "Delhi"),
            ("PNQ", "Pune"),
            ("AMD", "Ahmedabad"),
            ("JAI", "Jaipur"),
            ("LKO", "Lucknow"),
        ]
        for code, expected_name in cases:
            res = entity_alias_resolver.resolve_alias(code, expected_type="CITY")
            assert res is not None, f"Failed to resolve {code}"
            assert res["canonical_name"] == expected_name
            assert res["confidence"] >= 0.95

    def test_historical_and_alternate_names(self):
        cases = [
            ("Bangalore", "Bengaluru"),
            ("Bombay", "Mumbai"),
            ("Madras", "Chennai"),
            ("Calcutta", "Kolkata"),
            ("Poona", "Pune"),
            ("Vizag", "Visakhapatnam"),
            ("Baroda", "Vadodara"),
            ("Pondicherry", "Puducherry"),
        ]
        for name, expected_name in cases:
            res = entity_alias_resolver.resolve_alias(name)
            assert res is not None, f"Failed to resolve {name}"
            assert res["canonical_name"] == expected_name

    def test_department_codes(self):
        cases = [
            ("HR", "Human Resources"),
            ("IT", "Information Technology"),
            ("QA", "Quality Assurance"),
            ("R&D", "Research & Development"),
            ("ENG", "Engineering"),
            ("MKTG", "Marketing"),
            ("FIN", "Finance"),
        ]
        for code, expected_name in cases:
            res = entity_alias_resolver.resolve_alias(code, expected_type="DEPARTMENT")
            assert res is not None, f"Failed to resolve {code}"
            assert res["canonical_name"] == expected_name

    def test_country_codes(self):
        cases = [
            ("USA", "United States"),
            ("UK", "United Kingdom"),
            ("UAE", "United Arab Emirates"),
            ("USD", "US Dollar"),
            ("INR", "Indian Rupee"),
        ]
        for code, expected_name in cases:
            res = entity_alias_resolver.resolve_alias(code)
            assert res is not None, f"Failed to resolve {code}"
            assert res["canonical_name"] == expected_name

    def test_capital_and_state_reverse_lookups(self):
        # State -> Capital
        tg_cap = entity_alias_resolver.resolve_capital_of("TG")
        assert tg_cap is not None
        assert tg_cap["capital"] == "Hyderabad"

        ap_cap = entity_alias_resolver.resolve_capital_of("AP")
        assert ap_cap is not None
        assert ap_cap["capital"] == "Amaravati"

        wb_cap = entity_alias_resolver.resolve_capital_of("WB")
        assert wb_cap is not None
        assert wb_cap["capital"] == "Kolkata"

        # City -> State
        hyd_st = entity_alias_resolver.resolve_state_of("HYD")
        assert hyd_st is not None
        assert hyd_st["state"] == "Telangana"

        blr_st = entity_alias_resolver.resolve_state_of("BLR")
        assert blr_st is not None
        assert blr_st["state"] == "Karnataka"

        bom_st = entity_alias_resolver.resolve_state_of("BOM")
        assert bom_st is not None
        assert bom_st["state"] == "Maharashtra"


class TestDynamicDatasetAcronyms:
    """Test dynamic acronym generation from uploaded datasets."""

    def test_dynamic_indexing(self, sample_employee_df):
        entity_alias_resolver.index_dataset(sample_employee_df)
        assert "hr" in entity_alias_resolver.dynamic_acronyms
        assert "it" in entity_alias_resolver.dynamic_acronyms
        assert "qa" in entity_alias_resolver.dynamic_acronyms

        # Value resolver using dynamic acronyms
        res_hr = dataset_value_resolver.resolve_value_in_column("HR", "Department", sample_employee_df["Department"].tolist())
        assert res_hr["resolved_value"] == "Human Resources"

        res_it = dataset_value_resolver.resolve_value_in_column("IT", "Department", sample_employee_df["Department"].tolist())
        assert res_it["resolved_value"] == "Information Technology"


class TestDatasetValueResolverWithShortcuts:
    """Test resolving shortcuts against specific columns."""

    def test_resolve_city_airport_codes(self, sample_employee_df):
        dataset_indexer.build_index(sample_employee_df, "employees")
        cities = sample_employee_df["City"].tolist()

        res_hyd = dataset_value_resolver.resolve_value_in_column("HYD", "City", cities)
        assert res_hyd["resolved_value"] == "Hyderabad"

        res_blr = dataset_value_resolver.resolve_value_in_column("BLR", "City", cities)
        assert res_blr["resolved_value"] == "Bengaluru"

        res_bom = dataset_value_resolver.resolve_value_in_column("BOM", "City", cities)
        assert res_bom["resolved_value"] == "Mumbai"

        res_ccu = dataset_value_resolver.resolve_value_in_column("CCU", "City", cities)
        assert res_ccu["resolved_value"] == "Kolkata"

    def test_resolve_state_codes_in_states_dataset(self, states_capitals_df):
        dataset_indexer.build_index(states_capitals_df, "states")
        states = states_capitals_df["State"].tolist()

        res_tg = dataset_value_resolver.resolve_value_in_column("TG", "State", states)
        assert res_tg["resolved_value"] == "Telangana"

        res_ap = dataset_value_resolver.resolve_value_in_column("AP", "State", states)
        assert res_ap["resolved_value"] == "Andhra Pradesh"

        res_wb = dataset_value_resolver.resolve_value_in_column("WB", "State", states)
        assert res_wb["resolved_value"] == "West Bengal"


class TestSemanticParserWithShortcuts:
    """Test parsing relationships involving shortcuts."""

    def test_possessive_and_noun_adjunct_shortcuts(self, states_capitals_df):
        dataset_indexer.build_index(states_capitals_df, "states")
        cols = list(states_capitals_df.columns)

        # "TG's capital"
        rel1 = semantic_relationship_parser.parse("TG's capital", cols)
        assert rel1 is not None
        assert rel1.subject_val == "Telangana"
        assert rel1.target_col == "Capital"

        # "AP capital?"
        rel2 = semantic_relationship_parser.parse("AP capital?", cols)
        assert rel2 is not None
        assert rel2.subject_val == "Andhra Pradesh"
        assert rel2.target_col == "Capital"

        # "WB's capital"
        rel3 = semantic_relationship_parser.parse("WB's capital", cols)
        assert rel3 is not None
        assert rel3.subject_val == "West Bengal"
        assert rel3.target_col == "Capital"


class TestGeneralKnowledgeShortcuts:
    """Test General Knowledge Q&A on shortcuts when dataset doesn't have the data."""

    def test_gk_capital_shortcuts(self):
        # "TG's capital"
        ans_tg = general_knowledge_engine.answer_question("TG's capital")
        assert ans_tg["answered"] is True
        assert "Hyderabad" in ans_tg["answer"]

        # "AP capital?"
        ans_ap = general_knowledge_engine.answer_question("AP capital?")
        assert ans_ap["answered"] is True
        assert "Amaravati" in ans_ap["answer"]

        # "What is WB's capital?"
        ans_wb = general_knowledge_engine.answer_question("What is WB's capital?")
        assert ans_wb["answered"] is True
        assert "Kolkata" in ans_wb["answer"]

    def test_gk_city_state_shortcuts(self):
        # "Which state is HYD in?"
        ans_hyd = general_knowledge_engine.answer_question("Which state is HYD in?")
        assert ans_hyd["answered"] is True
        assert "Telangana" in ans_hyd["answer"]

        # "Which state does BLR belong to?"
        ans_blr = general_knowledge_engine.answer_question("Which state does BLR belong to?")
        assert ans_blr["answered"] is True
        assert "Karnataka" in ans_blr["answer"]

        # "BLR state?"
        ans_blr2 = general_knowledge_engine.answer_question("BLR state?")
        assert ans_blr2["answered"] is True
        assert "Karnataka" in ans_blr2["answer"]

    def test_gk_boolean_shortcuts(self):
        # "Is HYD in AP?" -> False
        ans_false = general_knowledge_engine.answer_question("Is HYD in AP?")
        assert ans_false["answered"] is True
        assert ans_false["result"] is False
        assert "Hyderabad is in Telangana, not Andhra Pradesh" in ans_false["answer"]

        # "Is HYD in TG?" -> True
        ans_true = general_knowledge_engine.answer_question("Is HYD in TG?")
        assert ans_true["answered"] is True
        assert ans_true["result"] is True
        assert "Hyderabad is in Telangana" in ans_true["answer"]
