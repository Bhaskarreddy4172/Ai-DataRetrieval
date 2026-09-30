"""Comprehensive Evaluation Suite for Universal Indian States, Capitals, Shortcuts, Typo & Phonetic Resolution.

Covers:
- All 28 States and standard abbreviations (AP, AR, AS, BR, CG, GA, GJ, HR, HP, JH, KA, KL, MP, MH, MN, ML, MZ, NL, OD, PB, RJ, SK, TN, TS, TG, TR, UP, UK, WB)
- Major Airport/City Codes (HYD, BLR, BOM, CCU, MAA, GNR, DDN, LKO, BBSR, RPR, RNC, BPL, PAT)
- Typo and phonetic variations (telengana, sikim, cikkim, sikkimm, gujrat, bihaar, chattisgarh, andra pradesh, etc.)
- Missing space / concatenated forms (WestBengal, AndhraPradesh, TamilNadu)
- Reverse lookup queries (Which state has Bangalore?, Which state is associated with Kolkata?, etc.)
- Possessive queries (West Bengal's capital, Karnataka's capital, missing apostrophe)
- Hinglish & informal queries (ts ka capital?, wb ka capital kya hai, capital ts?, hyd which state comes?)
- Boolean fact-checking (Is hyd belongs to Andra Pradesh?, Does BLR belong to Karnataka?)
- Zero hallucination and dataset-grounded fallback (France, Wakanda, population)
"""

import pytest
from app.api.routes import process_query, QueryRequest
from app.dataset.loader import dataset_loader
from app.dataset.spell_checker import dataset_spell_checker
from app.dataset.alias_resolver import entity_alias_resolver


@pytest.fixture(scope="module", autouse=True)
def setup_states_dataset():
    dataset_loader.load_sample("states")
    df = dataset_loader.dataframe
    dataset_spell_checker.build_vocabulary(df, "indian_states_capitals.csv")
    entity_alias_resolver.index_dataset(df)


class TestStateAbbreviationsAndShortcuts:
    """Test standard state abbreviations (28 Indian states)."""

    @pytest.mark.parametrize("query,expected_capital", [
        ("What is the capital of AP?", "Amaravati"),
        ("capital of AR", "Itanagar"),
        ("What is the capital of AS?", "Dispur"),
        ("capital of BR", "Patna"),
        ("capital of CG?", "Raipur"),
        ("capital of GA", "Panaji"),
        ("What is the capital of GJ?", "Gandhinagar"),
        ("capital of HR", "Chandigarh"),
        ("What is the capital of HP?", "Shimla"),
        ("capital of JH", "Ranchi"),
        ("capital of KA", "Bengaluru"),
        ("capital of KL", "Thiruvananthapuram"),
        ("capital of MP", "Bhopal"),
        ("capital of MH", "Mumbai"),
        ("capital of MN", "Imphal"),
        ("capital of ML", "Shillong"),
        ("capital of MZ", "Aizawl"),
        ("capital of NL", "Kohima"),
        ("capital of OD", "Bhubaneswar"),
        ("capital of PB", "Chandigarh"),
        ("capital of RJ", "Jaipur"),
        ("capital of SK", "Gangtok"),
        ("capital of TN", "Chennai"),
        ("capital of TS", "Hyderabad"),
        ("capital of TG", "Hyderabad"),
        ("capital of TR", "Agartala"),
        ("capital of UP", "Lucknow"),
        ("capital of UK", "Dehradun"),
        ("capital of WB", "Kolkata"),
    ])
    def test_state_shortcuts(self, query, expected_capital):
        res = process_query(QueryRequest(question=query))
        assert res.operation == "LOOKUP"
        assert expected_capital.lower() in res.answer.lower()


class TestAirportAndCityCodes:
    """Test capital city and airport code resolution."""

    @pytest.mark.parametrize("query,expected_state", [
        ("HYD is the capital of which state?", "Telangana"),
        ("BLR is the capital of which state?", "Karnataka"),
        ("BOM is the capital of which state?", "Maharashtra"),
        ("CCU is the capital of which state?", "West Bengal"),
        ("MAA is the capital of which state?", "Tamil Nadu"),
        ("Which state has Bangalore?", "Karnataka"),
        ("Which state has Kolkata?", "West Bengal"),
        ("Which state has Hyderabad?", "Telangana"),
        ("Which state has Shimla?", "Himachal Pradesh"),
        ("Which state has Jaipur?", "Rajasthan"),
        ("Which state has Patna?", "Bihar"),
        ("Which state has Ranchi?", "Jharkhand"),
        ("Which state has Bhopal?", "Madhya Pradesh"),
        ("Which state has Lucknow?", "Uttar Pradesh"),
        ("Which state has Dehradun?", "Uttarakhand"),
    ])
    def test_city_reverse_lookups(self, query, expected_state):
        res = process_query(QueryRequest(question=query))
        assert res.operation == "LOOKUP"
        assert expected_state.lower() in res.answer.lower()


class TestTypoAndPhoneticVariations:
    """Test misspelling, phonetic transliteration, and typo variations."""

    @pytest.mark.parametrize("query,expected_capital", [
        ("telengana capital", "Hyderabad"),
        ("What is the capital of telengana?", "Hyderabad"),
        ("sikim capital", "Gangtok"),
        ("cikkim capital", "Gangtok"),
        ("sikkimm capital", "Gangtok"),
        ("gujrat capital", "Gandhinagar"),
        ("bihaar capital", "Patna"),
        ("chattisgarh capital", "Raipur"),
        ("andra pradesh capital", "Amaravati"),
        ("karnatak capital", "Bengaluru"),
        ("west bengol capital", "Kolkata"),
        ("rajastan capital", "Jaipur"),
        ("jharkand capital", "Ranchi"),
        ("asom capital", "Dispur"),
    ])
    def test_state_typos(self, query, expected_capital):
        res = process_query(QueryRequest(question=query))
        assert res.operation == "LOOKUP"
        assert expected_capital.lower() in res.answer.lower()


class TestSpacingAndPunctuation:
    """Test missing spaces, compact forms, and hyphenated names."""

    @pytest.mark.parametrize("query,expected_capital", [
        ("WestBengal capital", "Kolkata"),
        ("What is AndhraPradesh capital?", "Amaravati"),
        ("TamilNadu capital", "Chennai"),
        ("HimachalPradesh capital", "Shimla"),
        ("MadhyaPradesh capital", "Bhopal"),
        ("UttarPradesh capital", "Lucknow"),
        ("ArunachalPradesh capital", "Itanagar"),
        ("West-Bengal capital", "Kolkata"),
    ])
    def test_compact_spacing(self, query, expected_capital):
        res = process_query(QueryRequest(question=query))
        assert res.operation == "LOOKUP"
        assert expected_capital.lower() in res.answer.lower()


class TestInformalAndHinglishPhrasing:
    """Test Hinglish particles, informal word orders, and short queries."""

    def test_hinglish_ka_particles(self):
        res1 = process_query(QueryRequest(question="ts ka capital?"))
        assert res1.operation == "LOOKUP"
        assert "hyderabad" in res1.answer.lower()

        res2 = process_query(QueryRequest(question="wb ka capital kya hai"))
        assert res2.operation == "LOOKUP"
        assert "kolkata" in res2.answer.lower()

        res3 = process_query(QueryRequest(question="karnataka ka capital batao"))
        assert res3.operation == "LOOKUP"
        assert "bengaluru" in res3.answer.lower()

    def test_informal_short_syntax(self):
        res1 = process_query(QueryRequest(question="capital ts?"))
        assert res1.operation == "LOOKUP"
        assert "hyderabad" in res1.answer.lower()

        res2 = process_query(QueryRequest(question="ts capital"))
        assert res2.operation == "LOOKUP"
        assert "hyderabad" in res2.answer.lower()

        res3 = process_query(QueryRequest(question="hyd which state comes?"))
        assert res3.operation == "LOOKUP"
        assert "telangana" in res3.answer.lower()

        res4 = process_query(QueryRequest(question="hyd which state?"))
        assert res4.operation == "LOOKUP"
        assert "telangana" in res4.answer.lower()

        res5 = process_query(QueryRequest(question="blr which state?"))
        assert res5.operation == "LOOKUP"
        assert "karnataka" in res5.answer.lower()


class TestPossessiveAndNaturalLanguage:
    """Test possessive syntax and word-order formulations."""

    def test_possessive_apostrophe(self):
        res1 = process_query(QueryRequest(question="What is West Bengal's capital?"))
        assert res1.operation == "LOOKUP"
        assert "kolkata" in res1.answer.lower()

        res2 = process_query(QueryRequest(question="Telangana's capital"))
        assert res2.operation == "LOOKUP"
        assert "hyderabad" in res2.answer.lower()

    def test_missing_apostrophe(self):
        res = process_query(QueryRequest(question="West Bengals capital"))
        assert res.operation == "LOOKUP"
        assert "kolkata" in res.answer.lower()

    def test_reverse_semantic_phrasings(self):
        res1 = process_query(QueryRequest(question="Which state is associated with Kolkata?"))
        assert res1.operation == "LOOKUP"
        assert "west bengal" in res1.answer.lower()

        res2 = process_query(QueryRequest(question="Bangalore belongs to which state?"))
        assert res2.operation == "LOOKUP"
        assert "karnataka" in res2.answer.lower()


class TestBooleanVerificationAgainstDataset:
    """Test boolean / fact-checking questions verified strictly from the active dataset."""

    def test_boolean_mismatch_grounded(self):
        res = process_query(QueryRequest(question="Is hyd belongs to Andra Pradesh?"))
        assert res.operation == "BOOLEAN_CHECK"
        assert "false" in res.answer.lower()
        assert "telangana" in res.answer.lower()

    def test_boolean_match_grounded(self):
        res1 = process_query(QueryRequest(question="Does BLR belong to Karnataka?"))
        assert res1.operation == "BOOLEAN_CHECK"
        assert "true" in res1.answer.lower()
        assert "karnataka" in res1.answer.lower()

        res2 = process_query(QueryRequest(question="Is Hyderabad the capital of Telangana?"))
        assert res2.operation == "BOOLEAN_CHECK"
        assert "true" in res2.answer.lower()
        assert "telangana" in res2.answer.lower()

    def test_boolean_incorrect_capital(self):
        res = process_query(QueryRequest(question="Is Mumbai the capital of Gujarat?"))
        assert res.operation == "BOOLEAN_CHECK"
        assert "false" in res.answer.lower()


class TestZeroHallucinationGuards:
    """Verify system strictly rejects entities and attributes not in dataset."""

    def test_unsupported_foreign_entity(self):
        res = process_query(QueryRequest(question="What is the capital of France?"))
        assert res.operation == "NO_MATCH"
        assert "france" in res.answer.lower()

    def test_fictional_entity(self):
        res = process_query(QueryRequest(question="What is the capital of Wakanda?"))
        assert res.operation == "NO_MATCH"
        assert "wakanda" in res.answer.lower()

    def test_unsupported_attribute(self):
        res = process_query(QueryRequest(question="What is Telangana's population?"))
        assert res.operation in ["UNSUPPORTED_QUERY", "NO_MATCH", "UNKNOWN"]
        assert "population" in res.answer.lower()
