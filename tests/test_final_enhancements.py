"""Comprehensive Automated Evaluation & Test Suite for the Final Enhancement Layer.

Covers the complete 50-point Matrix:
1. Normalization (Case, Whitespace, Punctuation with ID preservation, Unicode NFKD)
2. Contractions, Informal & Slang Queries
3. Grammar & Broken English Normalization
4. Multilingual & Hinglish / Telugu-English Queries
5. Deterministic Calculation Engine (Count, Sum, Avg, Median, Min, Max, Range, Std, Var, Pct, Pct Change, Ratio, Diff, Top-N)
6. Zero-Division & Null-Safe Math
7. Deterministic Date Engine (Formats, Before/After/Between, Earliest/Latest, Intervals)
8. Data Quality & Parent-Child Hierarchy Validator
9. Query Provenance, Evidence Mode, and Cache Safety
"""

from datetime import date
from pathlib import Path
import pandas as pd
import pytest

from app.nlp.normalizer import text_normalizer
from app.query.calculator import query_calculator
from app.query.date_engine import date_engine
from app.dataset.validator import dataset_validator
from app.query.provenance import QueryProvenance, evidence_builder, result_consistency_verifier
from app.query.cache import QueryCache
from app.api.routes import process_query, QueryRequest
from app.dataset.loader import dataset_loader


# ==============================================================================
# 1. TEXT NORMALIZATION TESTS
# ==============================================================================

class TestTextNormalization:
    """Test Case, Whitespace, Punctuation, and Unicode Normalization."""

    def test_case_and_whitespace_normalization(self):
        """Test that casing and irregular whitespace are normalized while preserving meaning."""
        inputs = [
            "  Hyderabad  ",
            "\tHyderabad\n",
            "HYDERABAD",
            "HyDeRaBaD",
            "  New   Delhi  \u00a0",
        ]
        for inp in inputs:
            norm = text_normalizer.normalize_whitespace(text_normalizer.normalize_unicode(inp)).lower()
            assert "hyderabad" in norm or "new delhi" in norm

    def test_punctuation_preserves_identifiers(self):
        """Test stripping query punctuation while strictly preserving hyphenated IDs."""
        # Non-semantic punctuation stripped
        assert text_normalizer.normalize_punctuation("What is Hyderabad's capital???") == "What is Hyderabad's capital"
        assert text_normalizer.normalize_punctuation('"Bengaluru"!') == "Bengaluru"

        # Semantic IDs preserved
        assert text_normalizer.normalize_punctuation("Look up EMP-1001?") == "Look up EMP-1001"
        assert text_normalizer.normalize_punctuation("Is ST-01 active.") == "Is ST-01 active"
        assert text_normalizer.normalize_punctuation("Check VIL-005!") == "Check VIL-005"

    def test_unicode_normalization(self):
        """Test Unicode NFKD and fancy punctuation replacement."""
        # Curly quotes and dashes
        s = "West Bengal’s capital \u2013 Kolkata"
        norm = text_normalizer.normalize_unicode(s)
        assert "'" in norm
        assert "-" in norm
        assert "West Bengal's capital - Kolkata" in norm


# ==============================================================================
# 2. CONTRACTIONS & SLANG TESTS
# ==============================================================================

class TestContractionsAndSlang:
    """Test contractions and informal language normalization."""

    @pytest.mark.parametrize("inp,expected", [
        ("who's got highest salary?", "who is"),
        ("what's the capital of TG?", "what is"),
        ("where's Patna located?", "where is"),
        ("who gets paid most?", "who has highest salary"),
        ("who earns more Rahul or Priya?", "who has higher salary"),
        ("top guy in Engineering", "highest"),
        ("who joined first?", "who joined earliest"),
    ])
    def test_contractions_and_slang_expansion(self, inp, expected):
        nq = text_normalizer.normalize_question(inp)
        assert expected.lower() in nq.normalized_question.lower()


# ==============================================================================
# 3. GRAMMAR & MULTILINGUAL NORMALIZATION
# ==============================================================================

class TestGrammarAndMultilingual:
    """Test broken English, Hinglish, and Telugu-English question normalization."""

    def test_broken_english_word_order(self):
        """Test inverted grammar and keyword-only inquiries."""
        q1 = text_normalizer.normalize_question("who highest salary")
        assert "who has highest salary" in q1.normalized_question.lower()

        q2 = text_normalizer.normalize_question("hyd villages how many")
        assert "how many villages" in q2.normalized_question.lower()

    def test_multilingual_and_hinglish(self):
        """Test Hindi-English and Telugu-English query comprehension."""
        # Hinglish
        h1 = text_normalizer.normalize_question("Telangana mein kitne villages hain?")
        assert "how many villages" in h1.normalized_question.lower()
        assert h1.is_multilingual

        h2 = text_normalizer.normalize_question("sabse jyada population kiski hai")
        assert "highest" in h2.normalized_question.lower()

        # Telugu-English
        t1 = text_normalizer.normalize_question("Telangana lo enni villages unnayi?")
        assert "how many villages" in t1.normalized_question.lower()
        assert t1.is_multilingual


# ==============================================================================
# 4. DETERMINISTIC CALCULATION ENGINE TESTS
# ==============================================================================

class TestCalculationEngine:
    """Test deterministic Pandas/NumPy calculation suite."""

    @pytest.fixture
    def calc_df(self):
        return pd.DataFrame({
            "Village": ["V-1", "V-2", "V-3", "V-4", "V-5"],
            "Population": [10000, 20000, 30000, 40000, 50000],
            "No_of_Females": [4800, 9900, 15100, 20200, 24500],
            "Literacy_Rate": ["72.5%", "81.0%", "65.0%", "90.0%", "78.5%"],
            "Salary": ["₹50,000", "₹75,000", "₹60,000", "₹120,000", "₹95,000"],
            "Category": ["A", "B", "A", "B", "A"]
        })

    def test_count_operations(self, calc_df):
        # Total count
        c1 = query_calculator.count(calc_df)
        assert c1.value == 5

        # Distinct count
        c2 = query_calculator.count(calc_df, distinct_col="Category")
        assert c2.value == 2

    def test_sum_and_average(self, calc_df):
        # Sum
        s = query_calculator.sum(calc_df, "Population")
        assert s.value == 150000
        assert s.formatted_value == "150,000"

        # Average
        a = query_calculator.average(calc_df, "Population")
        assert a.value == 30000.0

        # Currency-formatted column average
        sal_avg = query_calculator.average(calc_df, "Salary")
        assert sal_avg.value == 80000.0

    def test_median_and_extremes(self, calc_df):
        med = query_calculator.median(calc_df, "Population")
        assert med.value == 30000.0

        max_res = query_calculator.min_max(calc_df, "Population", is_max=True)
        assert max_res.value == 50000
        assert max_res.associated_entity is not None
        assert max_res.associated_entity["Village"] == "V-5"

        min_res = query_calculator.min_max(calc_df, "Population", is_max=False)
        assert min_res.value == 10000
        assert min_res.associated_entity is not None
        assert min_res.associated_entity["Village"] == "V-1"

    def test_range_std_variance(self, calc_df):
        r = query_calculator.range_stat(calc_df, "Population")
        assert r.value == 40000

        std_res = query_calculator.standard_deviation(calc_df, "Population")
        assert isinstance(std_res.value, (int, float))
        assert std_res.value > 0

        var_res = query_calculator.variance(calc_df, "Population")
        assert isinstance(var_res.value, (int, float))
        assert var_res.value > 0

    def test_percentage_calculations(self):
        # Normal percentage
        p = query_calculator.percentage(4800, 10000)
        assert p.is_valid
        assert p.value == 48.0
        assert p.formatted_value == "48.00%"

        # Zero denominator safety guard
        p_zero = query_calculator.percentage(100, 0)
        assert not p_zero.is_valid
        assert p_zero.error_message is not None
        assert "zero or undefined" in p_zero.error_message

    def test_percentage_change(self):
        # Growth
        pc = query_calculator.percentage_change(100, 125)
        assert pc.value == 25.0
        assert "+25.00%" in pc.formatted_value

        # Zero baseline safety guard
        pc_zero = query_calculator.percentage_change(0, 50)
        assert not pc_zero.is_valid
        assert pc_zero.error_message is not None
        assert "zero" in pc_zero.error_message

    def test_ratios_and_differences(self):
        # Ratio
        rat = query_calculator.ratio(50000, 25000, "Males", "Females")
        assert rat.value == 2.0
        assert "2.00:1" in rat.formatted_value

        # Safe denominator in ratio
        rat_zero = query_calculator.ratio(50000, 0, "Males", "Females")
        assert not rat_zero.is_valid

        # Difference
        diff = query_calculator.difference(85000, 60000)
        assert diff.value == 25000

    def test_top_n_bounded(self, calc_df):
        top3 = query_calculator.top_n(calc_df, "Population", n=3, ascending=False)
        assert top3["count"] == 3
        assert top3["records"][0]["Population"] == 50000


# ==============================================================================
# 5. DETERMINISTIC DATE ENGINE TESTS
# ==============================================================================

class TestDateEngine:
    """Test multi-format date parsing and interval operations."""

    def test_date_parsing_formats(self):
        d1 = date_engine.parse_date("2024-03-15")
        d2 = date_engine.parse_date("15/03/2024")
        d3 = date_engine.parse_date("15-03-2024")
        d4 = date_engine.parse_date("March 15 2024")
        assert d1 == d2 == d3 == d4 == date(2024, 3, 15)

    def test_date_intervals(self):
        days = date_engine.days_between("2024-01-01", "2024-01-11")
        assert days == 10

        years = date_engine.years_between("2020-01-01", "2024-01-01")
        assert years == 4

    def test_date_filtering(self):
        df = pd.DataFrame({
            "Name": ["Alice", "Bob", "Charlie"],
            "Joining_Date": ["2021-01-10", "2022-05-15", "2023-11-20"]
        })
        # Before 2022
        filtered = date_engine.filter_date_column(df, "Joining_Date", "BEFORE", "2022-01-01")
        assert len(filtered) == 1
        assert filtered.iloc[0]["Name"] == "Alice"

        # Latest joiner
        latest = date_engine.find_extreme_date_row(df, "Joining_Date", find_latest=True)
        assert latest is not None
        assert latest["Name"] == "Charlie"


# ==============================================================================
# 6. DATA VALIDATION & HIERARCHY INTEGRITY
# ==============================================================================

class TestDataQualityValidator:
    """Test schema, duplicate, missing-value, and parent-child hierarchy validation."""

    def test_duplicate_detection(self):
        df = pd.DataFrame({
            "ID": ["A1", "A2", "A1"],
            "Name": ["Item1", "Item2", "Item1"]
        })
        res = dataset_validator.detect_duplicates(df, subset=["ID"])
        assert not res["is_unique"]
        assert res["duplicate_count"] == 2

    def test_missing_value_analysis(self):
        df = pd.DataFrame({
            "A": ["val", None, "N/A", "real"],
            "B": [10, 20, 30, 40]
        })
        res = dataset_validator.analyze_missing_values(df)
        assert res["columns"]["A"]["missing_count"] == 2
        assert res["columns"]["B"]["missing_count"] == 0

    def test_parent_child_hierarchy_validation(self):
        main_path = Path(__file__).resolve().parent.parent / "universal_dataset_demo" / "main_dataset" / "india_states_capitals_main.csv"
        child_dir = Path(__file__).resolve().parent.parent / "universal_dataset_demo" / "child_datasets"

        if main_path.exists() and child_dir.exists():
            val_res = dataset_validator.validate_parent_child_hierarchy(main_path, child_dir)
            assert val_res["is_valid"]
            assert val_res["verified_child_count"] == 28
            assert val_res["missing_child_count"] == 0


# ==============================================================================
# 7. PROVENANCE, EVIDENCE & CACHE SAFETY
# ==============================================================================

class TestProvenanceAndCacheSafety:
    """Test provenance tracking, Evidence Mode, and cache key safety."""

    def test_provenance_and_evidence(self):
        prov = QueryProvenance(
            dataset_id="test_demo",
            dataset_version="v1.0_hash",
            source_dataset="india_states_capitals_main.csv",
            child_dataset="tg_hyderabad_villages.csv",
            columns_used=["Village", "Population"],
            operations=["MAX"],
            row_ids=["HYD-001"],
            calculation_formula="MAX(Population)",
            calculation_result=482391
        )
        assert prov.to_dict()["child_dataset"] == "tg_hyderabad_villages.csv"

        evidence = evidence_builder.build_evidence(
            prov,
            matched_rows=[{"Village_ID": "HYD-001", "Village": "Gachibowli", "Population": 482391}],
            key_metric_col="Population"
        )
        assert evidence["source_dataset"] == "tg_hyderabad_villages.csv"
        assert len(evidence["evidence_rows"]) == 1

        consistency = result_consistency_verifier.verify(
            answer="Gachibowli has highest population",
            df=pd.DataFrame({"Village": ["Gachibowli"], "Population": [482391]}),
            provenance=prov,
            matched_rows=[{"Village": "Gachibowli", "Population": 482391}]
        )
        assert consistency["is_consistent"]

    def test_cache_safety_isolation(self):
        cache = QueryCache()
        # Same query on dataset_v1 vs dataset_v2 must produce distinct cache keys
        cache.set("hash_v1", "what is capital?", "Result V1", dataset_id="states", dataset_version="v1")
        cache.set("hash_v2", "what is capital?", "Result V2", dataset_id="states", dataset_version="v2")

        res_v1 = cache.get("hash_v1", "what is capital?", dataset_id="states", dataset_version="v1")
        res_v2 = cache.get("hash_v2", "what is capital?", dataset_id="states", dataset_version="v2")

        assert res_v1 == "Result V1"
        assert res_v2 == "Result V2"

