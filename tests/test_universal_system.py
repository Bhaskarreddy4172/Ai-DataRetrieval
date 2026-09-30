"""Comprehensive test suite for Universal Customer Question Understanding and Training System."""

import pandas as pd
import pytest
from app.dataset.semantic_mapper import semantic_column_mapper, UNIVERSAL_ONTOLOGY
from app.retrieval.hybrid_search import hybrid_search
from app.query.analytics import dataset_analytics
from app.query.dsl import SafeQueryDSL, DSLFilter
from app.training.augmenter import linguistic_augmenter
from app.training.generator import training_generator
from app.training.datasets import dataset_manager
from app.training.feedback import feedback_manager


def test_universal_ontology_coverage():
    """Verify core business concepts exist in ontology."""
    core_concepts = [
        "PERSON", "EMPLOYEE", "CUSTOMER", "PRODUCT", "LOCATION", "CITY",
        "DEPARTMENT", "PRICE", "COST", "SALARY", "REVENUE", "QUANTITY",
        "DATE", "SCORE", "ID", "NAME"
    ]
    for c in core_concepts:
        assert c in UNIVERSAL_ONTOLOGY


def test_dynamic_column_semantics_discovery():
    """Verify dynamic semantic discovery on diverse column names."""
    res_salary = semantic_column_mapper.discover_column_semantics("Annual Compensation", [1500000, 2000000])
    assert "salary" in res_salary["semantic_concepts"] or "cost" in res_salary["semantic_concepts"]
    assert res_salary["confidence"] >= 0.85

    res_dept = semantic_column_mapper.discover_column_semantics("Team Division", ["Sales", "Engineering"])
    assert "department" in res_dept["semantic_concepts"] or "team" in res_dept["semantic_concepts"]


def test_hybrid_search_priority():
    """Test 6-tier search priority."""
    columns = ["Employee ID", "Employee Name", "Department", "Salary", "City"]

    # Tier 1: Exact
    res1 = hybrid_search.resolve_column("Salary", columns)
    assert res1["column"] == "Salary"
    assert res1["tier"] == 1

    # Tier 2: Normalized
    res2 = hybrid_search.resolve_column("salary", columns)
    assert res2["column"] == "Salary"
    assert res2["tier"] in {1, 2}

    # Tier 3: Alias / Universal Ontology
    res3 = hybrid_search.resolve_column("paycheck", columns)
    assert res3["column"] == "Salary"
    assert res3["tier"] == 3


def test_statistical_analytics_and_iqr_outliers():
    """Test deterministic statistics and IQR outlier detection."""
    data = {
        "Employee Name": ["A", "B", "C", "D", "E", "F", "G", "H", "Outlier"],
        "Salary": [50000, 52000, 48000, 51000, 53000, 49000, 50000, 52000, 500000]
    }
    df = pd.DataFrame(data)

    salary_series = df["Salary"]
    assert isinstance(salary_series, pd.Series)
    stats = dataset_analytics.calculate_statistics(salary_series, "Salary")
    assert stats["count"] == 9
    assert stats["min"] == 48000.0
    assert stats["max"] == 500000.0
    assert stats["median"] == 51000.0

    outliers = dataset_analytics.detect_outliers_iqr(df, "Salary")
    assert outliers["outlier_count"] == 1
    assert outliers["records"][0]["Employee Name"] == "Outlier"


def test_safe_query_dsl():
    """Test SafeQueryDSL serialization and AST."""
    dsl = SafeQueryDSL(
        operation="FILTER",
        filters=[DSLFilter(column="City", operator="=", value="Hyderabad")],
        limit=10
    )
    dsl_dict = dsl.to_dict()
    assert dsl_dict["operation"] == "FILTER"
    assert len(dsl_dict["filters"]) == 1
    assert dsl_dict["filters"][0]["column"] == "City"


def test_linguistic_augmentation():
    """Test slang, typo, and filler augmentations."""
    seed = "Who has the highest salary?"
    variations = linguistic_augmenter.augment(seed)
    assert len(variations) >= 3
    assert seed in variations


def test_training_generator_and_splits():
    """Test training generator produces pairs and splits them cleanly."""
    df = pd.DataFrame({
        "Product": ["Phone", "Laptop", "Watch"],
        "Price": [800, 1500, 300],
        "Category": ["Electronics", "Electronics", "Accessories"]
    })
    samples = training_generator.generate_from_dataframe(df, dataset_name="products", max_samples=20)
    assert len(samples) > 0
    assert "question" in samples[0]
    assert "intent" in samples[0]

    train, val, test = dataset_manager.create_splits(samples, train_ratio=0.6, val_ratio=0.2)
    assert len(train) + len(val) + len(test) == len(samples)


def test_active_learning_feedback():
    """Test feedback recording and queue retrieval."""
    entry = feedback_manager.record_feedback(
        question="yo who makes most cash in hyd?",
        is_correct=False,
        system_operation="FILTER",
        user_correction={"correct_operation": "MAX", "column": "Salary", "filter": "City=Hyderabad"}
    )
    assert entry["status"] == "reviewed"
    stats = feedback_manager.get_stats()
    assert stats["total_feedback_count"] >= 1
