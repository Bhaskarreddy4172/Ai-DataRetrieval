"""Tests for fuzzy matching, abbreviation expansion, indexing, and relationship engine."""

import pytest
import pandas as pd
from app.utils.fuzzy_match import (
    expand_abbreviations,
    fuzzy_match_column,
    fuzzy_match_value,
    extract_best_match_from_text,
)
from app.dataset.indexer import DatasetIndexer
from app.dataset.relationship_engine import RelationshipEngine


def test_abbreviation_expansion():
    assert "Hyderabad" in expand_abbreviations("show customers from hyd")
    assert "Bengaluru" in expand_abbreviations("who is in blr?")
    assert "Mumbai" in expand_abbreviations("how many in mum?")
    assert "salary" in expand_abbreviations("what is the sal?")
    assert "highest salary" in expand_abbreviations("who makes the most?")


def test_fuzzy_column_matching():
    cols = ["Customer Name", "Service Amount", "Vehicle", "City", "Warranty Expiry"]
    assert fuzzy_match_column("amount", cols) == "Service Amount"
    assert fuzzy_match_column("amt", cols) == "Service Amount"
    assert fuzzy_match_column("cust name", cols) == "Customer Name"
    assert fuzzy_match_column("city", cols) == "City"
    assert fuzzy_match_column("veh", cols) == "Vehicle"


def test_fuzzy_value_matching():
    cities = ["Hyderabad", "Bengaluru", "Mumbai", "Kolkata", "Delhi", "Chennai"]
    assert fuzzy_match_value("hyd", cities) == "Hyderabad"
    assert fuzzy_match_value("hydrabad", cities) == "Hyderabad"
    assert fuzzy_match_value("bangalore", cities) == "Bengaluru"
    assert fuzzy_match_value("bombay", cities) == "Mumbai"


def test_dataset_indexer():
    df = pd.DataFrame({
        "ID": ["P1", "P2", "P3"],
        "Product Name": ["Sony Headphones", "Apple MacBook Air", "Samsung Galaxy"],
        "Category": ["Audio", "Laptops", "Smartphones"],
        "Price": [12000, 85000, 65000]
    })
    indexer = DatasetIndexer()
    indexer.build_index(df, "test_products")

    assert indexer.find_matching_value_in_column("audio", "Category") == "Audio"
    assert indexer.find_matching_value_in_column("macbook", "Product Name") is not None
    locs = indexer.search_token_across_columns("sony")
    assert len(locs) > 0
    assert locs[0][1] == "Product Name"


def test_relationship_engine():
    df = pd.DataFrame({
        "Employee ID": ["EMP-1", "EMP-2", "EMP-3"],
        "Employee Name": ["Aarav Sharma", "Neha Patel", "Rohan Verma"],
        "Department": ["Engineering", "Engineering", "Sales"],
        "Salary": [800000, 1200000, 600000]
    })
    engine = RelationshipEngine()
    analysis = engine.analyze(df)

    assert "Employee ID" in analysis["candidate_keys"] or "Employee Name" in analysis["candidate_keys"]
    assert any(d["column"] == "Department" for d in analysis["dimensions"])
    assert any(m["column"] == "Salary" for m in analysis["metrics"])

    comp = engine.compare_entities(df, "Employee Name", "Aarav Sharma", "Neha Patel")
    assert comp["status"] == "success"
    assert len(comp["comparison"]) == len(df.columns)

