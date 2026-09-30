"""Tests for hallucination prevention, missing column detection, and out-of-scope queries."""

import pytest
from app.ai.response_generator import response_generator
from app.dataset.loader import DatasetLoader
from app.dataset.metadata import DatasetMetadata
from app.query.engine import query_engine
from app.query.parser import query_parser


def test_missing_column_detection():
    loader = DatasetLoader()
    cols = loader.get_columns()
    schema = DatasetMetadata.get_compact_schema(loader.profile["columns"], loader.dataset_name, len(loader.dataframe))

    # User asks for population on customer dataset (or states dataset without population)
    query = query_parser.parse("What is the population of Mumbai?", cols, schema)
    assert query.operation == "UNKNOWN"
    assert query.missing_column is not None


def test_anti_hallucination_response_generation():
    ans = response_generator.generate(
        question="What is the population of Mumbai?",
        operation="UNKNOWN",
        results=[],
        missing_column="population"
    )
    assert "does not contain" in ans
    assert "population" in ans


def test_general_knowledge_rejection():
    loader = DatasetLoader()
    cols = loader.get_columns()
    schema = DatasetMetadata.get_compact_schema(loader.profile["columns"], loader.dataset_name, len(loader.dataframe))

    query = query_parser.parse("Who is the Prime Minister of India?", cols, schema)
    assert query.operation == "UNKNOWN"

    ans = response_generator.generate(
        question="Who is the Prime Minister of India?",
        operation="UNKNOWN",
        results=[],
        missing_column="prime minister"
    )
    assert "does not contain" in ans


def test_empty_results_handling():
    ans = response_generator.generate(
        question="Show customers from NonExistentCity",
        operation="FILTER",
        results=[]
    )
    assert "No matching records were found" in ans
