"""Tests for phonetic matching (Soundex, Double Metaphone, Indian Phonetics, and Value Resolution)."""

import pytest
from app.utils.phonetic import (
    soundex,
    double_metaphone,
    normalize_indian_phonetics,
    phonetic_match_score,
    PhoneticIndex
)
from app.knowledge.general_knowledge import general_knowledge_engine
from app.dataset.value_resolver import dataset_value_resolver
from app.dataset.indexer import dataset_indexer
import pandas as pd


def test_soundex_basic():
    assert soundex("Hyderabad") == soundex("Haiderabad")
    assert soundex("Delhi") == soundex("Deli")
    assert len(soundex("Bengaluru")) == 4


def test_double_metaphone_basic():
    p1, _ = double_metaphone("Cikkim")
    p2, _ = double_metaphone("Sikkim")
    assert p1 == p2 == "SKM"

    kp1, _ = double_metaphone("Kalkata")
    kp2, _ = double_metaphone("Kolkata")
    assert kp1 == kp2 == "KLKT"


def test_normalize_indian_phonetics():
    assert normalize_indian_phonetics("Aandhra") == normalize_indian_phonetics("Andhra")
    assert normalize_indian_phonetics("Haiderabad") == normalize_indian_phonetics("Haydarabad")
    assert normalize_indian_phonetics("Cikkim") == normalize_indian_phonetics("Sikkim")


def test_phonetic_match_score():
    assert phonetic_match_score("Cikkim", "Sikkim") >= 0.85
    assert phonetic_match_score("Kalkata", "Kolkata") >= 0.85
    assert phonetic_match_score("Bengluru", "Bengaluru") >= 0.85
    assert phonetic_match_score("Haiderabad", "Hyderabad") >= 0.85


def test_phonetic_index():
    index = PhoneticIndex()
    index.add_term("Hyderabad")
    index.add_term("Bengaluru")
    index.add_term("Sikkim")

    cands = index.get_candidates("Cikkim")
    assert "Sikkim" in cands

    cands_hyd = index.get_candidates("Haiderabad")
    assert "Hyderabad" in cands_hyd


def test_general_knowledge_phonetic():
    assert general_knowledge_engine.find_state_match("Cikkim") == "Sikkim"
    assert general_knowledge_engine.find_city_match("Kalkata") == "Kolkata"
    assert general_knowledge_engine.find_city_match("Bengluru") == "Bengaluru"
    assert general_knowledge_engine.find_city_match("Haiderabad") == "Hyderabad"


def test_dataset_value_resolver_phonetic():
    values = ["Hyderabad", "Bengaluru", "Mumbai", "Kolkata", "Delhi", "Pune"]
    
    res1 = dataset_value_resolver.resolve_value_in_column("Kalkata", "City", available_values=values)
    assert res1["resolved_value"] == "Kolkata"

    res2 = dataset_value_resolver.resolve_value_in_column("Bengluru", "City", available_values=values)
    assert res2["resolved_value"] == "Bengaluru"

    res3 = dataset_value_resolver.resolve_value_in_column("Haiderabad", "City", available_values=values)
    assert res3["resolved_value"] == "Hyderabad"

