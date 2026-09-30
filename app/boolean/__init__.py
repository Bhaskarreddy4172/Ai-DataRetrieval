"""Boolean and Fact-Checking Engine for deterministic verification across datasets and general knowledge."""

from app.boolean.schema import BooleanAssertion, BooleanResult
from app.boolean.assertion_builder import boolean_assertion_builder, BooleanAssertionBuilder
from app.boolean.verifier import boolean_verifier, BooleanVerifier

__all__ = [
    "BooleanAssertion",
    "BooleanResult",
    "boolean_assertion_builder",
    "BooleanAssertionBuilder",
    "boolean_verifier",
    "BooleanVerifier",
]

