"""Decoupled Sentiment NLP Module.
Classifies query sentiment into POSITIVE, NEGATIVE, NEUTRAL, MIXED, or UNKNOWN.
Completely independent from dataset retrieval and execution.
"""

import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SentimentResult(BaseModel):
    sentiment: str = Field(..., description="POSITIVE | NEGATIVE | NEUTRAL | MIXED | UNKNOWN")
    score: float = Field(0.0, description="Normalized sentiment score from -1.0 to 1.0")
    positive_cues: List[str] = Field(default_factory=list)
    negative_cues: List[str] = Field(default_factory=list)


class SentimentAnalyzer:
    """Lightweight, fast, rule-based sentiment analyzer that does not distort query execution."""

    POSITIVE_WORDS = {
        "good", "great", "excellent", "best", "wonderful", "amazing", "happy",
        "pleased", "love", "awesome", "perfect", "superb", "brilliant",
        "fabulous", "thanks", "thank", "thank you", "nice", "helpful", "appreciate",
        "fantastic", "terrific", "outstanding", "impressive", "top notch", "clean",
        "satisfied", "satisfying", "pleasant", "efficient", "well done"
    }

    NEGATIVE_WORDS = {
        "bad", "worst", "poor", "terrible", "horrible", "awful", "unhappy",
        "disappointed", "disappointing", "sad", "annoying", "hate", "slow",
        "wrong", "useless", "garbage", "trash", "broken", "failed", "failure",
        "sucks", "problem", "issue", "error", "rubbish", "pathetic", "frustrated",
        "disaster", "inferior", "ugly", "painful", "worthless"
    }

    NEGATIONS = {"not", "never", "no", "hardly", "barely", "scarcely", "without", "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't"}

    def analyze(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult(sentiment="UNKNOWN", score=0.0)

        cleaned = text.lower().strip()
        tokens = re.findall(r"\b[a-zA-Z']+\b", cleaned)

        pos_cues: List[str] = []
        neg_cues: List[str] = []

        # Check multi-word expressions first
        if "thank you" in cleaned or "thanks" in tokens or "well done" in cleaned:
            pos_cues.append("gratitude")

        for i, token in enumerate(tokens):
            prev_token = tokens[i - 1] if i > 0 else ""
            is_negated = prev_token in self.NEGATIONS

            if token in self.POSITIVE_WORDS:
                if is_negated:
                    neg_cues.append(f"not {token}")
                else:
                    pos_cues.append(token)
            elif token in self.NEGATIVE_WORDS:
                if is_negated:
                    pos_cues.append(f"not {token}")
                else:
                    neg_cues.append(token)

        pos_count = len(pos_cues)
        neg_count = len(neg_cues)

        if pos_count > 0 and neg_count > 0:
            score = (pos_count - neg_count) / (pos_count + neg_count)
            return SentimentResult(sentiment="MIXED", score=round(score, 2), positive_cues=pos_cues, negative_cues=neg_cues)

        if pos_count > 0:
            score = min(1.0, 0.5 + 0.25 * pos_count)
            return SentimentResult(sentiment="POSITIVE", score=round(score, 2), positive_cues=pos_cues, negative_cues=[])

        if neg_count > 0:
            score = max(-1.0, -0.5 - 0.25 * neg_count)
            return SentimentResult(sentiment="NEGATIVE", score=round(score, 2), positive_cues=[], negative_cues=neg_cues)

        return SentimentResult(sentiment="NEUTRAL", score=0.0, positive_cues=[], negative_cues=[])


sentiment_analyzer = SentimentAnalyzer()


def get_sentiment(text: str) -> str:
    """Convenience helper returning sentiment label string."""
    return sentiment_analyzer.analyze(text).sentiment

