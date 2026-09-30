"""NLP module with decoupled sentiment and language utilities."""
from app.nlp.sentiment import sentiment_analyzer, get_sentiment, SentimentResult

__all__ = ["sentiment_analyzer", "get_sentiment", "SentimentResult"]

