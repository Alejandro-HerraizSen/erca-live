"""
Sentiment scoring via VADER (Valence Aware Dictionary and sEntiment Reasoner).
Used for both social media posts and news headlines.
No API key required — pure lexicon-based NLP.
"""

from __future__ import annotations
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Any

_analyzer = SentimentIntensityAnalyzer()


def score_text(text: str) -> float:
    """
    Returns the VADER compound score in [-1.0, +1.0].
    -1 = most negative, +1 = most positive, 0 = neutral.
    """
    if not text or not isinstance(text, str):
        return 0.0
    scores = _analyzer.polarity_scores(text)
    return float(scores["compound"])


def score_detail(text: str) -> dict[str, float]:
    """Returns full VADER scores: pos, neg, neu, compound."""
    if not text or not isinstance(text, str):
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}
    return _analyzer.polarity_scores(str(text))


def score_batch(items: list[dict], text_key: str = "text") -> list[dict]:
    """
    Score a list of dicts in-place.
    Adds 'sentiment' key (compound) and 'sentiment_label' key.
    """
    out = []
    for item in items:
        text = item.get(text_key, "")
        s = score_text(str(text))
        label = "🟢 Bullish" if s > 0.05 else "🔴 Bearish" if s < -0.05 else "⚪ Neutral"
        out.append({**item, "sentiment": s, "sentiment_label": label})
    return out


def sentiment_label(score: float) -> str:
    if score > 0.25:
        return "Very Bullish"
    elif score > 0.05:
        return "Bullish"
    elif score < -0.25:
        return "Very Bearish"
    elif score < -0.05:
        return "Bearish"
    return "Neutral"


def sentiment_color(score: float) -> str:
    if score > 0.05:
        return "#00C853"
    elif score < -0.05:
        return "#D50000"
    return "#9E9E9E"
