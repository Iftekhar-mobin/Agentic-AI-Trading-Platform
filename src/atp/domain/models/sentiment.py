"""Sentiment domain models.

Splits the FinBERT + LLM pipeline along the platform's standing rule: the
classifier produces numbers (per-article labels and probabilities), a
deterministic aggregation turns them into a summary, and only then does an LLM
interpret that summary. The LLM never scores text itself.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from atp.domain.models.analysis import SignalDirection
from atp.domain.models.explainability import Explanation
from atp.domain.models.news import NewsArticle

BULLISH_POLARITY_THRESHOLD = 0.15
"""Recency-weighted polarity at or above which the aggregate reads bullish."""

BEARISH_POLARITY_THRESHOLD = -0.15
"""Recency-weighted polarity at or below which the aggregate reads bearish."""


class SentimentLabel(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentScore(BaseModel):
    """One classifier verdict on one piece of text."""

    model_config = ConfigDict(frozen=True)

    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of the winning label")

    @property
    def polarity(self) -> float:
        """Signed strength in [-1, 1]; neutral is exactly 0."""
        if self.label is SentimentLabel.POSITIVE:
            return self.confidence
        if self.label is SentimentLabel.NEGATIVE:
            return -self.confidence
        return 0.0


class ScoredArticle(BaseModel):
    model_config = ConfigDict(frozen=True)

    article: NewsArticle
    sentiment: SentimentScore


class SentimentSummary(BaseModel):
    """Deterministic aggregate over scored articles — reproducible without an LLM."""

    model_config = ConfigDict(frozen=True)

    article_count: int = Field(ge=0)
    label_counts: dict[SentimentLabel, int]
    mean_polarity: float = Field(ge=-1.0, le=1.0)
    weighted_polarity: float = Field(
        ge=-1.0,
        le=1.0,
        description="Recency-weighted mean; stale headlines count for less",
    )
    direction: SignalDirection
    half_life_days: float = Field(gt=0)


class SentimentAssessment(Explanation):
    """The LLM's interpretation of the aggregate (evidence sources are article ids)."""

    direction: SignalDirection


class SentimentReport(BaseModel):
    """Full output of the Sentiment agent."""

    # protected_namespaces cleared so ``model_name`` (the classifier) is not
    # mistaken by Pydantic for one of its own ``model_*`` attributes.
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    symbol: str
    as_of: datetime
    model_name: str = Field(min_length=1, description="Classifier that produced the scores")
    scored_articles: tuple[ScoredArticle, ...]
    summary: SentimentSummary
    assessment: SentimentAssessment
