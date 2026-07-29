"""Deterministic aggregation of per-article sentiment scores.

Headlines decay: a downgrade from three weeks ago should not outvote this
morning's earnings beat. Aggregation therefore weights each score by an
exponential recency factor with a configurable half-life, and derives the
directional call from fixed thresholds. Nothing here consults an LLM, so the
same articles always produce the same summary.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from atp.domain.models.analysis import SignalDirection
from atp.domain.models.sentiment import (
    BEARISH_POLARITY_THRESHOLD,
    BULLISH_POLARITY_THRESHOLD,
    ScoredArticle,
    SentimentLabel,
    SentimentSummary,
)

DEFAULT_HALF_LIFE_DAYS = 3.0
"""Age at which an article carries half the weight of a brand-new one."""


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _recency_weight(published_at: datetime, now: datetime, half_life_days: float) -> float:
    age_days = max((now - published_at).total_seconds() / 86_400.0, 0.0)
    return float(0.5 ** (age_days / half_life_days))


def direction_for(weighted_polarity: float) -> SignalDirection:
    if weighted_polarity >= BULLISH_POLARITY_THRESHOLD:
        return SignalDirection.BULLISH
    if weighted_polarity <= BEARISH_POLARITY_THRESHOLD:
        return SignalDirection.BEARISH
    return SignalDirection.NEUTRAL


def summarize(
    scored: Sequence[ScoredArticle],
    *,
    now: datetime,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> SentimentSummary:
    """Reduce scored articles to one reproducible summary."""
    if half_life_days <= 0:
        msg = f"half_life_days must be positive, got {half_life_days}"
        raise ValueError(msg)

    label_counts = {
        label: sum(1 for item in scored if item.sentiment.label is label)
        for label in SentimentLabel
    }
    if not scored:
        return SentimentSummary(
            article_count=0,
            label_counts=label_counts,
            mean_polarity=0.0,
            weighted_polarity=0.0,
            direction=SignalDirection.NEUTRAL,
            half_life_days=half_life_days,
        )

    polarities = [item.sentiment.polarity for item in scored]
    mean_polarity = sum(polarities) / len(polarities)

    weights = [_recency_weight(item.article.published_at, now, half_life_days) for item in scored]
    total_weight = sum(weights)
    weighted_polarity = (
        sum(weight * polarity for weight, polarity in zip(weights, polarities, strict=True))
        / total_weight
        if total_weight > 0
        # Every article is old enough that its weight underflowed to zero;
        # fall back to the unweighted mean rather than reporting a false neutral.
        else mean_polarity
    )

    return SentimentSummary(
        article_count=len(scored),
        label_counts=label_counts,
        mean_polarity=_clamp(mean_polarity),
        weighted_polarity=_clamp(weighted_polarity),
        direction=direction_for(weighted_polarity),
        half_life_days=half_life_days,
    )
