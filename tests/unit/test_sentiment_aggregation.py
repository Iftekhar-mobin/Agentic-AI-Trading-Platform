"""Tests for the deterministic sentiment aggregation service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from atp.domain.models.analysis import SignalDirection
from atp.domain.models.news import NewsArticle
from atp.domain.models.sentiment import ScoredArticle, SentimentLabel, SentimentScore
from atp.domain.services.sentiment_aggregation import direction_for, summarize

NOW = datetime(2026, 7, 22, 12, tzinfo=UTC)


def scored(
    label: SentimentLabel,
    *,
    confidence: float = 0.8,
    age_days: float = 0.0,
    identifier: str = "a",
) -> ScoredArticle:
    return ScoredArticle(
        article=NewsArticle(
            id=identifier,
            title="headline",
            published_at=NOW - timedelta(days=age_days),
        ),
        sentiment=SentimentScore(label=label, confidence=confidence),
    )


def test_empty_input_is_neutral_and_counted_as_zero() -> None:
    summary = summarize([], now=NOW)

    assert summary.article_count == 0
    assert summary.mean_polarity == 0.0
    assert summary.weighted_polarity == 0.0
    assert summary.direction is SignalDirection.NEUTRAL
    assert summary.label_counts == dict.fromkeys(SentimentLabel, 0)


def test_label_counts_cover_every_label() -> None:
    summary = summarize(
        [
            scored(SentimentLabel.POSITIVE, identifier="a"),
            scored(SentimentLabel.POSITIVE, identifier="b"),
            scored(SentimentLabel.NEGATIVE, identifier="c"),
        ],
        now=NOW,
    )

    assert summary.label_counts == {
        SentimentLabel.POSITIVE: 2,
        SentimentLabel.NEGATIVE: 1,
        SentimentLabel.NEUTRAL: 0,
    }
    assert summary.article_count == 3


def test_recency_weighting_favours_the_newer_article() -> None:
    """One fresh negative outweighs one stale positive of equal confidence."""
    summary = summarize(
        [
            scored(SentimentLabel.POSITIVE, age_days=14, identifier="old"),
            scored(SentimentLabel.NEGATIVE, age_days=0, identifier="new"),
        ],
        now=NOW,
        half_life_days=3.0,
    )

    assert summary.mean_polarity == pytest.approx(0.0)
    assert summary.weighted_polarity < -0.7
    assert summary.direction is SignalDirection.BEARISH


def test_half_life_is_exactly_half_weight() -> None:
    summary = summarize(
        [
            scored(SentimentLabel.POSITIVE, confidence=1.0, age_days=0, identifier="new"),
            scored(SentimentLabel.NEGATIVE, confidence=1.0, age_days=3, identifier="old"),
        ],
        now=NOW,
        half_life_days=3.0,
    )

    # weights 1.0 and 0.5 -> (1.0*1 + 0.5*-1) / 1.5
    assert summary.weighted_polarity == pytest.approx(1 / 3)


def test_future_timestamps_do_not_exceed_full_weight() -> None:
    """A vendor clock skew must not let one article dominate with weight > 1."""
    summary = summarize(
        [scored(SentimentLabel.POSITIVE, confidence=0.9, age_days=-5, identifier="future")],
        now=NOW,
    )
    assert summary.weighted_polarity == pytest.approx(0.9)


def test_all_articles_ancient_falls_back_to_unweighted_mean() -> None:
    """Weights underflow to zero at extreme age; report the mean, not a false neutral."""
    summary = summarize(
        [scored(SentimentLabel.NEGATIVE, confidence=0.9, age_days=100_000)],
        now=NOW,
        half_life_days=0.001,
    )

    assert summary.weighted_polarity == pytest.approx(-0.9)
    assert summary.direction is SignalDirection.BEARISH


def test_neutral_articles_dilute_a_directional_one() -> None:
    """The same positive article reads bullish alone and neutral among filler."""
    positive = scored(SentimentLabel.POSITIVE, confidence=0.4, identifier="a")
    alone = summarize([positive], now=NOW)
    diluted = summarize(
        [
            positive,
            scored(SentimentLabel.NEUTRAL, confidence=0.9, identifier="b"),
            scored(SentimentLabel.NEUTRAL, confidence=0.9, identifier="c"),
        ],
        now=NOW,
    )

    assert alone.direction is SignalDirection.BULLISH
    assert 0 < diluted.weighted_polarity < alone.weighted_polarity
    assert diluted.direction is SignalDirection.NEUTRAL


@pytest.mark.parametrize(
    ("polarity", "expected"),
    [
        (0.15, SignalDirection.BULLISH),
        (0.14, SignalDirection.NEUTRAL),
        (-0.14, SignalDirection.NEUTRAL),
        (-0.15, SignalDirection.BEARISH),
    ],
)
def test_direction_thresholds_are_inclusive(polarity: float, expected: SignalDirection) -> None:
    assert direction_for(polarity) is expected


def test_non_positive_half_life_is_rejected() -> None:
    with pytest.raises(ValueError, match="half_life_days"):
        summarize([scored(SentimentLabel.POSITIVE)], now=NOW, half_life_days=0)
