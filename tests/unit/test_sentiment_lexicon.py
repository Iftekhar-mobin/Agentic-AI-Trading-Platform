"""Tests for the deterministic lexicon sentiment model."""

from __future__ import annotations

import pytest

from atp.domain.models.sentiment import SentimentLabel
from atp.infrastructure.sentiment import LexiconSentimentModel

model = LexiconSentimentModel()


def test_model_reports_a_stable_name() -> None:
    assert model.name == "lexicon-v1"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Apple beats earnings estimates as revenue surges to a record", SentimentLabel.POSITIVE),
        ("Apple misses estimates; shares plunge on weak guidance", SentimentLabel.NEGATIVE),
        ("Apple to hold its annual shareholder meeting on Tuesday", SentimentLabel.NEUTRAL),
    ],
)
def test_obvious_headlines_are_classified(text: str, expected: SentimentLabel) -> None:
    assert model.score_text(text).label is expected


def test_negation_flips_polarity() -> None:
    plain = model.score_text("The company beat expectations")
    negated = model.score_text("The company did not beat expectations")

    assert plain.label is SentimentLabel.POSITIVE
    assert negated.label is SentimentLabel.NEGATIVE


def test_negation_reaches_only_a_short_window() -> None:
    """A negator must not colour a sentiment word ten tokens later."""
    score = model.score_text(
        "Analysts could not confirm the report but the company beat expectations"
    )
    assert score.label is SentimentLabel.POSITIVE


def test_balanced_language_reads_neutral() -> None:
    score = model.score_text("Revenue grew but margins declined")
    assert score.label is SentimentLabel.NEUTRAL


def test_polarity_is_signed_by_label() -> None:
    positive = model.score_text("record profit surge")
    negative = model.score_text("bankruptcy fraud lawsuit")
    neutral = model.score_text("the meeting is on Tuesday")

    assert positive.polarity > 0
    assert negative.polarity < 0
    assert neutral.polarity == 0.0
    assert positive.polarity == positive.confidence
    assert negative.polarity == -negative.confidence


def test_confidence_stays_within_bounds() -> None:
    """Even unanimous language must not claim certainty."""
    score = model.score_text("record record record surge surge profit beat win")
    assert 0.0 <= score.confidence <= 0.95


def test_empty_text_is_neutral() -> None:
    assert model.score_text("").label is SentimentLabel.NEUTRAL


async def test_score_preserves_input_order_and_length() -> None:
    texts = ["record profit", "", "bankruptcy fraud"]
    scores = await model.score(texts)

    assert len(scores) == len(texts)
    assert [score.label for score in scores] == [
        SentimentLabel.POSITIVE,
        SentimentLabel.NEUTRAL,
        SentimentLabel.NEGATIVE,
    ]


async def test_scoring_no_texts_returns_nothing() -> None:
    assert await model.score([]) == ()
