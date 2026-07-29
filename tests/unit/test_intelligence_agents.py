"""Tests for the fundamental, news and sentiment agents (LLM faked at the port)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import TypeVar

import pytest
from factories import make_articles, make_fundamentals
from pydantic import BaseModel

from atp.application.agents import FundamentalAnalysisAgent, NewsAgent, SentimentAgent
from atp.domain.errors import InsufficientDataError
from atp.domain.models.analysis import SignalDirection
from atp.domain.models.explainability import Evidence
from atp.domain.models.fundamentals import FundamentalAssessment, Fundamentals
from atp.domain.models.news import NewsArticle, NewsAssessment
from atp.domain.models.sentiment import SentimentAssessment, SentimentLabel, SentimentScore
from atp.infrastructure.sentiment import LexiconSentimentModel

T = TypeVar("T", bound=BaseModel)

NOW = datetime(2026, 7, 22, 12, tzinfo=UTC)


class FakeLLM:
    def __init__(self, response: BaseModel) -> None:
        self._response = response
        self.system: str | None = None
        self.prompt: str | None = None

    async def generate_structured(self, response_model: type[T], *, system: str, prompt: str) -> T:
        self.system = system
        self.prompt = prompt
        assert isinstance(self._response, response_model)
        return self._response


class StubSentimentModel:
    """Returns a fixed label per call, and can under-deliver to test the guard."""

    def __init__(
        self,
        label: SentimentLabel = SentimentLabel.POSITIVE,
        *,
        truncate: int | None = None,
    ) -> None:
        self._label = label
        self._truncate = truncate
        self.texts: list[str] = []

    @property
    def name(self) -> str:
        return "stub-model"

    async def score(self, texts: Sequence[str]) -> tuple[SentimentScore, ...]:
        self.texts = list(texts)
        count = self._truncate if self._truncate is not None else len(texts)
        return tuple(SentimentScore(label=self._label, confidence=0.8) for _ in range(count))


def fundamental_assessment(source: str = "trailing_pe") -> FundamentalAssessment:
    return FundamentalAssessment(
        direction=SignalDirection.NEUTRAL,
        reasoning="Rich multiple, strong margins.",
        confidence=0.55,
        evidence=(Evidence(source=source, statement="Trailing P/E is 28.4."),),
        invalidation_conditions=("Margins compressing below 20%.",),
    )


def news_assessment(source: str = "article-0") -> NewsAssessment:
    return NewsAssessment(
        direction=SignalDirection.BULLISH,
        reasoning="Product cycle dominates coverage.",
        confidence=0.6,
        evidence=(Evidence(source=source, statement="Upgrade cycle reported."),),
        invalidation_conditions=("The company denies the report.",),
        key_themes=("product cycle",),
    )


def sentiment_assessment(
    direction: SignalDirection = SignalDirection.BULLISH,
    source: str = "article-0",
) -> SentimentAssessment:
    return SentimentAssessment(
        direction=direction,
        reasoning="Uniformly positive but thin.",
        confidence=0.45,
        evidence=(Evidence(source=source, statement="Positive headline."),),
        invalidation_conditions=("A guidance cut.",),
    )


# --- Fundamental agent ------------------------------------------------------


async def test_fundamental_report_combines_rules_and_llm() -> None:
    llm = FakeLLM(fundamental_assessment())
    report = await FundamentalAnalysisAgent(llm).analyze(make_fundamentals())

    assert report.symbol == "AAPL"
    assert report.readings
    assert sum(report.signal_counts.values()) == len(report.readings)
    assert report.assessment.direction is SignalDirection.NEUTRAL
    assert report.fundamentals.sector == "Technology"


async def test_fundamental_llm_sees_classified_readings_and_sector() -> None:
    llm = FakeLLM(fundamental_assessment())
    await FundamentalAnalysisAgent(llm).analyze(make_fundamentals())

    assert llm.prompt is not None
    payload = json.loads(llm.prompt)
    assert payload["sector"] == "Technology"
    names = {reading["name"] for reading in payload["readings"]}
    assert "trailing_pe" in names
    assert all("direction" in reading for reading in payload["readings"])
    assert llm.system is not None
    assert "never invent" in llm.system.lower()


async def test_fundamental_agent_rejects_thin_coverage() -> None:
    thin = Fundamentals(symbol="AAPL", as_of=NOW, trailing_pe=20.0)
    with pytest.raises(InsufficientDataError, match="required for an assessment"):
        await FundamentalAnalysisAgent(FakeLLM(fundamental_assessment())).analyze(thin)


async def test_fundamental_fabricated_evidence_is_logged_not_raised() -> None:
    """A citation outside the inputs is an audit-trail warning, not a workflow failure."""
    llm = FakeLLM(fundamental_assessment(source="invented_metric"))
    report = await FundamentalAnalysisAgent(llm).analyze(make_fundamentals())
    assert report.assessment.evidence[0].source == "invented_metric"


# --- News agent -------------------------------------------------------------


async def test_news_report_carries_articles_and_themes() -> None:
    llm = FakeLLM(news_assessment())
    agent = NewsAgent(llm, clock=lambda: NOW)
    articles = make_articles(3)

    report = await agent.analyze("aapl", articles, lookback_days=7)

    assert report.symbol == "AAPL"
    assert report.lookback_days == 7
    assert len(report.articles) == 3
    assert report.assessment.key_themes == ("product cycle",)


async def test_news_llm_receives_ids_and_article_age() -> None:
    llm = FakeLLM(news_assessment())
    await NewsAgent(llm, clock=lambda: NOW).analyze("AAPL", make_articles(2), lookback_days=7)

    assert llm.prompt is not None
    payload = json.loads(llm.prompt)
    assert {article["id"] for article in payload["articles"]} == {"article-0", "article-1"}
    assert all("age_hours" in article for article in payload["articles"])
    assert llm.system is not None
    assert "must be the id of one provided article" in llm.system


async def test_news_agent_rejects_empty_coverage() -> None:
    with pytest.raises(InsufficientDataError, match="no articles"):
        await NewsAgent(FakeLLM(news_assessment())).analyze("AAPL", [], lookback_days=7)


# --- Sentiment agent --------------------------------------------------------


async def test_sentiment_pipeline_scores_then_aggregates_then_interprets() -> None:
    llm = FakeLLM(sentiment_assessment())
    model = StubSentimentModel()
    agent = SentimentAgent(llm, model, clock=lambda: NOW)

    report = await agent.analyze("AAPL", make_articles(3))

    assert report.model_name == "stub-model"
    assert len(report.scored_articles) == 3
    assert report.summary.article_count == 3
    assert report.summary.label_counts[SentimentLabel.POSITIVE] == 3
    assert report.summary.direction is SignalDirection.BULLISH
    assert report.assessment.direction is SignalDirection.BULLISH


async def test_sentiment_model_receives_title_and_summary() -> None:
    model = StubSentimentModel()
    agent = SentimentAgent(FakeLLM(sentiment_assessment()), model, clock=lambda: NOW)

    await agent.analyze("AAPL", make_articles(1))

    assert model.texts == ["AAPL headline 0. Body text for headline 0."]


async def test_sentiment_llm_never_sees_raw_bodies_only_labels() -> None:
    llm = FakeLLM(sentiment_assessment())
    agent = SentimentAgent(llm, StubSentimentModel(), clock=lambda: NOW)

    await agent.analyze("AAPL", make_articles(2))

    assert llm.prompt is not None
    payload = json.loads(llm.prompt)
    assert payload["classifier"] == "stub-model"
    assert payload["summary"]["weighted_polarity"] == pytest.approx(0.8, abs=0.05)
    assert all("label" in article for article in payload["articles"])
    assert all("summary" not in article for article in payload["articles"])


async def test_sentiment_agent_rejects_empty_coverage() -> None:
    agent = SentimentAgent(FakeLLM(sentiment_assessment()), StubSentimentModel())
    with pytest.raises(InsufficientDataError, match="no articles"):
        await agent.analyze("AAPL", [])


async def test_sentiment_agent_rejects_a_short_score_batch() -> None:
    """A classifier that drops rows would silently misalign labels to articles."""
    agent = SentimentAgent(
        FakeLLM(sentiment_assessment()), StubSentimentModel(truncate=1), clock=lambda: NOW
    )
    with pytest.raises(InsufficientDataError, match="returned 1 scores for 3 articles"):
        await agent.analyze("AAPL", make_articles(3))


async def test_sentiment_agent_records_a_direction_override() -> None:
    """The LLM may disagree with the aggregate; the report keeps both."""
    llm = FakeLLM(sentiment_assessment(direction=SignalDirection.BEARISH))
    agent = SentimentAgent(llm, StubSentimentModel(), clock=lambda: NOW)

    report = await agent.analyze("AAPL", make_articles(3))

    assert report.summary.direction is SignalDirection.BULLISH
    assert report.assessment.direction is SignalDirection.BEARISH


async def test_sentiment_agent_with_the_real_lexicon_model() -> None:
    """End-to-end through the default classifier, no network involved."""
    articles = (
        NewsArticle(
            id="pos",
            title="Apple beats estimates as revenue surges to a record",
            published_at=NOW - timedelta(hours=1),
        ),
        NewsArticle(
            id="neg",
            title="Apple misses guidance; shares plunge on weak demand",
            published_at=NOW - timedelta(days=20),
        ),
    )
    agent = SentimentAgent(
        FakeLLM(sentiment_assessment(source="pos")), LexiconSentimentModel(), clock=lambda: NOW
    )

    report = await agent.analyze("AAPL", articles)

    assert report.model_name == "lexicon-v1"
    labels = {item.article.id: item.sentiment.label for item in report.scored_articles}
    assert labels == {"pos": SentimentLabel.POSITIVE, "neg": SentimentLabel.NEGATIVE}
    # The fresh positive outweighs the three-week-old negative.
    assert report.summary.weighted_polarity > 0
    assert report.summary.direction is SignalDirection.BULLISH
