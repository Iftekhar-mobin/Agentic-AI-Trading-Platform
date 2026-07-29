"""Unit tests for the fundamentals, news and FinBERT adapters (network faked)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest
import yfinance

from atp.domain.errors import ModelUnavailableError
from atp.domain.models.sentiment import SentimentLabel
from atp.infrastructure.fundamentals import YFinanceFundamentalsProvider
from atp.infrastructure.news import YFinanceNewsProvider
from atp.infrastructure.sentiment import FinBertSentimentModel

NOW = datetime(2026, 7, 22, 12, tzinfo=UTC)

YAHOO_INFO: dict[str, Any] = {
    "currency": "USD",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "marketCap": 3_200_000_000_000,
    "trailingPE": 28.42,
    "forwardPE": 24.1,
    "trailingPegRatio": 1.83,
    "priceToBook": 46.2,
    "profitMargins": 0.2471,
    "operatingMargins": 0.3123,
    "returnOnEquity": 1.4712,
    "revenueGrowth": 0.061,
    "earningsGrowth": 0.078,
    "debtToEquity": 145.0,
    "currentRatio": 0.867,
    "freeCashflow": 99_000_000_000,
    "dividendYield": 0.0044,
    "beta": 1.2,
}


class FakeTicker:
    """Stands in for yfinance.Ticker; raises if configured to."""

    info: ClassVar[dict[str, Any]] = {}
    news: ClassVar[list[Any]] = []
    error: ClassVar[Exception | None] = None

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def get_info(self) -> dict[str, Any]:
        if FakeTicker.error is not None:
            raise FakeTicker.error
        return FakeTicker.info

    def get_news(self, count: int = 10, tab: str = "news") -> list[Any]:
        if FakeTicker.error is not None:
            raise FakeTicker.error
        return FakeTicker.news[:count]


@pytest.fixture(autouse=True)
def fake_ticker(monkeypatch: pytest.MonkeyPatch) -> type[FakeTicker]:
    FakeTicker.info = dict(YAHOO_INFO)
    FakeTicker.news = []
    FakeTicker.error = None
    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    return FakeTicker


# --- Fundamentals adapter ---------------------------------------------------


async def test_fundamentals_are_normalized_from_vendor_keys() -> None:
    snapshot = await YFinanceFundamentalsProvider().get_fundamentals(" aapl ")

    assert snapshot.symbol == "AAPL"
    assert snapshot.sector == "Technology"
    assert snapshot.market_cap == 3.2e12
    assert snapshot.trailing_pe == pytest.approx(28.42)
    assert snapshot.peg_ratio == pytest.approx(1.83)  # from trailingPegRatio
    assert snapshot.profit_margin == pytest.approx(0.2471)
    assert snapshot.debt_to_equity == pytest.approx(145.0)
    assert snapshot.as_of.tzinfo is not None


async def test_legacy_peg_key_is_accepted() -> None:
    FakeTicker.info = {"pegRatio": 2.5}
    snapshot = await YFinanceFundamentalsProvider().get_fundamentals("AAPL")
    assert snapshot.peg_ratio == pytest.approx(2.5)


async def test_missing_metrics_stay_none_not_zero() -> None:
    FakeTicker.info = {"sector": "Technology"}
    snapshot = await YFinanceFundamentalsProvider().get_fundamentals("AAPL")

    assert snapshot.sector == "Technology"
    assert snapshot.trailing_pe is None
    assert snapshot.market_cap is None
    assert snapshot.free_cash_flow is None


@pytest.mark.parametrize("junk", [None, "n/a", True, float("nan")])
async def test_non_numeric_metrics_are_rejected(junk: Any) -> None:
    FakeTicker.info = {"trailingPE": junk}
    snapshot = await YFinanceFundamentalsProvider().get_fundamentals("AAPL")
    assert snapshot.trailing_pe is None


async def test_zero_market_cap_is_dropped_not_passed_to_the_model() -> None:
    """``Fundamentals.market_cap`` requires > 0; a vendor zero must not raise."""
    FakeTicker.info = {"marketCap": 0}
    snapshot = await YFinanceFundamentalsProvider().get_fundamentals("AAPL")
    assert snapshot.market_cap is None


async def test_vendor_failure_yields_an_empty_snapshot_not_an_exception() -> None:
    FakeTicker.error = RuntimeError("yahoo is down")
    snapshot = await YFinanceFundamentalsProvider().get_fundamentals("AAPL")

    assert snapshot.symbol == "AAPL"
    assert snapshot.trailing_pe is None


# --- News adapter -----------------------------------------------------------


def modern_item(identifier: str, *, iso: str, title: str = "Modern headline") -> dict[str, Any]:
    return {
        "id": identifier,
        "content": {
            "title": title,
            "summary": "Modern summary.",
            "pubDate": iso,
            "provider": {"displayName": "Reuters"},
            "canonicalUrl": {"url": f"https://example.test/{identifier}"},
        },
    }


def legacy_item(identifier: str, *, epoch: int, title: str = "Legacy headline") -> dict[str, Any]:
    return {
        "uuid": identifier,
        "title": title,
        "publisher": "Bloomberg",
        "link": f"https://example.test/{identifier}",
        "providerPublishTime": epoch,
        "relatedTickers": ["AAPL", "MSFT"],
    }


async def test_modern_payload_is_normalized() -> None:
    FakeTicker.news = [modern_item("a", iso="2026-07-22T10:00:00Z")]
    articles = await YFinanceNewsProvider().get_news("AAPL")

    assert len(articles) == 1
    article = articles[0]
    assert article.id == "a"
    assert article.title == "Modern headline"
    assert article.summary == "Modern summary."
    assert article.publisher == "Reuters"
    assert article.url == "https://example.test/a"
    assert article.published_at == datetime(2026, 7, 22, 10, tzinfo=UTC)


async def test_legacy_payload_is_normalized() -> None:
    FakeTicker.news = [legacy_item("b", epoch=1_784_000_000)]
    articles = await YFinanceNewsProvider().get_news("AAPL")

    assert len(articles) == 1
    article = articles[0]
    assert article.id == "b"
    assert article.publisher == "Bloomberg"
    assert article.published_at.tzinfo is UTC
    assert article.symbols == ("AAPL", "MSFT")


async def test_both_payload_shapes_coexist_and_sort_newest_first() -> None:
    FakeTicker.news = [
        legacy_item("old", epoch=int(datetime(2026, 7, 20, tzinfo=UTC).timestamp())),
        modern_item("new", iso="2026-07-22T10:00:00Z"),
    ]
    articles = await YFinanceNewsProvider().get_news("AAPL")

    assert [article.id for article in articles] == ["new", "old"]


async def test_unusable_items_are_dropped_not_fatal() -> None:
    FakeTicker.news = [
        {"content": {"summary": "no title"}},  # no title
        {"uuid": "x", "title": "No timestamp"},  # no usable timestamp
        modern_item("good", iso="2026-07-22T10:00:00Z"),
        "not-a-dict",
    ]
    articles = await YFinanceNewsProvider().get_news("AAPL")

    assert [article.id for article in articles] == ["good"]


async def test_duplicates_are_collapsed_by_id() -> None:
    FakeTicker.news = [
        modern_item("dup", iso="2026-07-22T10:00:00Z"),
        modern_item("dup", iso="2026-07-22T09:00:00Z"),
    ]
    articles = await YFinanceNewsProvider().get_news("AAPL")
    assert len(articles) == 1


async def test_since_filters_and_limit_caps() -> None:
    FakeTicker.news = [
        modern_item("d1", iso="2026-07-22T10:00:00Z"),
        modern_item("d2", iso="2026-07-21T10:00:00Z"),
        modern_item("d3", iso="2026-07-01T10:00:00Z"),
    ]
    provider = YFinanceNewsProvider()

    recent = await provider.get_news("AAPL", since=datetime(2026, 7, 15, tzinfo=UTC))
    assert [article.id for article in recent] == ["d1", "d2"]

    capped = await provider.get_news("AAPL", limit=1)
    assert [article.id for article in capped] == ["d1"]


async def test_naive_vendor_timestamps_are_treated_as_utc() -> None:
    FakeTicker.news = [modern_item("naive", iso="2026-07-22T10:00:00")]
    articles = await YFinanceNewsProvider().get_news("AAPL")
    assert articles[0].published_at == datetime(2026, 7, 22, 10, tzinfo=UTC)


async def test_news_vendor_failure_yields_no_articles() -> None:
    FakeTicker.error = RuntimeError("yahoo is down")
    assert await YFinanceNewsProvider().get_news("AAPL") == ()


# --- FinBERT adapter --------------------------------------------------------


async def test_finbert_without_the_optional_extra_fails_loudly() -> None:
    """`transformers` is not installed in the default environment."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("the finbert extra is installed in this environment")

    with pytest.raises(ModelUnavailableError, match="finbert"):
        await FinBertSentimentModel().score(["Apple beats estimates"])


async def test_finbert_scores_nothing_without_loading_the_model() -> None:
    assert await FinBertSentimentModel().score([]) == ()


def test_finbert_name_is_the_model_id() -> None:
    assert FinBertSentimentModel("ProsusAI/finbert").name == "ProsusAI/finbert"


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("positive", SentimentLabel.POSITIVE),
        ("NEGATIVE", SentimentLabel.NEGATIVE),
        ("Neutral", SentimentLabel.NEUTRAL),
        ("LABEL_0", SentimentLabel.POSITIVE),
        ("something_unexpected", SentimentLabel.NEUTRAL),
    ],
)
def test_finbert_label_mapping(label: str, expected: SentimentLabel) -> None:
    score = FinBertSentimentModel._to_score([{"label": label, "score": 0.91}])
    assert score.label is expected
    assert score.confidence == pytest.approx(0.91)


def test_finbert_picks_the_highest_scoring_class() -> None:
    score = FinBertSentimentModel._to_score(
        [
            {"label": "positive", "score": 0.10},
            {"label": "negative", "score": 0.75},
            {"label": "neutral", "score": 0.15},
        ]
    )
    assert score.label is SentimentLabel.NEGATIVE
    assert score.confidence == pytest.approx(0.75)


def test_finbert_rejects_an_unusable_pipeline_result() -> None:
    with pytest.raises(ModelUnavailableError):
        FinBertSentimentModel._to_score([])
