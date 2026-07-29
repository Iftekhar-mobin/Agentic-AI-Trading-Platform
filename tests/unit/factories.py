"""Report and stub factories shared by the orchestrator and API tests.

Both need one valid artifact per analysis agent; building them in each test
module would triple the churn every time a report model gains a field.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from atp.domain.models.analysis import (
    SignalDirection,
    TechnicalAssessment,
    TechnicalReport,
)
from atp.domain.models.explainability import Evidence
from atp.domain.models.fundamentals import (
    FundamentalAssessment,
    FundamentalReport,
    Fundamentals,
)
from atp.domain.models.market import BarInterval
from atp.domain.models.news import NewsArticle, NewsAssessment, NewsReport
from atp.domain.models.sentiment import (
    ScoredArticle,
    SentimentAssessment,
    SentimentLabel,
    SentimentReport,
    SentimentScore,
    SentimentSummary,
)

AS_OF = datetime(2026, 7, 22, tzinfo=UTC)


def make_technical_report(symbol: str = "AAPL") -> TechnicalReport:
    return TechnicalReport(
        symbol=symbol,
        interval=BarInterval.DAY_1,
        as_of=AS_OF,
        latest_close=325.89,
        readings=(),
        signal_counts=dict.fromkeys(SignalDirection, 0),
        assessment=TechnicalAssessment(
            direction=SignalDirection.BULLISH,
            reasoning="Trend indicators align.",
            confidence=0.7,
            evidence=(Evidence(source="sma_trend", statement="Above SMA(50)."),),
            invalidation_conditions=("Close below SMA(50).",),
        ),
    )


def make_fundamentals(symbol: str = "AAPL") -> Fundamentals:
    return Fundamentals(
        symbol=symbol,
        as_of=AS_OF,
        currency="USD",
        sector="Technology",
        industry="Consumer Electronics",
        market_cap=3.2e12,
        trailing_pe=28.4,
        forward_pe=24.1,
        peg_ratio=1.8,
        price_to_book=46.2,
        profit_margin=0.247,
        operating_margin=0.312,
        return_on_equity=1.47,
        revenue_growth=0.061,
        earnings_growth=0.078,
        debt_to_equity=145.0,
        current_ratio=0.87,
        free_cash_flow=9.9e10,
        dividend_yield=0.0044,
        beta=1.2,
    )


def make_fundamental_report(symbol: str = "AAPL") -> FundamentalReport:
    return FundamentalReport(
        symbol=symbol,
        as_of=AS_OF,
        fundamentals=make_fundamentals(symbol),
        readings=(),
        signal_counts=dict.fromkeys(SignalDirection, 0),
        assessment=FundamentalAssessment(
            direction=SignalDirection.NEUTRAL,
            reasoning="Rich multiple offset by strong margins.",
            confidence=0.55,
            evidence=(Evidence(source="profit_margin", statement="Net margin 24.7%."),),
            invalidation_conditions=("Revenue growth turning negative.",),
        ),
    )


def make_articles(count: int = 3, *, symbol: str = "AAPL") -> tuple[NewsArticle, ...]:
    return tuple(
        NewsArticle(
            id=f"article-{index}",
            title=f"{symbol} headline {index}",
            summary=f"Body text for headline {index}.",
            publisher="Reuters",
            url=f"https://example.test/{index}",
            published_at=AS_OF - timedelta(hours=index),
            symbols=(symbol,),
        )
        for index in range(count)
    )


def make_news_report(symbol: str = "AAPL") -> NewsReport:
    return NewsReport(
        symbol=symbol,
        as_of=AS_OF,
        lookback_days=7,
        articles=make_articles(symbol=symbol),
        assessment=NewsAssessment(
            direction=SignalDirection.BULLISH,
            reasoning="Coverage centres on a product cycle upgrade.",
            confidence=0.6,
            evidence=(Evidence(source="article-0", statement="Upgrade cycle reported."),),
            invalidation_conditions=("The company denies the report.",),
            key_themes=("product cycle",),
        ),
    )


def make_sentiment_report(symbol: str = "AAPL") -> SentimentReport:
    articles = make_articles(symbol=symbol)
    scored = tuple(
        ScoredArticle(
            article=article,
            sentiment=SentimentScore(label=SentimentLabel.POSITIVE, confidence=0.8),
        )
        for article in articles
    )
    return SentimentReport(
        symbol=symbol,
        as_of=AS_OF,
        model_name="lexicon-v1",
        scored_articles=scored,
        summary=SentimentSummary(
            article_count=len(scored),
            label_counts={
                SentimentLabel.POSITIVE: len(scored),
                SentimentLabel.NEGATIVE: 0,
                SentimentLabel.NEUTRAL: 0,
            },
            mean_polarity=0.8,
            weighted_polarity=0.8,
            direction=SignalDirection.BULLISH,
            half_life_days=3.0,
        ),
        assessment=SentimentAssessment(
            direction=SignalDirection.BULLISH,
            reasoning="Uniformly positive but thin coverage.",
            confidence=0.45,
            evidence=(Evidence(source="article-0", statement="Positive headline."),),
            invalidation_conditions=("A negative earnings pre-announcement.",),
        ),
    )
