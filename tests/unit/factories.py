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
    TimeframeReadings,
)
from atp.domain.models.explainability import Evidence
from atp.domain.models.fundamentals import (
    FundamentalAssessment,
    FundamentalReport,
    Fundamentals,
)
from atp.domain.models.market import BarInterval
from atp.domain.models.memory import (
    EpisodeKind,
    EpisodeMatch,
    JournalEntry,
    LearningReport,
    MemoryEpisode,
    RegimeTag,
    RegimeTrend,
    RegimeVolatility,
)
from atp.domain.models.news import NewsArticle, NewsAssessment, NewsReport
from atp.domain.models.patterns import (
    ChartPattern,
    ChartPatternReport,
    PatternAssessment,
    PatternKind,
    TimeframePatterns,
)
from atp.domain.models.research import (
    MarketContextReading,
    MarketResearchAssessment,
    MarketResearchReport,
)
from atp.domain.models.sentiment import (
    ScoredArticle,
    SentimentAssessment,
    SentimentLabel,
    SentimentReport,
    SentimentScore,
    SentimentSummary,
)

AS_OF = datetime(2026, 7, 22, tzinfo=UTC)


def make_timeframe_readings(
    interval: BarInterval = BarInterval.DAY_1,
    direction: SignalDirection = SignalDirection.BULLISH,
) -> TimeframeReadings:
    return TimeframeReadings(
        interval=interval,
        as_of=AS_OF,
        latest_close=325.89,
        bars=200,
        readings=(),
        signal_counts=dict.fromkeys(SignalDirection, 0),
        direction=direction,
    )


def make_technical_report(
    symbol: str = "AAPL",
    intervals: tuple[BarInterval, ...] = (BarInterval.DAY_1,),
) -> TechnicalReport:
    return TechnicalReport(
        symbol=symbol,
        timeframes=tuple(make_timeframe_readings(interval) for interval in intervals),
        assessment=TechnicalAssessment(
            direction=SignalDirection.BULLISH,
            reasoning="Trend indicators align.",
            confidence=0.7,
            evidence=(Evidence(source="1d:sma_trend", statement="Above SMA(50)."),),
            invalidation_conditions=("Close below SMA(50).",),
            timeframe_alignment="All requested timeframes point the same way.",
        ),
    )


def make_chart_pattern_report(
    symbol: str = "AAPL",
    intervals: tuple[BarInterval, ...] = (BarInterval.DAY_1,),
) -> ChartPatternReport:
    pattern = ChartPattern(
        kind=PatternKind.DOUBLE_BOTTOM,
        direction=SignalDirection.BULLISH,
        start=AS_OF - timedelta(days=30),
        end=AS_OF,
        levels={"trough_1": 300.0, "trough_2": 302.0, "neckline": 320.0},
        confirmed=True,
        quality=0.8,
        summary="Double bottom at 300.00/302.00 with a neckline at 320.00 - confirmed",
    )
    return ChartPatternReport(
        symbol=symbol,
        timeframes=tuple(
            TimeframePatterns(
                interval=interval,
                as_of=AS_OF,
                latest_close=325.89,
                bars=200,
                swings=(),
                levels=(),
                patterns=(pattern,),
                signal_counts=dict.fromkeys(SignalDirection, 0),
                direction=SignalDirection.BULLISH,
            )
            for interval in intervals
        ),
        assessment=PatternAssessment(
            direction=SignalDirection.BULLISH,
            reasoning="A confirmed double bottom on the highest timeframe.",
            confidence=0.6,
            evidence=(Evidence(source="1d:double_bottom", statement="Neckline broken at 320.00."),),
            invalidation_conditions=("A close back below 302.00.",),
            timeframe_alignment="Timeframes agree.",
        ),
    )


def make_market_research_report(symbol: str = "AAPL") -> MarketResearchReport:
    return MarketResearchReport(
        symbol=symbol,
        benchmark="SPY",
        interval=BarInterval.DAY_1,
        as_of=AS_OF,
        overlapping_bars=252,
        readings=(
            MarketContextReading(
                name="relative_strength_3m",
                values={"spread_pct": 6.2},
                direction=SignalDirection.BULLISH,
                summary="Outperforming the benchmark by 6.2pp over 63 bars.",
            ),
        ),
        signal_counts=dict.fromkeys(SignalDirection, 0),
        assessment=MarketResearchAssessment(
            direction=SignalDirection.BULLISH,
            reasoning="Leading a rising market.",
            confidence=0.65,
            evidence=(
                Evidence(source="relative_strength_3m", statement="Outperforming by 6.2pp."),
            ),
            invalidation_conditions=("Relative strength turning negative over 1m.",),
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


def make_regime(
    trend: RegimeTrend = RegimeTrend.UPTREND,
    volatility: RegimeVolatility = RegimeVolatility.NORMAL,
) -> RegimeTag:
    return RegimeTag(
        trend=trend,
        volatility=volatility,
        as_of=AS_OF,
        metrics={"sma_fast": 320.0, "sma_slow": 310.0, "vol_ratio": 1.0},
    )


def make_episode(
    identifier: str = "episode-0",
    *,
    symbol: str = "AAPL",
    kind: EpisodeKind = EpisodeKind.ANALYSIS,
    summary: str = "Technical bullish, news supportive.",
) -> MemoryEpisode:
    return MemoryEpisode(
        id=identifier,
        symbol=symbol,
        kind=kind,
        occurred_at=AS_OF - timedelta(days=30),
        summary=summary,
        regime=make_regime(),
        metadata={"recalled": 0},
    )


def make_learning_report(symbol: str = "AAPL") -> LearningReport:
    return LearningReport(
        symbol=symbol,
        as_of=AS_OF,
        regime=make_regime(),
        recalled=(EpisodeMatch(episode=make_episode(symbol=symbol), score=0.62),),
        entry=JournalEntry(
            reasoning="The setup rhymes with a precedent from a calmer tape.",
            confidence=0.4,
            evidence=(Evidence(source="episode-0", statement="Similar bullish setup."),),
            invalidation_conditions=("Volatility regime shifting to volatile.",),
            lessons=("Entries against the slow-SMA slope gave back gains.",),
            regime_note="The precedent happened in the same uptrend but calmer volatility.",
        ),
    )
