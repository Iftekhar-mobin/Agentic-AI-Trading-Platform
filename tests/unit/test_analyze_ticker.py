"""Tests for the AnalyzeTicker use case: storage-first with provider fallback."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TypeVar

from pydantic import BaseModel

from atp.application.agents import TechnicalAnalysisAgent
from atp.application.use_cases import AnalyzeTicker, LoadPriceHistory, LoadTimeframes
from atp.domain.errors import RepositoryUnavailableError
from atp.domain.models.analysis import SignalDirection, TechnicalAssessment
from atp.domain.models.explainability import Evidence
from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.infrastructure.indicators import PandasIndicatorEngine

T = TypeVar("T", bound=BaseModel)

NOW = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)


def make_history(n: int, interval: BarInterval = BarInterval.DAY_1) -> PriceHistory:
    step = timedelta(minutes=interval.minutes)
    start = NOW - step * n
    bars = tuple(
        Bar(
            timestamp=start + step * i,
            open=99.5 + i,
            high=101.0 + i,
            low=98.5 + i,
            close=100.0 + i,
            volume=1_000.0,
        )
        for i in range(n)
    )
    return PriceHistory(symbol="AAPL", interval=interval, bars=bars)


class FakeLLM:
    async def generate_structured(self, response_model: type[T], *, system: str, prompt: str) -> T:
        assessment = TechnicalAssessment(
            direction=SignalDirection.BULLISH,
            reasoning="Trend up.",
            confidence=0.6,
            evidence=(Evidence(source="sma_trend", statement="Above SMA."),),
            invalidation_conditions=("Close below SMA(50).",),
            timeframe_alignment="Only one timeframe was requested.",
        )
        assert isinstance(assessment, response_model)
        return assessment


class StubRepository:
    def __init__(self, history: PriceHistory | None = None, *, unavailable: bool = False) -> None:
        self._history = history
        self._unavailable = unavailable

    async def upsert_bars(self, history: PriceHistory) -> int:
        raise NotImplementedError

    async def get_bars(
        self,
        symbol: str,
        interval: BarInterval,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> PriceHistory:
        if self._unavailable:
            raise RepositoryUnavailableError("db down")
        return self._history or PriceHistory(symbol=symbol, interval=interval)

    async def latest_timestamp(self, symbol: str, interval: BarInterval) -> datetime | None:
        return None


class StubProvider:
    """Serves however many bars it was built with, on whichever interval is asked."""

    def __init__(self, history: PriceHistory) -> None:
        self._bars = len(history)
        self.calls = 0

    async def get_bars(
        self,
        symbol: str,
        interval: BarInterval,
        *,
        start: datetime,
        end: datetime | None = None,
    ) -> PriceHistory:
        self.calls += 1
        return make_history(self._bars, interval)


def make_use_case(repository: StubRepository, provider: StubProvider) -> AnalyzeTicker:
    agent = TechnicalAnalysisAgent(FakeLLM(), PandasIndicatorEngine())
    loader = LoadPriceHistory(repository, provider, clock=lambda: NOW)
    return AnalyzeTicker(LoadTimeframes(loader), agent)


async def test_uses_stored_bars_when_sufficient() -> None:
    provider = StubProvider(make_history(80))
    use_case = make_use_case(StubRepository(make_history(80)), provider)

    report = await use_case.execute("aapl")

    assert report.symbol == "AAPL"
    assert provider.calls == 0


async def test_falls_back_to_provider_when_storage_is_thin() -> None:
    provider = StubProvider(make_history(80))
    use_case = make_use_case(StubRepository(make_history(5)), provider)

    report = await use_case.execute("AAPL")

    assert provider.calls == 1
    assert report.symbol == "AAPL"


async def test_falls_back_to_provider_when_storage_is_down() -> None:
    provider = StubProvider(make_history(80))
    use_case = make_use_case(StubRepository(unavailable=True), provider)

    report = await use_case.execute("AAPL")

    assert provider.calls == 1
    assert report.assessment.direction is SignalDirection.BULLISH


async def test_multi_timeframe_request_loads_each_timeframe() -> None:
    """One report, one assessment, one entry per timeframe — highest first."""
    provider = StubProvider(make_history(80))
    use_case = make_use_case(StubRepository(unavailable=True), provider)

    report = await use_case.execute("AAPL", [BarInterval.HOUR_1, BarInterval.DAY_1])

    assert provider.calls == 2
    assert [frame.interval for frame in report.timeframes] == [
        BarInterval.DAY_1,
        BarInterval.HOUR_1,
    ]
    assert report.primary.interval is BarInterval.DAY_1


async def test_a_timeframe_that_fails_does_not_sink_the_rest() -> None:
    """Intraday history is fragile; losing 1H must not cost the 1D view."""

    class PartialProvider(StubProvider):
        async def get_bars(
            self,
            symbol: str,
            interval: BarInterval,
            *,
            start: datetime,
            end: datetime | None = None,
        ) -> PriceHistory:
            self.calls += 1
            if interval is BarInterval.HOUR_1:
                return PriceHistory(symbol=symbol, interval=interval)
            return make_history(self._bars, interval)

    provider = PartialProvider(make_history(80))
    use_case = make_use_case(StubRepository(unavailable=True), provider)

    report = await use_case.execute("AAPL", [BarInterval.DAY_1, BarInterval.HOUR_1])

    assert [frame.interval for frame in report.timeframes] == [BarInterval.DAY_1]


async def test_every_timeframe_failing_raises() -> None:
    import pytest

    from atp.domain.errors import InsufficientHistoryError

    provider = StubProvider(make_history(5))
    use_case = make_use_case(StubRepository(unavailable=True), provider)

    with pytest.raises(InsufficientHistoryError, match="no timeframe"):
        await use_case.execute("AAPL", [BarInterval.DAY_1, BarInterval.HOUR_1])
