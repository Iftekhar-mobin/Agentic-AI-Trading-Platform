"""Tests for the AnalyzeTicker use case: storage-first with provider fallback."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TypeVar

from pydantic import BaseModel

from atp.application.agents import TechnicalAnalysisAgent
from atp.application.use_cases import AnalyzeTicker, LoadPriceHistory
from atp.domain.errors import RepositoryUnavailableError
from atp.domain.models.analysis import SignalDirection, TechnicalAssessment
from atp.domain.models.explainability import Evidence
from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.infrastructure.indicators import PandasIndicatorEngine

T = TypeVar("T", bound=BaseModel)

NOW = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)


def make_history(n: int) -> PriceHistory:
    start = NOW - timedelta(days=n)
    bars = tuple(
        Bar(
            timestamp=start + timedelta(days=i),
            open=99.5 + i,
            high=101.0 + i,
            low=98.5 + i,
            close=100.0 + i,
            volume=1_000.0,
        )
        for i in range(n)
    )
    return PriceHistory(symbol="AAPL", interval=BarInterval.DAY_1, bars=bars)


class FakeLLM:
    async def generate_structured(self, response_model: type[T], *, system: str, prompt: str) -> T:
        assessment = TechnicalAssessment(
            direction=SignalDirection.BULLISH,
            reasoning="Trend up.",
            confidence=0.6,
            evidence=(Evidence(source="sma_trend", statement="Above SMA."),),
            invalidation_conditions=("Close below SMA(50).",),
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
    def __init__(self, history: PriceHistory) -> None:
        self._history = history
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
        return self._history


def make_use_case(repository: StubRepository, provider: StubProvider) -> AnalyzeTicker:
    agent = TechnicalAnalysisAgent(FakeLLM(), PandasIndicatorEngine())
    loader = LoadPriceHistory(repository, provider, clock=lambda: NOW)
    return AnalyzeTicker(loader, agent)


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
