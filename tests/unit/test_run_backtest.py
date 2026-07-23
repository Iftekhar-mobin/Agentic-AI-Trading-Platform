"""Tests for the RunBacktest use case."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from atp.application.use_cases import LoadPriceHistory, RunBacktest
from atp.domain.errors import RepositoryUnavailableError
from atp.domain.models.backtest import BacktestResult
from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.domain.models.strategy import StrategyDefinition
from atp.domain.strategy_presets import PRESETS

NOW = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)


def make_history(n: int) -> PriceHistory:
    start = NOW - timedelta(days=n)
    closes = [100.0 - 0.25 * i for i in range(n // 2)]
    closes += [closes[-1] + 0.5 * i for i in range(1, n - len(closes) + 1)]
    bars = tuple(
        Bar(
            timestamp=start + timedelta(days=i),
            open=close * 0.999,
            high=close * 1.01,
            low=close * 0.985,
            close=close,
            volume=1_000_000.0,
        )
        for i, close in enumerate(closes)
    )
    return PriceHistory(symbol="TEST", interval=BarInterval.DAY_1, bars=bars)


class StubRepository:
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
        raise RepositoryUnavailableError("db down")

    async def latest_timestamp(self, symbol: str, interval: BarInterval) -> datetime | None:
        return None


class StubProvider:
    def __init__(self, history: PriceHistory) -> None:
        self._history = history

    async def get_bars(
        self,
        symbol: str,
        interval: BarInterval,
        *,
        start: datetime,
        end: datetime | None = None,
    ) -> PriceHistory:
        return self._history


class SpyEngine:
    def __init__(self) -> None:
        self.received_bars: int | None = None

    def run(
        self,
        strategy: StrategyDefinition,
        history: PriceHistory,
        *,
        initial_cash: float = 10_000.0,
        commission: float = 0.001,
    ) -> BacktestResult:
        self.received_bars = len(history)
        from atp.infrastructure.backtesting import BacktestingPyEngine

        return BacktestingPyEngine().run(
            strategy, history, initial_cash=initial_cash, commission=commission
        )


async def test_backtest_via_provider_fallback() -> None:
    history = make_history(200)
    engine = SpyEngine()
    loader = LoadPriceHistory(StubRepository(), StubProvider(history), clock=lambda: NOW)
    use_case = RunBacktest(loader, engine)

    result = await use_case.execute(PRESETS["ema_cross"], "test", initial_cash=5_000.0)

    assert engine.received_bars == 200
    assert result.initial_cash == 5_000.0
    assert result.symbol == "TEST"
    assert result.metrics.trades >= 1
