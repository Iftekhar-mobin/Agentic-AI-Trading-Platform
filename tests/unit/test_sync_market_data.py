"""Unit tests for the SyncMarketData use case, using in-memory fakes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from atp.application.use_cases import SyncMarketData
from atp.application.use_cases.sync_market_data import DEFAULT_LOOKBACK
from atp.domain.models.market import Bar, BarInterval, PriceHistory

NOW = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)


def make_history(*timestamps: datetime) -> PriceHistory:
    bars = tuple(
        Bar(timestamp=ts, open=100.0, high=105.0, low=99.0, close=103.0, volume=1_000.0)
        for ts in sorted(timestamps)
    )
    return PriceHistory(symbol="AAPL", interval=BarInterval.DAY_1, bars=bars)


class FakeProvider:
    def __init__(self, history: PriceHistory) -> None:
        self._history = history
        self.requested_start: datetime | None = None

    async def get_bars(
        self,
        symbol: str,
        interval: BarInterval,
        *,
        start: datetime,
        end: datetime | None = None,
    ) -> PriceHistory:
        self.requested_start = start
        return self._history


class InMemoryBarRepository:
    def __init__(self) -> None:
        self.storage: dict[tuple[str, str, datetime], Bar] = {}

    async def upsert_bars(self, history: PriceHistory) -> int:
        for bar in history.bars:
            self.storage[history.symbol, history.interval.value, bar.timestamp] = bar
        return len(history.bars)

    async def get_bars(
        self,
        symbol: str,
        interval: BarInterval,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> PriceHistory:
        bars = sorted(
            (
                bar
                for (sym, ivl, _), bar in self.storage.items()
                if sym == symbol and ivl == interval.value
            ),
            key=lambda bar: bar.timestamp,
        )
        if limit is not None:
            bars = bars[-limit:]
        return PriceHistory(symbol=symbol, interval=interval, bars=tuple(bars))

    async def latest_timestamp(self, symbol: str, interval: BarInterval) -> datetime | None:
        timestamps = [
            ts for (sym, ivl, ts) in self.storage if sym == symbol and ivl == interval.value
        ]
        return max(timestamps, default=None)


async def test_first_sync_uses_default_lookback() -> None:
    provider = FakeProvider(make_history(NOW - timedelta(days=1), NOW))
    repository = InMemoryBarRepository()
    use_case = SyncMarketData(provider, repository, clock=lambda: NOW)

    result = await use_case.execute("aapl", BarInterval.DAY_1)

    assert provider.requested_start == NOW - DEFAULT_LOOKBACK[BarInterval.DAY_1]
    assert result.symbol == "AAPL"
    assert result.fetched == 2
    assert result.stored == 2
    assert len(repository.storage) == 2


async def test_incremental_sync_resumes_from_latest_stored_bar() -> None:
    existing = NOW - timedelta(days=3)
    repository = InMemoryBarRepository()
    await repository.upsert_bars(make_history(existing))

    provider = FakeProvider(make_history(existing, NOW - timedelta(days=1)))
    use_case = SyncMarketData(provider, repository, clock=lambda: NOW)

    result = await use_case.execute("AAPL", BarInterval.DAY_1)

    # Resumes from (and re-fetches) the newest stored bar; upsert deduplicates.
    assert provider.requested_start == existing
    assert result.fetched == 2
    assert len(repository.storage) == 2


async def test_explicit_lookback_overrides_default() -> None:
    provider = FakeProvider(make_history(NOW))
    use_case = SyncMarketData(provider, InMemoryBarRepository(), clock=lambda: NOW)

    await use_case.execute("AAPL", BarInterval.DAY_1, lookback=timedelta(days=7))

    assert provider.requested_start == NOW - timedelta(days=7)
