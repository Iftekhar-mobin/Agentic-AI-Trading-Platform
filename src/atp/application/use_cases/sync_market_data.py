"""Use case: pull bars from a market data provider into local storage.

Sync is incremental: it resumes from the newest stored bar (re-fetching that
bar itself, since the provider may have corrected it — the upsert makes this
idempotent) and falls back to a per-interval default lookback on first sync.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import structlog
from pydantic import BaseModel

from atp.domain.models.market import BarInterval
from atp.domain.ports.market_data import MarketDataProvider
from atp.domain.ports.repositories import BarRepository

log = structlog.get_logger()

DEFAULT_LOOKBACK: dict[BarInterval, timedelta] = {
    # Intraday windows respect Yahoo's fetch limits (7d for 1m, 60d for 5m/15m).
    BarInterval.MIN_1: timedelta(days=5),
    BarInterval.MIN_5: timedelta(days=30),
    BarInterval.MIN_15: timedelta(days=30),
    BarInterval.HOUR_1: timedelta(days=180),
    BarInterval.DAY_1: timedelta(days=730),
    BarInterval.WEEK_1: timedelta(days=1825),
}


class SyncResult(BaseModel):
    symbol: str
    interval: BarInterval
    window_start: datetime
    fetched: int
    stored: int


class SyncMarketData:
    def __init__(
        self,
        provider: MarketDataProvider,
        repository: BarRepository,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._provider = provider
        self._repository = repository
        self._clock = clock

    async def execute(
        self,
        symbol: str,
        interval: BarInterval = BarInterval.DAY_1,
        *,
        lookback: timedelta | None = None,
    ) -> SyncResult:
        symbol = symbol.strip().upper()
        latest = await self._repository.latest_timestamp(symbol, interval)
        start = latest or self._clock() - (lookback or DEFAULT_LOOKBACK[interval])

        history = await self._provider.get_bars(symbol, interval, start=start)
        stored = await self._repository.upsert_bars(history)

        log.info(
            "market_data.synced",
            symbol=symbol,
            interval=interval.value,
            window_start=start.isoformat(),
            fetched=len(history),
            stored=stored,
        )
        return SyncResult(
            symbol=symbol,
            interval=interval,
            window_start=start,
            fetched=len(history),
            stored=stored,
        )
