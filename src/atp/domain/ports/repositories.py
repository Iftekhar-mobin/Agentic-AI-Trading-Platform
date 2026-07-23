"""Repository ports for persisted market data."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from atp.domain.models.market import BarInterval, PriceHistory


class BarRepository(Protocol):
    """Stores and queries OHLCV bars keyed by (symbol, interval, timestamp)."""

    async def upsert_bars(self, history: PriceHistory) -> int:
        """Insert or update bars; idempotent. Returns the number of bars written."""
        ...

    async def get_bars(
        self,
        symbol: str,
        interval: BarInterval,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> PriceHistory:
        """Return stored bars ascending; ``limit`` keeps the most recent N."""
        ...

    async def latest_timestamp(self, symbol: str, interval: BarInterval) -> datetime | None:
        """Timestamp of the newest stored bar, or None if none exist."""
        ...
