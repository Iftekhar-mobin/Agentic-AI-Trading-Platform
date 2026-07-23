"""Port for external market data sources."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from atp.domain.models.market import BarInterval, PriceHistory


class MarketDataProvider(Protocol):
    """Fetches historical bars from an external source (Yahoo, Polygon, Alpaca, ...).

    Implementations must return bars normalized to domain models: UTC
    timestamps, ascending order, invalid rows dropped (and logged), never
    raising on partially bad vendor data.
    """

    async def get_bars(
        self,
        symbol: str,
        interval: BarInterval,
        *,
        start: datetime,
        end: datetime | None = None,
    ) -> PriceHistory:
        """Return bars in ``[start, end]`` (``end=None`` means "up to now")."""
        ...
