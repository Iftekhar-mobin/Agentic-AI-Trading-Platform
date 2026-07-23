"""Use case: read stored price history."""

from __future__ import annotations

from datetime import datetime

from atp.domain.models.market import BarInterval, PriceHistory
from atp.domain.ports.repositories import BarRepository


class GetPriceHistory:
    def __init__(self, repository: BarRepository) -> None:
        self._repository = repository

    async def execute(
        self,
        symbol: str,
        interval: BarInterval = BarInterval.DAY_1,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> PriceHistory:
        return await self._repository.get_bars(
            symbol.strip().upper(), interval, start=start, end=end, limit=limit
        )
