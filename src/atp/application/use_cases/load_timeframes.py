"""Use case: load price history for several timeframes at once.

Shared by the technical and chart-pattern agents, which both read the same set
of timeframes. Loads run concurrently — they are independent I/O, and doing
them in sequence would make a three-timeframe request three times slower for
no reason.

A timeframe that cannot be loaded is dropped with a warning rather than failing
the request: intraday history is far more fragile than daily (vendors cap it at
60 days, some symbols have none at all), and losing the 1H view is not a reason
to withhold the 1D one. Only when nothing loads does this raise.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import structlog

from atp.application.use_cases.load_price_history import LoadPriceHistory
from atp.domain.errors import DomainError, InsufficientHistoryError
from atp.domain.models.market import BarInterval, PriceHistory, sort_timeframes

log = structlog.get_logger()


class LoadTimeframes:
    def __init__(self, history: LoadPriceHistory) -> None:
        self._history = history

    async def execute(
        self,
        symbol: str,
        intervals: Sequence[BarInterval],
        *,
        min_bars: int,
    ) -> tuple[PriceHistory, ...]:
        """Return one history per usable timeframe, highest to lowest."""
        ordered = sort_timeframes(intervals)
        if not ordered:
            msg = f"{symbol}: no timeframes requested"
            raise InsufficientHistoryError(msg)

        results = await asyncio.gather(
            *(self._history.execute(symbol, interval, min_bars=min_bars) for interval in ordered),
            return_exceptions=True,
        )

        histories: list[PriceHistory] = []
        for interval, result in zip(ordered, results, strict=True):
            if isinstance(result, BaseException):
                if not isinstance(result, DomainError):
                    raise result
                log.warning(
                    "timeframes.unavailable",
                    symbol=symbol,
                    interval=interval.value,
                    error=str(result),
                )
                continue
            if len(result) < min_bars:
                log.warning(
                    "timeframes.too_few_bars",
                    symbol=symbol,
                    interval=interval.value,
                    bars=len(result),
                    required=min_bars,
                )
                continue
            histories.append(result)

        if not histories:
            requested = ", ".join(interval.value for interval in ordered)
            msg = f"{symbol}: no timeframe among [{requested}] has {min_bars} usable bars"
            raise InsufficientHistoryError(msg)
        return tuple(histories)
