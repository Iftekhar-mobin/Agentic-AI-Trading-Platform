"""MarketDataProvider decorator that synthesizes higher timeframes.

Wraps any provider and transparently serves intervals the source does not have
(today: 4H, aggregated from 1H). Written as a decorator rather than baked into
the yfinance adapter so the capability survives the swap to Polygon or Alpaca —
neither of which serves 4H either.

The fetch window is widened slightly when aggregating, because the caller's
``start`` may land mid-bucket and a 4H candle built from two of its four hours
would be quietly wrong.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import structlog

from atp.domain.models.market import BarInterval, PriceHistory
from atp.domain.ports.market_data import MarketDataProvider
from atp.domain.services.resample import resample, source_interval

log = structlog.get_logger()


class ResamplingMarketDataProvider:
    def __init__(self, inner: MarketDataProvider) -> None:
        self._inner = inner

    async def get_bars(
        self,
        symbol: str,
        interval: BarInterval,
        *,
        start: datetime,
        end: datetime | None = None,
    ) -> PriceHistory:
        source = source_interval(interval)
        if source is None:
            return await self._inner.get_bars(symbol, interval, start=start, end=end)

        # Step back one full target bucket so the first one is complete.
        widened = start - timedelta(minutes=interval.minutes)
        history = await self._inner.get_bars(symbol, source, start=widened, end=end)
        aggregated = resample(history, interval)
        log.debug(
            "market_data.resampled",
            symbol=symbol,
            source=source.value,
            target=interval.value,
            source_bars=len(history),
            target_bars=len(aggregated),
        )
        return aggregated
