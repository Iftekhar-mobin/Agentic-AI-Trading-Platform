"""Yahoo Finance adapter for the MarketDataProvider port.

Development data source only: yfinance wraps Yahoo's unofficial API, which is
rate-limited and occasionally returns malformed rows. Production deployments
swap in a licensed provider (Polygon, Alpaca) behind the same port.

yfinance is synchronous, so downloads run in a worker thread to keep the event
loop free. Vendor rows that violate domain invariants (NaN, inconsistent OHLC)
are dropped and logged rather than failing the whole fetch.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pandas as pd
import structlog
import yfinance
from pydantic import ValidationError

from atp.domain.models.market import Bar, BarInterval, PriceHistory

log = structlog.get_logger()

_YF_INTERVALS: dict[BarInterval, str] = {
    BarInterval.MIN_1: "1m",
    BarInterval.MIN_5: "5m",
    BarInterval.MIN_15: "15m",
    BarInterval.HOUR_1: "1h",
    BarInterval.DAY_1: "1d",
    BarInterval.WEEK_1: "1wk",
}

_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class YFinanceMarketDataProvider:
    async def get_bars(
        self,
        symbol: str,
        interval: BarInterval,
        *,
        start: datetime,
        end: datetime | None = None,
    ) -> PriceHistory:
        symbol = symbol.strip().upper()
        frame = await asyncio.to_thread(self._download, symbol, interval, start, end)
        bars = self._to_bars(symbol, frame)
        return PriceHistory(symbol=symbol, interval=interval, bars=bars)

    @staticmethod
    def _download(
        symbol: str, interval: BarInterval, start: datetime, end: datetime | None
    ) -> pd.DataFrame:
        frame = yfinance.download(
            tickers=symbol,
            start=start,
            end=end,
            interval=_YF_INTERVALS[interval],
            auto_adjust=True,  # split/dividend-adjusted prices for analysis
            progress=False,
            threads=False,
        )
        return frame if frame is not None else pd.DataFrame()

    @staticmethod
    def _to_bars(symbol: str, frame: pd.DataFrame) -> tuple[Bar, ...]:
        if frame.empty:
            log.warning("market_data.empty_response", symbol=symbol)
            return ()

        if isinstance(frame.columns, pd.MultiIndex):
            # Single-ticker downloads still come back with a (field, ticker) header.
            frame = frame.droplevel(1, axis=1)
        frame = frame.rename(columns=lambda name: str(name).lower())[_OHLCV_COLUMNS].dropna()
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()

        index = pd.DatetimeIndex(frame.index)
        index = index.tz_localize(UTC) if index.tz is None else index.tz_convert(UTC)

        bars: list[Bar] = []
        dropped = 0
        for timestamp, row in zip(index, frame.to_numpy(), strict=True):
            open_, high, low, close, volume = (float(value) for value in row)
            try:
                bars.append(
                    Bar(
                        timestamp=timestamp.to_pydatetime(),
                        open=open_,
                        high=high,
                        low=low,
                        close=close,
                        volume=volume,
                    )
                )
            except ValidationError:
                dropped += 1
        if dropped:
            log.warning("market_data.invalid_bars_dropped", symbol=symbol, count=dropped)
        return tuple(bars)
