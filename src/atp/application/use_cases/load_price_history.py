"""Use case: load price history, storage-first with live-provider fallback.

Shared by analysis and backtesting: reads from the repository when possible,
and falls back to the market data provider when storage is unreachable or has
fewer bars than the caller needs.

Ahead of both sits a third source: bars supplied with the request itself (see
``application.request_scope``). A caller that already holds the candles — a
trading bot asking whether the agents agree with a decision it just made on
them — should be analysed on those exact candles, not on a re-fetch.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import structlog

from atp.application.request_scope import supplied_history
from atp.application.use_cases.sync_market_data import DEFAULT_LOOKBACK
from atp.domain.errors import RepositoryUnavailableError
from atp.domain.models.market import BarInterval, PriceHistory
from atp.domain.ports.market_data import MarketDataProvider
from atp.domain.ports.repositories import BarRepository

log = structlog.get_logger()


class LoadPriceHistory:
    def __init__(
        self,
        repository: BarRepository,
        provider: MarketDataProvider,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._provider = provider
        self._clock = clock

    async def execute(
        self,
        symbol: str,
        interval: BarInterval,
        *,
        min_bars: int,
        lookback: timedelta | None = None,
    ) -> PriceHistory:
        symbol = symbol.strip().upper()

        # Supplied bars win outright, and are checked *before* the repository
        # rather than after it. This use case is storage-first: were the order
        # reversed, a symbol already stored with enough bars would return those
        # and the caller's candles would be silently ignored. The failure would
        # look like the agents simply disagreeing, with nothing anywhere to
        # indicate they had read entirely different data.
        supplied = supplied_history(symbol, interval)
        if supplied is not None:
            log.info(
                "history.using_supplied_bars",
                symbol=symbol,
                interval=interval.value,
                bars=len(supplied),
            )
            return supplied

        history = PriceHistory(symbol=symbol, interval=interval)
        try:
            history = await self._repository.get_bars(symbol, interval)
        except RepositoryUnavailableError:
            log.warning("history.repository_unavailable", symbol=symbol)

        if len(history) < min_bars:
            log.info(
                "history.fetching_from_provider",
                symbol=symbol,
                stored_bars=len(history),
                min_bars=min_bars,
            )
            start = self._clock() - (lookback or DEFAULT_LOOKBACK[interval])
            history = await self._provider.get_bars(symbol, interval, start=start)
        return history
