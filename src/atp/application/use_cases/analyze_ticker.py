"""Use case: run technical analysis on a ticker.

Reads bars from local storage first; when storage is unreachable or too thin,
falls back to fetching directly from the market data provider so analysis
works even without a database (useful in development, resilient in production).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import structlog

from atp.application.agents.technical_analysis import TechnicalAnalysisAgent
from atp.application.use_cases.sync_market_data import DEFAULT_LOOKBACK
from atp.domain.errors import RepositoryUnavailableError
from atp.domain.models.analysis import MIN_HISTORY_BARS, TechnicalReport
from atp.domain.models.market import BarInterval, PriceHistory
from atp.domain.ports.market_data import MarketDataProvider
from atp.domain.ports.repositories import BarRepository

log = structlog.get_logger()


class AnalyzeTicker:
    def __init__(
        self,
        repository: BarRepository,
        provider: MarketDataProvider,
        agent: TechnicalAnalysisAgent,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._provider = provider
        self._agent = agent
        self._clock = clock

    async def execute(
        self, symbol: str, interval: BarInterval = BarInterval.DAY_1
    ) -> TechnicalReport:
        symbol = symbol.strip().upper()

        history = PriceHistory(symbol=symbol, interval=interval)
        try:
            history = await self._repository.get_bars(symbol, interval)
        except RepositoryUnavailableError:
            log.warning("analysis.repository_unavailable", symbol=symbol)

        if len(history) < MIN_HISTORY_BARS:
            log.info(
                "analysis.fetching_from_provider",
                symbol=symbol,
                stored_bars=len(history),
            )
            start = self._clock() - DEFAULT_LOOKBACK[interval]
            history = await self._provider.get_bars(symbol, interval, start=start)

        return await self._agent.analyze(history)
