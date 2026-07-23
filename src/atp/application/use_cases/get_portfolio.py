"""Use case: load the portfolio with refreshed prices and updated equity peak."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import structlog

from atp.domain.models.market import BarInterval
from atp.domain.models.trading import Portfolio
from atp.domain.ports.market_data import MarketDataProvider
from atp.domain.ports.portfolio import PortfolioRepository

log = structlog.get_logger()

_PRICE_LOOKBACK = timedelta(days=10)  # covers weekends/holidays for a latest close


class GetPortfolio:
    def __init__(
        self,
        repository: PortfolioRepository,
        market_data: MarketDataProvider,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._market_data = market_data
        self._clock = clock

    async def execute(self, *, refresh_prices: bool = True) -> Portfolio:
        portfolio = await self._repository.load()
        if refresh_prices and portfolio.positions:
            prices = await self._latest_prices([p.symbol for p in portfolio.positions])
            portfolio = portfolio.with_prices(prices)
        # Track the equity high-water mark: the drawdown halt depends on it.
        portfolio = portfolio.with_peak_updated()
        await self._repository.save(portfolio)
        return portfolio

    async def _latest_prices(self, symbols: list[str]) -> dict[str, Decimal]:
        start = self._clock() - _PRICE_LOOKBACK
        prices: dict[str, Decimal] = {}
        for symbol in symbols:
            history = await self._market_data.get_bars(symbol, BarInterval.DAY_1, start=start)
            if history.latest is not None:
                prices[symbol] = Decimal(str(history.latest.close))
            else:
                log.warning("portfolio.price_unavailable", symbol=symbol)
        return prices
