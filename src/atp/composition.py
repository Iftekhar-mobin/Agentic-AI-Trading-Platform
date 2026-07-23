"""Composition root: the only place where layers are wired together.

Everything downstream of here depends on ports, so swapping an adapter
(yfinance -> Polygon, Timescale -> anything) is a one-line change in ``build``.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncEngine

from atp.application.use_cases import GetPriceHistory, SyncMarketData
from atp.domain.ports import BarRepository, MarketDataProvider
from atp.infrastructure.config import Settings, get_settings
from atp.infrastructure.market_data import YFinanceMarketDataProvider
from atp.infrastructure.persistence import TimescaleBarRepository, create_engine


@dataclass(frozen=True)
class Container:
    settings: Settings
    engine: AsyncEngine
    bar_repository: BarRepository
    market_data: MarketDataProvider
    sync_market_data: SyncMarketData
    get_price_history: GetPriceHistory

    @classmethod
    def build(cls, settings: Settings | None = None) -> Container:
        settings = settings or get_settings()
        engine = create_engine(settings.database.dsn)
        bar_repository = TimescaleBarRepository(engine)
        market_data = YFinanceMarketDataProvider()
        return cls(
            settings=settings,
            engine=engine,
            bar_repository=bar_repository,
            market_data=market_data,
            sync_market_data=SyncMarketData(market_data, bar_repository),
            get_price_history=GetPriceHistory(bar_repository),
        )

    async def aclose(self) -> None:
        await self.engine.dispose()
