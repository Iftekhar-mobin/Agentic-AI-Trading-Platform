"""Composition root: the only place where layers are wired together.

Everything downstream of here depends on ports, so swapping an adapter
(yfinance -> Polygon, Anthropic -> another LLM) is a one-line change in
``build``.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncEngine

from atp.application.agents import TechnicalAnalysisAgent
from atp.application.orchestration import TradingOrchestrator
from atp.application.use_cases import AnalyzeTicker, GetPriceHistory, SyncMarketData
from atp.domain.ports import BarRepository, IndicatorEngine, LLMClient, MarketDataProvider
from atp.infrastructure.config import Settings, get_settings
from atp.infrastructure.indicators import PandasIndicatorEngine
from atp.infrastructure.llm import AnthropicLLMClient
from atp.infrastructure.market_data import YFinanceMarketDataProvider
from atp.infrastructure.persistence import TimescaleBarRepository, create_engine


@dataclass(frozen=True)
class Container:
    settings: Settings
    engine: AsyncEngine
    bar_repository: BarRepository
    market_data: MarketDataProvider
    indicator_engine: IndicatorEngine
    llm: LLMClient
    technical_analysis_agent: TechnicalAnalysisAgent
    sync_market_data: SyncMarketData
    get_price_history: GetPriceHistory
    analyze_ticker: AnalyzeTicker
    orchestrator: TradingOrchestrator

    @classmethod
    def build(cls, settings: Settings | None = None) -> Container:
        settings = settings or get_settings()
        engine = create_engine(settings.database.dsn)
        bar_repository = TimescaleBarRepository(engine)
        market_data = YFinanceMarketDataProvider()
        indicator_engine = PandasIndicatorEngine()
        llm = AnthropicLLMClient(settings.llm)
        technical_analysis_agent = TechnicalAnalysisAgent(llm, indicator_engine)
        analyze_ticker = AnalyzeTicker(bar_repository, market_data, technical_analysis_agent)
        return cls(
            settings=settings,
            engine=engine,
            bar_repository=bar_repository,
            market_data=market_data,
            indicator_engine=indicator_engine,
            llm=llm,
            technical_analysis_agent=technical_analysis_agent,
            sync_market_data=SyncMarketData(market_data, bar_repository),
            get_price_history=GetPriceHistory(bar_repository),
            analyze_ticker=analyze_ticker,
            orchestrator=TradingOrchestrator(analyze_ticker),
        )

    async def aclose(self) -> None:
        await self.engine.dispose()
