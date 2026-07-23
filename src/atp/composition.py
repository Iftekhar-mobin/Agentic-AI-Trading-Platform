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
from atp.application.use_cases import (
    AnalyzeTicker,
    GetPriceHistory,
    LoadPriceHistory,
    OptimizeStrategy,
    RunBacktest,
    SyncMarketData,
)
from atp.domain.ports import (
    BacktestEngine,
    BarRepository,
    IndicatorEngine,
    LLMClient,
    MarketDataProvider,
    StrategyOptimizer,
)
from atp.infrastructure.backtesting import BacktestingPyEngine
from atp.infrastructure.config import Settings, get_settings
from atp.infrastructure.indicators import PandasIndicatorEngine
from atp.infrastructure.llm import AnthropicLLMClient
from atp.infrastructure.market_data import YFinanceMarketDataProvider
from atp.infrastructure.optimization import OptunaStrategyOptimizer
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
    backtest_engine: BacktestEngine
    strategy_optimizer: StrategyOptimizer
    sync_market_data: SyncMarketData
    get_price_history: GetPriceHistory
    load_price_history: LoadPriceHistory
    analyze_ticker: AnalyzeTicker
    run_backtest: RunBacktest
    optimize_strategy: OptimizeStrategy
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
        load_price_history = LoadPriceHistory(bar_repository, market_data)
        analyze_ticker = AnalyzeTicker(load_price_history, technical_analysis_agent)
        backtest_engine = BacktestingPyEngine()
        strategy_optimizer = OptunaStrategyOptimizer(backtest_engine)
        return cls(
            settings=settings,
            engine=engine,
            bar_repository=bar_repository,
            market_data=market_data,
            indicator_engine=indicator_engine,
            llm=llm,
            technical_analysis_agent=technical_analysis_agent,
            backtest_engine=backtest_engine,
            strategy_optimizer=strategy_optimizer,
            sync_market_data=SyncMarketData(market_data, bar_repository),
            get_price_history=GetPriceHistory(bar_repository),
            load_price_history=load_price_history,
            analyze_ticker=analyze_ticker,
            run_backtest=RunBacktest(load_price_history, backtest_engine),
            optimize_strategy=OptimizeStrategy(load_price_history, strategy_optimizer),
            orchestrator=TradingOrchestrator(analyze_ticker),
        )

    async def aclose(self) -> None:
        await self.engine.dispose()
