"""Composition root: the only place where layers are wired together.

Everything downstream of here depends on ports, so swapping an adapter
(yfinance -> Polygon, Anthropic -> another LLM) is a one-line change in
``build``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy.ext.asyncio import AsyncEngine

from atp.application.agents import (
    ContinuousLearningAgent,
    FundamentalAnalysisAgent,
    NewsAgent,
    SentimentAgent,
    TechnicalAnalysisAgent,
)
from atp.application.orchestration import TradingOrchestrator
from atp.application.use_cases import (
    AnalyzeFundamentals,
    AnalyzeNews,
    AnalyzeSentiment,
    AnalyzeTicker,
    CheckTradeRisk,
    ExecuteTrade,
    GetPortfolio,
    GetPriceHistory,
    JournalTrade,
    LearnFromContext,
    LoadNews,
    LoadPriceHistory,
    OptimizeStrategy,
    RunBacktest,
    SyncMarketData,
)
from atp.domain.ports import (
    BacktestEngine,
    BarRepository,
    Broker,
    EmbeddingModel,
    EpisodicMemory,
    FundamentalsProvider,
    IndicatorEngine,
    LLMClient,
    MarketDataProvider,
    NewsProvider,
    OrderRepository,
    PortfolioRepository,
    SentimentModel,
    StrategyOptimizer,
)
from atp.infrastructure.backtesting import BacktestingPyEngine
from atp.infrastructure.brokers import PaperBroker
from atp.infrastructure.config import (
    MemoryBackend,
    SentimentModelName,
    Settings,
    get_settings,
)
from atp.infrastructure.embeddings import HashingEmbeddingModel
from atp.infrastructure.fundamentals import YFinanceFundamentalsProvider
from atp.infrastructure.indicators import PandasIndicatorEngine
from atp.infrastructure.llm import AnthropicLLMClient
from atp.infrastructure.market_data import YFinanceMarketDataProvider
from atp.infrastructure.memory import JsonEpisodicMemory, QdrantEpisodicMemory
from atp.infrastructure.news import YFinanceNewsProvider
from atp.infrastructure.optimization import OptunaStrategyOptimizer
from atp.infrastructure.persistence import TimescaleBarRepository, create_engine
from atp.infrastructure.persistence.json_orders import JsonOrderRepository
from atp.infrastructure.persistence.json_portfolio import JsonPortfolioRepository
from atp.infrastructure.sentiment import FinBertSentimentModel, LexiconSentimentModel


@dataclass(frozen=True)
class Container:
    settings: Settings
    engine: AsyncEngine
    bar_repository: BarRepository
    market_data: MarketDataProvider
    indicator_engine: IndicatorEngine
    llm: LLMClient
    technical_analysis_agent: TechnicalAnalysisAgent
    fundamentals_provider: FundamentalsProvider
    fundamental_analysis_agent: FundamentalAnalysisAgent
    news_provider: NewsProvider
    sentiment_model: SentimentModel
    news_agent: NewsAgent
    sentiment_agent: SentimentAgent
    embedding_model: EmbeddingModel
    memory: EpisodicMemory
    continuous_learning_agent: ContinuousLearningAgent
    backtest_engine: BacktestEngine
    strategy_optimizer: StrategyOptimizer
    sync_market_data: SyncMarketData
    get_price_history: GetPriceHistory
    load_price_history: LoadPriceHistory
    load_news: LoadNews
    analyze_ticker: AnalyzeTicker
    analyze_fundamentals: AnalyzeFundamentals
    analyze_news: AnalyzeNews
    analyze_sentiment: AnalyzeSentiment
    learn_from_context: LearnFromContext
    journal_trade: JournalTrade
    run_backtest: RunBacktest
    optimize_strategy: OptimizeStrategy
    portfolio_repository: PortfolioRepository
    get_portfolio: GetPortfolio
    check_trade_risk: CheckTradeRisk
    broker: Broker
    order_repository: OrderRepository
    execute_trade: ExecuteTrade
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

        fundamentals_provider = YFinanceFundamentalsProvider()
        fundamental_analysis_agent = FundamentalAnalysisAgent(llm)
        news_provider = YFinanceNewsProvider()
        sentiment_model = cls._build_sentiment_model(settings)
        news_agent = NewsAgent(llm)
        sentiment_agent = SentimentAgent(
            llm, sentiment_model, half_life_days=settings.sentiment.half_life_days
        )
        load_news = LoadNews(
            news_provider,
            lookback_days=settings.news.lookback_days,
            limit=settings.news.limit,
            ttl=timedelta(seconds=settings.news.cache_ttl_seconds),
        )
        analyze_fundamentals = AnalyzeFundamentals(
            fundamentals_provider, fundamental_analysis_agent
        )
        analyze_news = AnalyzeNews(load_news, news_agent)
        analyze_sentiment = AnalyzeSentiment(load_news, sentiment_agent)

        embedding_model = HashingEmbeddingModel(settings.memory.embedding_dimensions)
        memory = cls._build_memory(settings, embedding_model)
        continuous_learning_agent = ContinuousLearningAgent(llm)
        learn_from_context = LearnFromContext(
            load_price_history,
            memory,
            continuous_learning_agent,
            recall_limit=settings.memory.recall_limit,
        )

        backtest_engine = BacktestingPyEngine()
        strategy_optimizer = OptunaStrategyOptimizer(backtest_engine)
        portfolio_repository = JsonPortfolioRepository(
            settings.data_dir / "portfolio.json",
            starting_cash=settings.paper_starting_cash,
        )
        get_portfolio = GetPortfolio(portfolio_repository, market_data)
        check_trade_risk = CheckTradeRisk(get_portfolio, market_data, settings.risk)
        broker = PaperBroker(market_data, slippage_bps=settings.execution.slippage_bps)
        order_repository = JsonOrderRepository(settings.data_dir / "orders.jsonl")
        return cls(
            settings=settings,
            engine=engine,
            bar_repository=bar_repository,
            market_data=market_data,
            indicator_engine=indicator_engine,
            llm=llm,
            technical_analysis_agent=technical_analysis_agent,
            fundamentals_provider=fundamentals_provider,
            fundamental_analysis_agent=fundamental_analysis_agent,
            news_provider=news_provider,
            sentiment_model=sentiment_model,
            news_agent=news_agent,
            sentiment_agent=sentiment_agent,
            backtest_engine=backtest_engine,
            strategy_optimizer=strategy_optimizer,
            sync_market_data=SyncMarketData(market_data, bar_repository),
            get_price_history=GetPriceHistory(bar_repository),
            load_price_history=load_price_history,
            load_news=load_news,
            analyze_ticker=analyze_ticker,
            analyze_fundamentals=analyze_fundamentals,
            analyze_news=analyze_news,
            analyze_sentiment=analyze_sentiment,
            embedding_model=embedding_model,
            memory=memory,
            continuous_learning_agent=continuous_learning_agent,
            learn_from_context=learn_from_context,
            journal_trade=JournalTrade(memory),
            run_backtest=RunBacktest(load_price_history, backtest_engine),
            optimize_strategy=OptimizeStrategy(load_price_history, strategy_optimizer),
            portfolio_repository=portfolio_repository,
            get_portfolio=get_portfolio,
            check_trade_risk=check_trade_risk,
            broker=broker,
            order_repository=order_repository,
            execute_trade=ExecuteTrade(
                check_trade_risk, broker, portfolio_repository, order_repository
            ),
            orchestrator=TradingOrchestrator(
                analyze_ticker,
                analyze_fundamentals,
                analyze_news,
                analyze_sentiment,
                learn_from_context,
            ),
        )

    @staticmethod
    def _build_memory(settings: Settings, embeddings: EmbeddingModel) -> EpisodicMemory:
        """Pick the episodic memory backend.

        The Qdrant client is constructed lazily-ish: creating it opens no
        connection, so choosing this backend costs nothing until the first
        recall or write.
        """
        if settings.memory.backend is MemoryBackend.QDRANT:
            from qdrant_client import AsyncQdrantClient

            return QdrantEpisodicMemory(
                AsyncQdrantClient(url=settings.memory.qdrant_url),
                embeddings,
                collection=settings.memory.qdrant_collection,
            )
        return JsonEpisodicMemory(settings.data_dir / "memory.jsonl", embeddings)

    @staticmethod
    def _build_sentiment_model(settings: Settings) -> SentimentModel:
        """Pick the sentiment classifier. FinBERT is lazy: selecting it here does
        not import torch, so an unused adapter costs nothing at startup."""
        if settings.sentiment.model is SentimentModelName.FINBERT:
            return FinBertSentimentModel(settings.sentiment.finbert_model_name)
        return LexiconSentimentModel()

    async def aclose(self) -> None:
        await self.engine.dispose()
