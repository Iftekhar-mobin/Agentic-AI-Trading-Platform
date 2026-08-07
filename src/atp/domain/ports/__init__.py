"""Ports: abstract interfaces the application layer depends on.

Infrastructure adapters implement these; the application never imports
infrastructure directly.
"""

from atp.domain.ports.backtesting import BacktestEngine
from atp.domain.ports.broker import Broker
from atp.domain.ports.embeddings import EmbeddingModel
from atp.domain.ports.fundamentals import FundamentalsProvider
from atp.domain.ports.indicators import IndicatorEngine
from atp.domain.ports.llm import LLMClient
from atp.domain.ports.market_data import MarketDataProvider
from atp.domain.ports.memory import EpisodicMemory
from atp.domain.ports.model_catalog import ModelCatalog
from atp.domain.ports.news import NewsProvider
from atp.domain.ports.optimization import StrategyOptimizer
from atp.domain.ports.orders import OrderRepository
from atp.domain.ports.portfolio import PortfolioRepository
from atp.domain.ports.repositories import BarRepository
from atp.domain.ports.sentiment import SentimentModel
from atp.domain.ports.signals import SignalRepository

__all__ = [
    "BacktestEngine",
    "BarRepository",
    "Broker",
    "EmbeddingModel",
    "EpisodicMemory",
    "FundamentalsProvider",
    "IndicatorEngine",
    "LLMClient",
    "MarketDataProvider",
    "ModelCatalog",
    "NewsProvider",
    "OrderRepository",
    "PortfolioRepository",
    "SentimentModel",
    "SignalRepository",
    "StrategyOptimizer",
]
