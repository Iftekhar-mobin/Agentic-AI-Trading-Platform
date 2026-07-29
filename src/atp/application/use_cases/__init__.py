"""Application use cases."""

from atp.application.use_cases.analyze_fundamentals import AnalyzeFundamentals
from atp.application.use_cases.analyze_news import AnalyzeNews
from atp.application.use_cases.analyze_sentiment import AnalyzeSentiment
from atp.application.use_cases.analyze_ticker import AnalyzeTicker
from atp.application.use_cases.check_trade_risk import CheckTradeRisk
from atp.application.use_cases.execute_trade import ExecuteTrade
from atp.application.use_cases.get_portfolio import GetPortfolio
from atp.application.use_cases.get_price_history import GetPriceHistory
from atp.application.use_cases.journal_trade import JournalTrade
from atp.application.use_cases.learn_from_context import LearnFromContext
from atp.application.use_cases.load_news import LoadNews
from atp.application.use_cases.load_price_history import LoadPriceHistory
from atp.application.use_cases.optimize_strategy import OptimizeStrategy
from atp.application.use_cases.run_backtest import RunBacktest
from atp.application.use_cases.sync_market_data import SyncMarketData, SyncResult

__all__ = [
    "AnalyzeFundamentals",
    "AnalyzeNews",
    "AnalyzeSentiment",
    "AnalyzeTicker",
    "CheckTradeRisk",
    "ExecuteTrade",
    "GetPortfolio",
    "GetPriceHistory",
    "JournalTrade",
    "LearnFromContext",
    "LoadNews",
    "LoadPriceHistory",
    "OptimizeStrategy",
    "RunBacktest",
    "SyncMarketData",
    "SyncResult",
]
