"""Use case: compare a ticker against its market benchmark.

Both histories load concurrently and on the same interval — comparing a daily
series to an hourly one would produce numbers that mean nothing.
"""

from __future__ import annotations

import asyncio

from atp.application.agents.market_research import MarketResearchAgent
from atp.application.use_cases.load_price_history import LoadPriceHistory
from atp.domain.models.market import BarInterval
from atp.domain.models.research import MIN_RESEARCH_BARS, MarketResearchReport

DEFAULT_BENCHMARK = "SPY"


class AnalyzeMarketResearch:
    def __init__(
        self,
        history: LoadPriceHistory,
        agent: MarketResearchAgent,
        *,
        benchmark: str = DEFAULT_BENCHMARK,
    ) -> None:
        self._history = history
        self._agent = agent
        self._benchmark = benchmark

    async def execute(
        self,
        symbol: str,
        interval: BarInterval = BarInterval.DAY_1,
        *,
        benchmark: str | None = None,
    ) -> MarketResearchReport:
        benchmark_symbol = (benchmark or self._benchmark).strip().upper()
        symbol_history, benchmark_history = await asyncio.gather(
            self._history.execute(symbol, interval, min_bars=MIN_RESEARCH_BARS),
            self._history.execute(benchmark_symbol, interval, min_bars=MIN_RESEARCH_BARS),
        )
        return await self._agent.analyze(symbol_history, benchmark_history)
