"""Use case: run technical analysis on a ticker."""

from __future__ import annotations

from atp.application.agents.technical_analysis import TechnicalAnalysisAgent
from atp.application.use_cases.load_price_history import LoadPriceHistory
from atp.domain.models.analysis import MIN_HISTORY_BARS, TechnicalReport
from atp.domain.models.market import BarInterval


class AnalyzeTicker:
    def __init__(self, history: LoadPriceHistory, agent: TechnicalAnalysisAgent) -> None:
        self._history = history
        self._agent = agent

    async def execute(
        self, symbol: str, interval: BarInterval = BarInterval.DAY_1
    ) -> TechnicalReport:
        history = await self._history.execute(symbol, interval, min_bars=MIN_HISTORY_BARS)
        return await self._agent.analyze(history)
