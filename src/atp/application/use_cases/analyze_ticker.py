"""Use case: run technical analysis on a ticker across one or more timeframes."""

from __future__ import annotations

from collections.abc import Sequence

from atp.application.agents.technical_analysis import TechnicalAnalysisAgent
from atp.application.use_cases.load_timeframes import LoadTimeframes
from atp.domain.models.analysis import MIN_HISTORY_BARS, TechnicalReport
from atp.domain.models.market import BarInterval


class AnalyzeTicker:
    def __init__(self, timeframes: LoadTimeframes, agent: TechnicalAnalysisAgent) -> None:
        self._timeframes = timeframes
        self._agent = agent

    async def execute(
        self,
        symbol: str,
        intervals: Sequence[BarInterval] = (BarInterval.DAY_1,),
    ) -> TechnicalReport:
        histories = await self._timeframes.execute(symbol, intervals, min_bars=MIN_HISTORY_BARS)
        return await self._agent.analyze(histories)
