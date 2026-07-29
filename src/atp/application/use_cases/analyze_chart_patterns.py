"""Use case: detect chart patterns on a ticker across one or more timeframes."""

from __future__ import annotations

from collections.abc import Sequence

from atp.application.agents.chart_pattern import ChartPatternAgent
from atp.application.use_cases.load_timeframes import LoadTimeframes
from atp.domain.models.market import BarInterval
from atp.domain.models.patterns import MIN_PATTERN_BARS, ChartPatternReport


class AnalyzeChartPatterns:
    def __init__(self, timeframes: LoadTimeframes, agent: ChartPatternAgent) -> None:
        self._timeframes = timeframes
        self._agent = agent

    async def execute(
        self,
        symbol: str,
        intervals: Sequence[BarInterval] = (BarInterval.DAY_1,),
    ) -> ChartPatternReport:
        histories = await self._timeframes.execute(symbol, intervals, min_bars=MIN_PATTERN_BARS)
        return await self._agent.analyze(histories)
