"""Use case: run fundamental analysis on a ticker."""

from __future__ import annotations

from atp.application.agents.fundamental_analysis import FundamentalAnalysisAgent
from atp.domain.models.fundamentals import FundamentalReport
from atp.domain.ports.fundamentals import FundamentalsProvider


class AnalyzeFundamentals:
    def __init__(self, provider: FundamentalsProvider, agent: FundamentalAnalysisAgent) -> None:
        self._provider = provider
        self._agent = agent

    async def execute(self, symbol: str) -> FundamentalReport:
        fundamentals = await self._provider.get_fundamentals(symbol)
        return await self._agent.analyze(fundamentals)
