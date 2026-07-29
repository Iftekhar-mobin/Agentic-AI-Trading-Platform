"""Use case: run news analysis on a ticker."""

from __future__ import annotations

from atp.application.agents.news_analysis import NewsAgent
from atp.application.use_cases.load_news import LoadNews
from atp.domain.models.news import NewsReport


class AnalyzeNews:
    def __init__(self, news: LoadNews, agent: NewsAgent) -> None:
        self._news = news
        self._agent = agent

    async def execute(self, symbol: str) -> NewsReport:
        articles = await self._news.execute(symbol)
        return await self._agent.analyze(symbol, articles, lookback_days=self._news.lookback_days)
