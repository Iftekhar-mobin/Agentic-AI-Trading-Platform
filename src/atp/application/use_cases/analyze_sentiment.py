"""Use case: run sentiment analysis on a ticker's news flow."""

from __future__ import annotations

from atp.application.agents.sentiment_analysis import SentimentAgent
from atp.application.use_cases.load_news import LoadNews
from atp.domain.models.sentiment import SentimentReport


class AnalyzeSentiment:
    def __init__(self, news: LoadNews, agent: SentimentAgent) -> None:
        self._news = news
        self._agent = agent

    async def execute(self, symbol: str) -> SentimentReport:
        articles = await self._news.execute(symbol)
        return await self._agent.analyze(symbol, articles)
