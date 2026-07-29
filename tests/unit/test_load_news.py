"""Tests for the shared news loader and its fan-out cache."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from factories import make_articles

from atp.application.use_cases import LoadNews
from atp.domain.models.news import NewsArticle

NOW = datetime(2026, 7, 22, 12, tzinfo=UTC)


class CountingNewsProvider:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.calls: list[tuple[str, int, datetime | None]] = []
        self._delay = delay

    async def get_news(
        self,
        symbol: str,
        *,
        limit: int = 20,
        since: datetime | None = None,
    ) -> tuple[NewsArticle, ...]:
        self.calls.append((symbol, limit, since))
        if self._delay:
            await asyncio.sleep(self._delay)
        return make_articles(2, symbol=symbol)


class Clock:
    """A manually advanced clock, so TTL expiry is tested without sleeping."""

    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


async def test_symbol_is_normalized_and_lookback_becomes_a_since_bound() -> None:
    provider = CountingNewsProvider()
    loader = LoadNews(provider, lookback_days=7, limit=15, clock=Clock())

    articles = await loader.execute(" aapl ")

    assert len(articles) == 2
    symbol, limit, since = provider.calls[0]
    assert symbol == "AAPL"
    assert limit == 15
    assert since == NOW - timedelta(days=7)


async def test_repeat_calls_within_the_ttl_hit_the_cache() -> None:
    provider = CountingNewsProvider()
    loader = LoadNews(provider, clock=Clock(), ttl=timedelta(minutes=5))

    await loader.execute("AAPL")
    await loader.execute("AAPL")

    assert len(provider.calls) == 1


async def test_concurrent_callers_collapse_to_one_fetch() -> None:
    """This is the fan-out case: news and sentiment agents in the same superstep."""
    provider = CountingNewsProvider(delay=0.05)
    loader = LoadNews(provider, clock=Clock())

    results = await asyncio.gather(loader.execute("AAPL"), loader.execute("AAPL"))

    assert len(provider.calls) == 1
    assert results[0] == results[1]


async def test_cache_expires_after_the_ttl() -> None:
    clock = Clock()
    provider = CountingNewsProvider()
    loader = LoadNews(provider, clock=clock, ttl=timedelta(minutes=5))

    await loader.execute("AAPL")
    clock.advance(timedelta(minutes=5))
    await loader.execute("AAPL")

    assert len(provider.calls) == 2


async def test_zero_ttl_disables_caching() -> None:
    provider = CountingNewsProvider()
    loader = LoadNews(provider, clock=Clock(), ttl=timedelta(0))

    await loader.execute("AAPL")
    await loader.execute("AAPL")

    assert len(provider.calls) == 2


async def test_different_symbols_and_windows_are_cached_separately() -> None:
    provider = CountingNewsProvider()
    loader = LoadNews(provider, clock=Clock())

    await loader.execute("AAPL")
    await loader.execute("MSFT")
    await loader.execute("AAPL", lookback_days=30)
    await loader.execute("AAPL", limit=5)

    assert len(provider.calls) == 4


def test_lookback_days_is_exposed_for_reporting() -> None:
    assert LoadNews(CountingNewsProvider(), lookback_days=14).lookback_days == 14
