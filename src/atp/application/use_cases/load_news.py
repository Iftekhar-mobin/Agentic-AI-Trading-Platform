"""Use case: load recent news for a symbol, with a short TTL cache.

The news and sentiment agents are separate specialists that happen to need the
same articles, and the supervisor dispatches them concurrently — without this,
one workflow run would hit a rate-limited vendor twice within milliseconds for
an identical payload. The cache collapses that to a single fetch while keeping
each agent unaware the other exists.

A lock (not just a TTL check) is what makes it work under fan-out: concurrent
callers would otherwise all miss the cache before any of them filled it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import structlog

from atp.domain.models.news import NewsArticle
from atp.domain.ports.news import NewsProvider

log = structlog.get_logger()

DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_LIMIT = 20
DEFAULT_TTL = timedelta(minutes=5)


class LoadNews:
    def __init__(
        self,
        provider: NewsProvider,
        *,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        limit: int = DEFAULT_LIMIT,
        ttl: timedelta = DEFAULT_TTL,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._provider = provider
        self._lookback_days = lookback_days
        self._limit = limit
        self._ttl = ttl
        self._clock = clock
        self._lock = asyncio.Lock()
        self._cache: dict[tuple[str, int, int], tuple[datetime, tuple[NewsArticle, ...]]] = {}

    @property
    def lookback_days(self) -> int:
        return self._lookback_days

    async def execute(
        self,
        symbol: str,
        *,
        lookback_days: int | None = None,
        limit: int | None = None,
    ) -> tuple[NewsArticle, ...]:
        symbol = symbol.strip().upper()
        days = lookback_days if lookback_days is not None else self._lookback_days
        count = limit if limit is not None else self._limit
        key = (symbol, days, count)

        cached = self._fresh(key)
        if cached is not None:
            return cached

        async with self._lock:
            # Another caller may have filled the cache while we waited.
            cached = self._fresh(key)
            if cached is not None:
                log.debug("news.cache_hit_after_wait", symbol=symbol)
                return cached

            since = self._clock() - timedelta(days=days)
            articles = await self._provider.get_news(symbol, limit=count, since=since)
            self._cache[key] = (self._clock(), articles)
            log.info(
                "news.fetched",
                symbol=symbol,
                articles=len(articles),
                lookback_days=days,
            )
            return articles

    def _fresh(self, key: tuple[str, int, int]) -> tuple[NewsArticle, ...] | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        fetched_at, articles = entry
        if self._clock() - fetched_at >= self._ttl:
            return None
        return articles
