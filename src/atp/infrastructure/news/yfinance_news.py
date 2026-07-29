"""Yahoo Finance adapter for the NewsProvider port.

Yahoo has shipped two incompatible news payload shapes: a flat legacy item
(``title``/``publisher``/``providerPublishTime``) and a nested modern one
(``content.title`` / ``content.provider.displayName`` / ``content.pubDate``).
Both are normalized here so the domain never learns about either.

Items that cannot be normalized — no title, no usable timestamp — are dropped
and counted rather than failing the fetch.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import structlog
import yfinance

from atp.domain.models.news import NewsArticle

log = structlog.get_logger()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_text(value: Any) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def _as_timestamp(value: Any) -> datetime | None:
    """Accept both epoch seconds (legacy) and ISO-8601 strings (modern)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


class YFinanceNewsProvider:
    async def get_news(
        self,
        symbol: str,
        *,
        limit: int = 20,
        since: datetime | None = None,
    ) -> tuple[NewsArticle, ...]:
        symbol = symbol.strip().upper()
        # Over-fetch: the vendor count is pre-filter, and `since` may cut deeply.
        items = await asyncio.to_thread(self._download, symbol, max(limit * 2, limit))
        articles = self._to_articles(symbol, items)
        if since is not None:
            articles = tuple(article for article in articles if article.published_at >= since)
        return articles[:limit]

    @staticmethod
    def _download(symbol: str, count: int) -> list[Any]:
        try:
            items = yfinance.Ticker(symbol).get_news(count=count)
        except Exception as exc:  # vendor raises bare Exception subclasses
            log.warning("news.fetch_failed", symbol=symbol, error=str(exc))
            return []
        return items if isinstance(items, list) else []

    @classmethod
    def _to_articles(cls, symbol: str, items: list[Any]) -> tuple[NewsArticle, ...]:
        if not items:
            log.warning("news.empty_response", symbol=symbol)
            return ()

        articles: dict[str, NewsArticle] = {}
        dropped = 0
        for item in items:
            article = cls._normalize(symbol, _as_dict(item))
            if article is None:
                dropped += 1
                continue
            articles.setdefault(article.id, article)  # first win: vendor lists newest first

        if dropped:
            log.warning("news.invalid_items_dropped", symbol=symbol, count=dropped)
        return tuple(
            sorted(articles.values(), key=lambda article: article.published_at, reverse=True)
        )

    @staticmethod
    def _normalize(symbol: str, item: dict[str, Any]) -> NewsArticle | None:
        content = _as_dict(item.get("content"))
        source = content or item

        title = _as_text(source.get("title"))
        published_at = _as_timestamp(
            source.get("pubDate") or source.get("displayTime") or item.get("providerPublishTime")
        )
        if title is None or published_at is None:
            return None

        publisher = _as_text(_as_dict(source.get("provider")).get("displayName")) or _as_text(
            item.get("publisher")
        )
        url = _as_text(_as_dict(source.get("canonicalUrl")).get("url")) or _as_text(
            item.get("link")
        )
        related = source.get("relatedTickers") or item.get("relatedTickers") or []
        symbols = tuple(
            dict.fromkeys(
                [symbol, *(t.strip().upper() for t in related if isinstance(t, str) and t.strip())]
            )
        )
        identifier = (
            _as_text(item.get("id"))
            or _as_text(content.get("id"))
            or _as_text(item.get("uuid"))
            or url
            or f"{symbol}:{published_at.isoformat()}:{title[:64]}"
        )
        return NewsArticle(
            id=identifier,
            title=title,
            summary=_as_text(source.get("summary")) or _as_text(source.get("description")),
            publisher=publisher,
            url=url,
            published_at=published_at,
            symbols=symbols,
        )
