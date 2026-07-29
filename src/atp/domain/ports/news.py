"""Port for news sources."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from atp.domain.models.news import NewsArticle


class NewsProvider(Protocol):
    """Fetches recent articles for one symbol.

    Implementations return articles newest-first, deduplicated by id, with
    timezone-aware timestamps. Vendor items that cannot be normalized are
    dropped and logged rather than failing the whole fetch.
    """

    async def get_news(
        self,
        symbol: str,
        *,
        limit: int = 20,
        since: datetime | None = None,
    ) -> tuple[NewsArticle, ...]: ...
