"""TimescaleDB-backed BarRepository.

Connection failures are translated into the domain's
``RepositoryUnavailableError`` so callers can degrade gracefully without
importing SQLAlchemy exception types.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from atp.domain.errors import RepositoryUnavailableError
from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.infrastructure.persistence.schema import bars_table

_PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


class TimescaleBarRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @asynccontextmanager
    async def _connect(self, *, begin: bool = False) -> AsyncIterator[AsyncConnection]:
        try:
            if begin:
                async with self._engine.begin() as conn:
                    yield conn
            else:
                async with self._engine.connect() as conn:
                    yield conn
        except (OSError, OperationalError, InterfaceError) as exc:
            msg = f"database unreachable: {exc}"
            raise RepositoryUnavailableError(msg) from exc

    async def upsert_bars(self, history: PriceHistory) -> int:
        if not history.bars:
            return 0
        rows = [
            {
                "symbol": history.symbol,
                "interval": history.interval.value,
                "ts": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in history.bars
        ]
        stmt = pg_insert(bars_table).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "interval", "ts"],
            set_={column: stmt.excluded[column] for column in _PRICE_COLUMNS},
        )
        async with self._connect(begin=True) as conn:
            await conn.execute(stmt)
        return len(rows)

    async def get_bars(
        self,
        symbol: str,
        interval: BarInterval,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> PriceHistory:
        query = select(bars_table).where(
            bars_table.c.symbol == symbol,
            bars_table.c.interval == interval.value,
        )
        if start is not None:
            query = query.where(bars_table.c.ts >= start)
        if end is not None:
            query = query.where(bars_table.c.ts <= end)
        # Descending + limit keeps the most recent N; rows are reversed after.
        query = query.order_by(bars_table.c.ts.desc())
        if limit is not None:
            query = query.limit(limit)

        async with self._connect() as conn:
            rows = (await conn.execute(query)).all()

        bars = tuple(
            Bar(
                timestamp=row.ts,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
            )
            for row in reversed(rows)
        )
        return PriceHistory(symbol=symbol, interval=interval, bars=bars)

    async def latest_timestamp(self, symbol: str, interval: BarInterval) -> datetime | None:
        query = select(func.max(bars_table.c.ts)).where(
            bars_table.c.symbol == symbol,
            bars_table.c.interval == interval.value,
        )
        async with self._connect() as conn:
            result: datetime | None = (await conn.execute(query)).scalar()
        return result
