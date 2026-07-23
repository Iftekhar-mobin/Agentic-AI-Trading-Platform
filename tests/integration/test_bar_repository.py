"""Integration tests for TimescaleBarRepository against a real database.

Skipped automatically when TimescaleDB is unreachable (``docker compose up -d``
starts it). CI and local runs stay green either way.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.infrastructure.config import Settings
from atp.infrastructure.persistence import TimescaleBarRepository, create_engine, init_db

pytestmark = pytest.mark.integration

SYMBOL = "TEST-ATP"


def make_history(days: int, *, base_close: float = 103.0) -> PriceHistory:
    start = datetime(2026, 1, 5, tzinfo=UTC)
    bars = tuple(
        Bar(
            timestamp=start + timedelta(days=offset),
            open=100.0,
            high=110.0,
            low=99.0,
            close=base_close + offset,
            volume=1_000.0,
        )
        for offset in range(days)
    )
    return PriceHistory(symbol=SYMBOL, interval=BarInterval.DAY_1, bars=bars)


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    engine = create_engine(Settings(_env_file=None).database.dsn)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        await engine.dispose()
        pytest.skip("TimescaleDB unreachable — run `docker compose up -d`")
    await init_db(engine)
    yield engine
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM bars WHERE symbol = :symbol"), {"symbol": SYMBOL})
    await engine.dispose()


async def test_upsert_then_read_round_trip(engine: AsyncEngine) -> None:
    repository = TimescaleBarRepository(engine)
    stored = await repository.upsert_bars(make_history(3))
    assert stored == 3

    history = await repository.get_bars(SYMBOL, BarInterval.DAY_1)
    assert len(history) == 3
    assert [bar.close for bar in history.bars] == [103.0, 104.0, 105.0]


async def test_upsert_is_idempotent_and_updates_prices(engine: AsyncEngine) -> None:
    repository = TimescaleBarRepository(engine)
    await repository.upsert_bars(make_history(3))
    await repository.upsert_bars(make_history(3, base_close=200.0))

    history = await repository.get_bars(SYMBOL, BarInterval.DAY_1)
    assert len(history) == 3  # no duplicates
    assert [bar.close for bar in history.bars] == [200.0, 201.0, 202.0]


async def test_limit_returns_most_recent_bars_ascending(engine: AsyncEngine) -> None:
    repository = TimescaleBarRepository(engine)
    await repository.upsert_bars(make_history(5))

    history = await repository.get_bars(SYMBOL, BarInterval.DAY_1, limit=2)
    assert [bar.close for bar in history.bars] == [106.0, 107.0]


async def test_latest_timestamp(engine: AsyncEngine) -> None:
    repository = TimescaleBarRepository(engine)
    assert await repository.latest_timestamp(SYMBOL, BarInterval.DAY_1) is None

    await repository.upsert_bars(make_history(2))
    latest = await repository.latest_timestamp(SYMBOL, BarInterval.DAY_1)
    assert latest == datetime(2026, 1, 6, tzinfo=UTC)
