"""Database schema.

Schema evolution is create-if-missing for now; Alembic migrations arrive once
the schema starts churning (before the strategy/backtesting milestone).
"""

from __future__ import annotations

import structlog
from sqlalchemy import Column, DateTime, Double, MetaData, Table, Text, text
from sqlalchemy.exc import DBAPIError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncEngine

log = structlog.get_logger()

metadata = MetaData()

# The time column is part of the primary key, as TimescaleDB hypertables require.
bars_table = Table(
    "bars",
    metadata,
    Column("symbol", Text, primary_key=True),
    Column("interval", Text, primary_key=True),
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("open", Double, nullable=False),
    Column("high", Double, nullable=False),
    Column("low", Double, nullable=False),
    Column("close", Double, nullable=False),
    Column("volume", Double, nullable=False),
)


async def init_db(engine: AsyncEngine) -> None:
    """Create tables and convert ``bars`` to a hypertable when TimescaleDB is present.

    Runs in two transactions so a missing timescaledb extension (plain
    PostgreSQL) degrades gracefully without rolling back table creation.
    """
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
            await conn.execute(
                text(
                    "SELECT create_hypertable('bars', 'ts', "
                    "if_not_exists => TRUE, migrate_data => TRUE)"
                )
            )
    except (DBAPIError, ProgrammingError):
        log.warning(
            "persistence.timescaledb_unavailable",
            detail="'bars' remains a regular PostgreSQL table",
        )
