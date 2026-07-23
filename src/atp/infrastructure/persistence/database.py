"""Async database engine management."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def create_engine(dsn: str, *, echo: bool = False) -> AsyncEngine:
    return create_async_engine(dsn, echo=echo, pool_pre_ping=True)
