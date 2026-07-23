"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from atp import __version__
from atp.composition import Container
from atp.interfaces.api.routes import analysis, health


def create_app(container: Container | None = None) -> FastAPI:
    """Build the app. Pass a prebuilt ``container`` to inject test doubles."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or Container.build()
        yield
        await app.state.container.aclose()

    app = FastAPI(
        title="Agentic AI Trading Platform",
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(analysis.router)
    return app
