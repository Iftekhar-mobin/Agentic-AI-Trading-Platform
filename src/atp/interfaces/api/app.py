"""FastAPI application factory.

Middleware order matters and is not arbitrary. Starlette runs middleware
outermost-first, and each is added inside the previous, so the list below runs:

    security headers -> correlation id -> rate limit -> CORS -> routes

Correlation wraps the rate limiter so a 429 still gets a request id and a log
line — rejections are exactly what someone will need to investigate later. The
limiter sits outside the routes so a throttled request never reaches a
dependency, an LLM, or a broker.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from atp import __version__
from atp.composition import Container
from atp.infrastructure.config import Environment
from atp.infrastructure.observability import configure_tracing, instrument_app
from atp.interfaces.api.errors import register_error_handlers
from atp.interfaces.api.middleware import RequestContextMiddleware, SecurityHeadersMiddleware
from atp.interfaces.api.rate_limit import RateLimiter, RateLimitMiddleware
from atp.interfaces.api.routes import (
    analysis,
    consensus,
    health,
    memory,
    models,
    portfolio,
    trading,
)

log = structlog.get_logger()

DESCRIPTION = """\
Multi-agent trading assistant. Every recommendation carries its reasoning,
evidence, calibrated confidence, and the conditions that would invalidate it.

Authenticate with `Authorization: Bearer <api-key>`. Read endpoints need the
`read` scope, `POST /trade` needs `trade`, and changing the active language
model needs `admin`.
"""


def create_app(container: Container | None = None) -> FastAPI:
    """Build the app. Pass a prebuilt ``container`` to inject test doubles."""
    built = container or Container.build()
    settings = built.settings

    tracing_enabled = configure_tracing(settings.observability)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = built
        if not settings.api.auth_required:
            log.warning(
                "api.authentication_disabled",
                environment=settings.environment.value,
                detail="no ATP_API__KEYS configured; every request is anonymous",
            )
        log.info(
            "api.started",
            version=__version__,
            environment=settings.environment.value,
            trading_mode=settings.trading_mode.value,
            auth=settings.api.auth_required,
            llm=str(built.llm_router.active),
            tracing=tracing_enabled,
            rate_limit_per_minute=settings.api.rate_limit_per_minute,
        )
        yield
        await app.state.container.aclose()

    app = FastAPI(
        title="Agentic AI Trading Platform",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )
    app.state.container = built

    if settings.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.api.cors_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        )
    app.add_middleware(RateLimitMiddleware, limiter=RateLimiter(settings.api.rate_limit_per_minute))
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        SecurityHeadersMiddleware,
        hsts=settings.environment is Environment.PRODUCTION,
    )

    register_error_handlers(app)
    for router in (health, analysis, consensus, portfolio, memory, trading, models):
        app.include_router(router.router)

    instrument_app(app)
    return app
