"""Cross-cutting HTTP middleware: correlation, timing, security headers.

Correlation is the important one. Every log line the platform emits during a
request — including the ones from deep inside agents and adapters — carries the
same ``request_id``, because it is bound into ``structlog.contextvars`` here and
merged by the logging configuration. A trader asking "why did it recommend
that?" can be answered from the logs of one request rather than by guessing
which interleaved lines belong together.

An inbound ``X-Request-ID`` is honoured so a trace started at the dashboard or a
gateway stays one trace end to end, and it is always echoed back.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from atp.infrastructure.observability import bind_trace_context

log = structlog.get_logger()

REQUEST_ID_HEADER = "X-Request-ID"

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
}


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Bind a correlation id for the life of the request and log its outcome."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        # Also on state, so error handlers can echo it without re-deriving it.
        request.state.request_id = request_id
        # clear_contextvars first: worker tasks are reused across requests, and
        # inherited context would attribute one request's logs to another.
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        # Ids of the active span, so logs and traces can be joined on either side.
        bind_trace_context()
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception(
                "http.request_failed",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            raise
        finally:
            structlog.contextvars.unbind_contextvars("method", "path")

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        log.info("http.request", status_code=response.status_code, duration_ms=duration_ms)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Conservative defaults; HSTS only when the deployment is actually on TLS."""

    def __init__(self, app: ASGIApp, *, hsts: bool = False) -> None:
        super().__init__(app)
        self._hsts = hsts

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        if self._hsts:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response
