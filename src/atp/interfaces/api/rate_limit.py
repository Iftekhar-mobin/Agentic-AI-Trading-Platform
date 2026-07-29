"""Per-credential request rate limiting.

A fixed-window counter held in process memory. That is honestly a starter
implementation: it does not coordinate across replicas, and a caller can burst
at a window boundary. It is here because the endpoints behind it each cost real
money — an unbounded ``/analysis`` loop is five LLM calls per iteration — and
"no limit at all" is the worse failure.

Redis is already a dependency of this platform and is where this belongs the
moment there is more than one process; the shape of the code does not change,
only where the counter lives.

Callers are identified by API key name when authenticated, and by client
address otherwise, so one noisy key cannot exhaust everyone else's budget.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

import structlog
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

log = structlog.get_logger()

WINDOW_SECONDS = 60.0

EXEMPT_PATHS = frozenset({"/health"})
"""Liveness probes must never be throttled — that turns a burst into an outage."""


class RateLimiter:
    """Fixed-window counter keyed by caller."""

    def __init__(
        self,
        limit_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit_per_minute
        self._clock = clock
        self._windows: dict[str, tuple[float, int]] = {}

    @property
    def enabled(self) -> bool:
        return self._limit > 0

    def check(self, caller: str) -> tuple[bool, int]:
        """Count this request. Returns (allowed, requests remaining in the window)."""
        if not self.enabled:
            return True, 0

        now = self._clock()
        window_start, count = self._windows.get(caller, (now, 0))
        if now - window_start >= WINDOW_SECONDS:
            window_start, count = now, 0

        count += 1
        self._windows[caller] = (window_start, count)
        return count <= self._limit, max(self._limit - count, 0)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, limiter: RateLimiter) -> None:
        super().__init__(app)
        self._limiter = limiter

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not self._limiter.enabled or request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        caller = self._caller(request)
        allowed, remaining = self._limiter.check(caller)
        if not allowed:
            log.warning("api.rate_limited", caller=caller, path=request.url.path)
            return JSONResponse(
                status_code=429,
                content={"detail": "rate limit exceeded"},
                headers={"Retry-After": str(int(WINDOW_SECONDS))},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    @staticmethod
    def _caller(request: Request) -> str:
        """Identify by credential when present; the raw key is never stored or logged."""
        authorization = request.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            # A hash keeps distinct keys in distinct buckets without holding the secret.
            return f"key:{hash(authorization[7:].strip())}"
        client = request.client
        return f"ip:{client.host}" if client else "ip:unknown"
