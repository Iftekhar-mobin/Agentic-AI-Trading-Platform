"""Domain-error to HTTP-status mapping.

Registered once on the app so routes stay free of try/except: a handler raises
the domain error that actually describes what went wrong, and the status code
is decided here, in one table, consistently across every endpoint.

The distinctions matter to a client. "You asked about a ticker with three bars
of history" (422, do not retry) is not "Qdrant is restarting" (503, retry
shortly) and neither is "the model returned unparseable output" (502).

Responses carry the correlation id, so a user reporting an error hands over the
one string needed to find every log line for that request.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from atp.domain.errors import (
    DomainError,
    ExecutionError,
    InsufficientDataError,
    LLMGenerationError,
    ModelUnavailableError,
    OptimizationError,
    RepositoryUnavailableError,
)

log = structlog.get_logger()

# Most specific first: the lookup walks the exception's MRO.
STATUS_BY_ERROR: tuple[tuple[type[DomainError], int], ...] = (
    (InsufficientDataError, 422),
    (OptimizationError, 422),
    (ExecutionError, 409),
    (LLMGenerationError, 502),
    (RepositoryUnavailableError, 503),
    (ModelUnavailableError, 503),
    (DomainError, 500),
)


def status_for(error: DomainError) -> int:
    for error_type, status_code in STATUS_BY_ERROR:
        if isinstance(error, error_type):
            return status_code
    return 500


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def _handle_domain_error(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, DomainError)
        status_code = status_for(exc)
        request_id = getattr(request.state, "request_id", "")
        log.warning(
            "api.domain_error",
            error_type=type(exc).__name__,
            status_code=status_code,
            error=str(exc),
        )
        return JSONResponse(
            status_code=status_code,
            content={
                "detail": str(exc),
                "error_type": type(exc).__name__,
                "request_id": request_id,
            },
        )
