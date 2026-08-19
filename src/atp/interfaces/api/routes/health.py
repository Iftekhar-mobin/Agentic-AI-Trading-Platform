"""Liveness endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from atp import __version__
from atp.domain.models.llm import ActiveModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    active_model: ActiveModel | None = None
    """Which vendor and model the agents would reason with right now.

    Carried here, on the one endpoint every client already polls, so a console
    can keep the answer on screen without paying for a catalog listing - the
    ``/models`` alternative calls out to the provider and is slow when the
    provider is slow, which is when knowing the active model matters most.
    """


@router.get("/health")
async def health(request: Request) -> HealthResponse:
    container = getattr(request.app.state, "container", None)
    router_ = getattr(container, "llm_router", None)
    return HealthResponse(
        status="ok",
        version=__version__,
        # Liveness must not depend on the LLM stack being wired: a health check
        # that 500s because a model is unconfigured defeats its own purpose.
        active_model=getattr(router_, "active", None),
    )
