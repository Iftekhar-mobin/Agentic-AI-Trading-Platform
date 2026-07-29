"""Episodic memory recall endpoint (read scope)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from atp.domain.models.memory import EpisodeKind, EpisodeMatch
from atp.interfaces.api.security import RequiresRead

router = APIRouter(tags=["memory"], dependencies=[RequiresRead])


class RecallResponse(BaseModel):
    symbol: str
    query: str
    matches: list[EpisodeMatch]


@router.get("/memory/{symbol}", response_model=RecallResponse)
async def recall(
    request: Request,
    symbol: str,
    query: Annotated[str | None, Query(description="Defaults to the symbol itself")] = None,
    kind: Annotated[EpisodeKind | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> RecallResponse:
    """Recall journalled episodes for a symbol, ranked by similarity."""
    symbol = symbol.strip().upper()
    text = query or symbol
    matches = await request.app.state.container.memory.recall(
        text, symbol=symbol, kinds=[kind] if kind else None, limit=limit
    )
    return RecallResponse(symbol=symbol, query=text, matches=list(matches))
