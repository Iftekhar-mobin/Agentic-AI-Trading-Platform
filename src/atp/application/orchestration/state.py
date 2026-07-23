"""Shared state flowing through the supervisor graph.

The state is the single source of truth for a workflow run: agents read from
it and return partial updates; LangGraph merges those updates. List fields use
the ``operator.add`` reducer so concurrent agents can append without
overwriting each other once the graph fans out.
"""

from __future__ import annotations

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from atp.domain.models.analysis import TechnicalReport
from atp.domain.models.market import BarInterval


class AgentFailure(BaseModel):
    """A recorded, non-fatal agent failure (the graph routes around it)."""

    agent: str
    error: str


class TradingState(BaseModel):
    # Request
    symbol: str
    interval: BarInterval = BarInterval.DAY_1

    # Artifacts produced by agents (one field per agent family, grows per milestone)
    technical_report: TechnicalReport | None = None

    # Bookkeeping
    completed: Annotated[list[str], operator.add] = Field(default_factory=list)
    failures: Annotated[list[AgentFailure], operator.add] = Field(default_factory=list)
