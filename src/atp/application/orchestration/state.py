"""Shared state flowing through the supervisor graph.

The state is the single source of truth for a workflow run: agents read from
it and return partial updates; LangGraph merges those updates. List fields use
the ``operator.add`` reducer so the analysis agents, which fan out
concurrently, can append without overwriting each other.
"""

from __future__ import annotations

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from atp.domain.models.analysis import TechnicalReport
from atp.domain.models.fundamentals import FundamentalReport
from atp.domain.models.market import BarInterval
from atp.domain.models.memory import LearningReport
from atp.domain.models.news import NewsReport
from atp.domain.models.sentiment import SentimentReport


class AgentFailure(BaseModel):
    """A recorded, non-fatal agent failure (the graph routes around it)."""

    agent: str
    error: str


class TradingState(BaseModel):
    # Request
    symbol: str
    interval: BarInterval = BarInterval.DAY_1
    requested_agents: tuple[str, ...] = Field(
        default=(),
        description="Agents to dispatch; empty means every analysis agent",
    )

    # Artifacts produced by agents (one field per agent family, grows per milestone)
    technical_report: TechnicalReport | None = None
    fundamental_report: FundamentalReport | None = None
    news_report: NewsReport | None = None
    sentiment_report: SentimentReport | None = None

    # Produced by the feedback pool, after the analysis pool has run
    learning_report: LearningReport | None = None

    # Bookkeeping
    completed: Annotated[list[str], operator.add] = Field(default_factory=list)
    failures: Annotated[list[AgentFailure], operator.add] = Field(default_factory=list)

    @property
    def has_analysis(self) -> bool:
        """True when at least one analysis agent produced an artifact.

        This gates the learning phase: reflecting on nothing is not useful.
        """
        return any(
            report is not None
            for report in (
                self.technical_report,
                self.fundamental_report,
                self.news_report,
                self.sentiment_report,
            )
        )

    @property
    def has_report(self) -> bool:
        """True when at least one agent of any pool produced an artifact."""
        return self.has_analysis or self.learning_report is not None
