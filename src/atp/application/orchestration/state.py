"""Shared state flowing through the supervisor graph.

The state is the single source of truth for a workflow run: agents read from
it and return partial updates; LangGraph merges those updates. List fields use
the ``operator.add`` reducer so the analysis agents, which fan out
concurrently, can append without overwriting each other.
"""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from atp.domain.models.analysis import TechnicalReport
from atp.domain.models.fundamentals import FundamentalReport
from atp.domain.models.market import BarInterval
from atp.domain.models.memory import LearningReport
from atp.domain.models.news import NewsReport
from atp.domain.models.patterns import ChartPatternReport
from atp.domain.models.research import MarketResearchReport
from atp.domain.models.sentiment import SentimentReport


class AgentFailure(BaseModel):
    """A recorded, non-fatal agent failure (the graph routes around it)."""

    agent: str
    error: str


class AgentStep(BaseModel):
    """One agent's execution, recorded so a caller can see how the work was done.

    ``failures`` says what went wrong; this says what happened at all — which
    agents ran, in what order, how long each took, and what it concluded. That
    is the difference between "the news agent failed" and "the news agent was
    dispatched in phase one, took 31s, and timed out against the vendor", and it
    is what makes a slow or partial run diagnosable from the outside.

    Timings are wall-clock and the analysis agents run concurrently, so the
    durations overlap and will not sum to the total.
    """

    agent: str
    phase: Literal["analysis", "feedback"]
    status: Literal["ok", "failed"]
    started_at: datetime
    duration_ms: float
    detail: str = ""
    """What it concluded, or why it failed."""


class TradingState(BaseModel):
    # Request
    symbol: str
    intervals: tuple[BarInterval, ...] = Field(
        default=(BarInterval.DAY_1,),
        min_length=1,
        description="Timeframes to analyze, highest first (multi-timeframe analysis)",
    )
    requested_agents: tuple[str, ...] = Field(
        default=(),
        description="Agents to dispatch; empty means every analysis agent",
    )

    # Artifacts produced by agents (one field per agent family, grows per milestone)
    technical_report: TechnicalReport | None = None
    chart_pattern_report: ChartPatternReport | None = None
    market_research_report: MarketResearchReport | None = None
    fundamental_report: FundamentalReport | None = None
    news_report: NewsReport | None = None
    sentiment_report: SentimentReport | None = None

    # Produced by the feedback pool, after the analysis pool has run
    learning_report: LearningReport | None = None

    # Bookkeeping
    completed: Annotated[list[str], operator.add] = Field(default_factory=list)
    failures: Annotated[list[AgentFailure], operator.add] = Field(default_factory=list)
    steps: Annotated[list[AgentStep], operator.add] = Field(
        default_factory=list,
        description="Execution trace: what ran, in what order, and for how long",
    )

    @property
    def interval(self) -> BarInterval:
        """The primary (highest) timeframe — what single-timeframe callers mean."""
        return self.intervals[0]

    @property
    def has_analysis(self) -> bool:
        """True when at least one analysis agent produced an artifact.

        This gates the learning phase: reflecting on nothing is not useful.
        """
        return any(
            report is not None
            for report in (
                self.technical_report,
                self.chart_pattern_report,
                self.market_research_report,
                self.fundamental_report,
                self.news_report,
                self.sentiment_report,
            )
        )

    @property
    def has_report(self) -> bool:
        """True when at least one agent of any pool produced an artifact."""
        return self.has_analysis or self.learning_report is not None
