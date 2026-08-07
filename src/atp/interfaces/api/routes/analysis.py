"""Analysis endpoints: run agent workflows through the orchestrator.

POST (not GET) because a run has side effects and real cost: market data
fetches, vendor calls and one LLM round-trip per agent.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from atp.application.orchestration import (
    ALL_AGENTS,
    AgentStep,
    TradingOrchestrator,
    UnknownAgentError,
)
from atp.domain.models.analysis import TechnicalReport
from atp.domain.models.fundamentals import FundamentalReport
from atp.domain.models.llm import ActiveModel
from atp.domain.models.market import BarInterval
from atp.domain.models.memory import LearningReport
from atp.domain.models.news import NewsReport
from atp.domain.models.patterns import ChartPatternReport
from atp.domain.models.research import MarketResearchReport
from atp.domain.models.sentiment import SentimentReport
from atp.interfaces.api.schemas import AgentFailureSchema
from atp.interfaces.api.security import RequiresRead

router = APIRouter(tags=["analysis"], dependencies=[RequiresRead])


class AnalysisRequest(BaseModel):
    """``intervals`` drives multi-timeframe analysis.

    Pass several (``["1d", "4h", "1h"]``) and the technical and chart-pattern
    agents reason about all of them at once, weighting the highest as context.
    They are sorted highest-first regardless of the order given.
    """

    symbol: str = Field(min_length=1, max_length=12, examples=["AAPL"])
    intervals: list[BarInterval] = Field(
        default_factory=lambda: [BarInterval.DAY_1],
        min_length=1,
        max_length=5,
        examples=[["1d", "4h", "1h"]],
    )
    agents: list[str] = Field(
        default_factory=list,
        description=f"Subset of {list(ALL_AGENTS)}; empty runs all of them",
        examples=[["technical_analysis", "chart_pattern"]],
    )


class AnalysisResponse(BaseModel):
    """Reports are individually optional: an agent that fails is reported in
    ``failures`` while the others still return their work."""

    symbol: str
    intervals: list[BarInterval]
    requested_agents: list[str]
    completed_agents: list[str]
    failures: list[AgentFailureSchema] = Field(default_factory=list)
    steps: list[AgentStep] = Field(
        default_factory=list,
        description="Execution trace: which agents ran, in what order, and for how long",
    )
    active_model: ActiveModel | None = Field(
        default=None,
        description="The provider and model that produced this reasoning",
    )
    duration_ms: float = Field(default=0.0, description="Total wall-clock time for the run")
    technical_report: TechnicalReport | None = None
    chart_pattern_report: ChartPatternReport | None = None
    market_research_report: MarketResearchReport | None = None
    fundamental_report: FundamentalReport | None = None
    news_report: NewsReport | None = None
    sentiment_report: SentimentReport | None = None
    learning_report: LearningReport | None = None


def _orchestrator(request: Request) -> TradingOrchestrator:
    return request.app.state.container.orchestrator  # type: ignore[no-any-return]


@router.post("/analysis", response_model=AnalysisResponse)
async def run_analysis(request: AnalysisRequest, http_request: Request) -> AnalysisResponse:
    orchestrator = _orchestrator(http_request)
    started = time.perf_counter()
    try:
        state = await orchestrator.run(request.symbol, request.intervals, agents=request.agents)
    except UnknownAgentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    duration_ms = round((time.perf_counter() - started) * 1000, 1)

    failures = [AgentFailureSchema(agent=f.agent, error=f.error) for f in state.failures]
    if not state.has_report:
        # Every selected agent failed - there is nothing to serve.
        raise HTTPException(
            status_code=502,
            detail={
                "message": "analysis workflow produced no report",
                "failures": [failure.model_dump() for failure in failures],
            },
        )
    return AnalysisResponse(
        symbol=state.symbol,
        intervals=list(state.intervals),
        requested_agents=list(state.requested_agents),
        completed_agents=state.completed,
        failures=failures,
        # Chronological, so the two phases read in the order they happened.
        steps=sorted(state.steps, key=lambda step: step.started_at),
        active_model=http_request.app.state.container.llm_router.active,
        duration_ms=duration_ms,
        technical_report=state.technical_report,
        chart_pattern_report=state.chart_pattern_report,
        market_research_report=state.market_research_report,
        fundamental_report=state.fundamental_report,
        news_report=state.news_report,
        sentiment_report=state.sentiment_report,
        learning_report=state.learning_report,
    )
