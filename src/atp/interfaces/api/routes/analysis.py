"""Analysis endpoints: run agent workflows through the orchestrator.

POST (not GET) because a run has side effects and real cost: market data
fetches and LLM calls.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from atp.application.orchestration import TradingOrchestrator
from atp.domain.models.analysis import TechnicalReport
from atp.domain.models.market import BarInterval
from atp.interfaces.api.schemas import AgentFailureSchema

router = APIRouter(tags=["analysis"])


class AnalysisRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=12, examples=["AAPL"])
    interval: BarInterval = BarInterval.DAY_1


class AnalysisResponse(BaseModel):
    symbol: str
    interval: BarInterval
    completed_agents: list[str]
    technical_report: TechnicalReport


def _orchestrator(request: Request) -> TradingOrchestrator:
    return request.app.state.container.orchestrator  # type: ignore[no-any-return]


@router.post("/analysis", response_model=AnalysisResponse)
async def run_analysis(request: AnalysisRequest, http_request: Request) -> AnalysisResponse:
    orchestrator = _orchestrator(http_request)
    state = await orchestrator.run(request.symbol, request.interval)

    if state.technical_report is None:
        failures = [AgentFailureSchema(agent=f.agent, error=f.error) for f in state.failures]
        raise HTTPException(
            status_code=502,
            detail={
                "message": "analysis workflow produced no report",
                "failures": [failure.model_dump() for failure in failures],
            },
        )
    return AnalysisResponse(
        symbol=state.symbol,
        interval=state.interval,
        completed_agents=state.completed,
        technical_report=state.technical_report,
    )
