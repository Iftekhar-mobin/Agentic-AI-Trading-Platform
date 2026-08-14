"""Screening endpoints: pick a basket, get a ranked shortlist back.

``GET /universe`` is the menu - the instruments the console offers, grouped by
class, with the vendor ticker each one resolves to so a surprising result is
traceable to the series it was computed on.

``POST /screen`` is the expensive one. It runs the full consensus workflow on
every selected symbol and then ranks the outcomes, so its cost scales with
``symbols x agents``. POST rather than GET for the same reason as ``/analysis``:
this is not a lookup, it spends money.

The response carries the whole analysis for every symbol, not just the winners.
That is deliberate and it is why the payload is large: an interface that shows a
shortlist has to be able to answer "and why is the one I care about *not* in
it?" without paying for a second run.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from atp.application.orchestration import ScreenOpportunities
from atp.application.orchestration.consensus import ConsensusResult
from atp.domain.models.analysis import TechnicalReport
from atp.domain.models.fundamentals import FundamentalReport
from atp.domain.models.llm import ActiveModel
from atp.domain.models.market import BarInterval
from atp.domain.models.memory import LearningReport
from atp.domain.models.news import NewsReport
from atp.domain.models.patterns import ChartPatternReport
from atp.domain.models.ranking import OpportunityRanking
from atp.domain.models.research import MarketResearchReport
from atp.domain.models.sentiment import SentimentReport
from atp.domain.models.trading import RiskDecision
from atp.domain.models.universe import Asset, AssetClass
from atp.domain.models.voting import BotStatus, ConsensusDecision
from atp.domain.universe import CATALOG, DEFAULT_SELECTION, by_class
from atp.infrastructure.vendor_symbols import to_vendor_symbol
from atp.interfaces.api.schemas import AgentFailureSchema
from atp.interfaces.api.security import RequiresRead

router = APIRouter(tags=["screening"], dependencies=[RequiresRead])


class UniverseAsset(BaseModel):
    """A catalog entry, plus the ticker it is actually fetched under."""

    symbol: str
    name: str
    asset_class: AssetClass
    group: str
    vendor_symbol: str = Field(
        description="What the market data vendor is asked for. Differs for spot "
        "metals, FX and crypto - XAUUSD is served as the front-month gold future"
    )


class UniverseCategory(BaseModel):
    asset_class: AssetClass
    label: str
    assets: list[UniverseAsset]


class UniverseResponse(BaseModel):
    categories: list[UniverseCategory]
    default_selection: list[str] = Field(
        description="A starter basket spanning every class, for a first screen"
    )
    count: int
    max_symbols: int = Field(description="Most symbols one screen will accept")


class SymbolReports(BaseModel):
    """One symbol's full analysis - what a row expands into."""

    technical_report: TechnicalReport | None = None
    chart_pattern_report: ChartPatternReport | None = None
    market_research_report: MarketResearchReport | None = None
    fundamental_report: FundamentalReport | None = None
    news_report: NewsReport | None = None
    sentiment_report: SentimentReport | None = None
    learning_report: LearningReport | None = None
    failures: list[AgentFailureSchema] = Field(default_factory=list)


class SymbolOutcome(BaseModel):
    """The consensus verdict for one screened symbol, with its analysis."""

    symbol: str
    decision: ConsensusDecision
    risk: RiskDecision | None = None
    bot_status: BotStatus
    bot_note: str
    agreement: float
    tally: dict[str, int]
    executable: bool
    reports: SymbolReports


class ScreenRequest(BaseModel):
    symbols: list[str] = Field(
        min_length=1,
        max_length=50,
        description="Symbols to screen. Anything in /universe, or your own ticker",
        examples=[["XAUUSD", "EURUSD", "NVDA", "^GSPC", "BTCUSD"]],
    )
    intervals: list[BarInterval] = Field(
        default_factory=lambda: [BarInterval.DAY_1],
        min_length=1,
        max_length=5,
        examples=[["1d", "4h"]],
    )
    agents: list[str] = Field(
        default_factory=list, description="Subset of the voting agents; empty runs all"
    )
    top_n: int | None = Field(
        default=None,
        ge=1,
        le=25,
        description="Shortlist length. Every symbol is still returned in `outcomes`",
    )
    expected_bot: str | None = Field(default=None, max_length=64, examples=["sniper_bot"])
    stop_loss: float | None = Field(
        default=None,
        gt=0,
        description="Applied to every symbol, so only meaningful for a basket of one",
    )


class ScreenResponse(BaseModel):
    ranking: OpportunityRanking
    outcomes: list[SymbolOutcome] = Field(
        description="Every symbol that was screened, ranked or not, in request order"
    )
    failures: list[dict[str, str]] = Field(
        default_factory=list, description="Symbols that could not be screened at all"
    )
    requested: list[str]
    active_model: ActiveModel | None = None
    duration_ms: float


def _screener(request: Request) -> ScreenOpportunities:
    return request.app.state.container.screen_opportunities  # type: ignore[no-any-return]


def _asset(asset: Asset) -> UniverseAsset:
    return UniverseAsset(
        symbol=asset.symbol,
        name=asset.name,
        asset_class=asset.asset_class,
        group=asset.group,
        vendor_symbol=to_vendor_symbol(asset.symbol),
    )


@router.get("/universe", response_model=UniverseResponse)
async def list_universe(http_request: Request) -> UniverseResponse:
    """The selectable instruments, grouped for an interface to render directly."""
    settings = http_request.app.state.container.settings
    categories = [
        UniverseCategory(
            asset_class=asset_class,
            label=asset_class.label,
            assets=[_asset(asset) for asset in by_class(asset_class)],
        )
        for asset_class in AssetClass
        if by_class(asset_class)
    ]
    return UniverseResponse(
        categories=categories,
        default_selection=list(DEFAULT_SELECTION),
        count=len(CATALOG),
        max_symbols=settings.screening.max_symbols,
    )


@router.post("/screen", response_model=ScreenResponse)
async def screen(request: ScreenRequest, http_request: Request) -> ScreenResponse:
    """Run consensus across the basket and rank what comes back.

    A screen where every symbol failed is a 502: there is nothing to rank, and
    an empty leaderboard would read as "no opportunities" rather than "no data".
    """
    result = await _screener(http_request).execute(
        request.symbols,
        request.intervals,
        agents=request.agents or None,
        top_n=request.top_n,
        stop_loss=request.stop_loss,
        expected_bot=request.expected_bot,
    )

    if not result.outcomes:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "no symbol in the basket could be screened",
                "failures": [failure.model_dump() for failure in result.failures],
            },
        )

    return ScreenResponse(
        ranking=result.ranking,
        outcomes=[
            _outcome(symbol, outcome)
            for symbol in result.requested
            if (outcome := result.outcomes.get(symbol)) is not None
        ],
        failures=[failure.model_dump() for failure in result.failures],
        requested=result.requested,
        active_model=http_request.app.state.container.llm_router.active,
        duration_ms=result.duration_ms,
    )


def _outcome(symbol: str, result: ConsensusResult) -> SymbolOutcome:
    state = result.state
    reports: dict[str, Any] = {}
    if state is not None:
        reports = {
            field: getattr(state, field)
            for field in (
                "technical_report",
                "chart_pattern_report",
                "market_research_report",
                "fundamental_report",
                "news_report",
                "sentiment_report",
                "learning_report",
            )
        }
        reports["failures"] = [
            AgentFailureSchema(agent=failure.agent, error=failure.error)
            for failure in state.failures
        ]
    return SymbolOutcome(
        symbol=symbol,
        decision=result.decision,
        risk=result.risk,
        bot_status=result.bot_status,
        bot_note=result.bot_note,
        agreement=round(result.decision.agreement, 4),
        tally=result.decision.tally,
        executable=result.risk is not None and result.risk.verdict.value == "approved",
        reports=SymbolReports(**reports),
    )
