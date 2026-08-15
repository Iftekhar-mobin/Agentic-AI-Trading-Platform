"""Consensus endpoints: where an external bot and the agent pool meet.

Two entry points, because bots integrate in two different shapes:

- ``POST /signals`` records an opinion and returns immediately. Right for a bot
  that evaluates on a fast cycle and does not want to block on an LLM workflow.
- ``POST /consensus`` runs the agents and returns the combined verdict. It
  accepts the bot's vote inline, so a bot that wants an answer needs one call
  rather than two.

Both take the bot's own vocabulary. ``python_signal_bot`` emits ``BUY``/``SELL``
at decision level and ``bullish``/``bearish`` at signal level; requiring it to
translate would be this platform's problem leaking into the caller's.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from atp.application.orchestration import ReachConsensus
from atp.application.request_scope import supplied_inputs
from atp.domain.models.analysis import SignalDirection
from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.domain.models.signals import BotSignal
from atp.domain.models.trading import RiskDecision
from atp.domain.models.voting import BotStatus, ConsensusDecision
from atp.domain.ports.signals import SignalRepository
from atp.interfaces.api.security import RequiresRead, RequiresSignal

router = APIRouter(tags=["consensus"])

_DIRECTION_ALIASES: dict[str, SignalDirection] = {
    "buy": SignalDirection.BULLISH,
    "long": SignalDirection.BULLISH,
    "bullish": SignalDirection.BULLISH,
    "up": SignalDirection.BULLISH,
    "sell": SignalDirection.BEARISH,
    "short": SignalDirection.BEARISH,
    "bearish": SignalDirection.BEARISH,
    "down": SignalDirection.BEARISH,
    "neutral": SignalDirection.NEUTRAL,
    "none": SignalDirection.NEUTRAL,
    "hold": SignalDirection.NEUTRAL,
    "flat": SignalDirection.NEUTRAL,
}
"""Every spelling of a direction seen in the wild, mapped to the domain's."""


def _coerce_direction(value: Any) -> SignalDirection:
    if isinstance(value, SignalDirection):
        return value
    if isinstance(value, str) and (mapped := _DIRECTION_ALIASES.get(value.strip().lower())):
        return mapped
    allowed = sorted(_DIRECTION_ALIASES)
    msg = f"unknown direction {value!r}; expected one of {allowed}"
    raise ValueError(msg)


class SignalRequest(BaseModel):
    """A bot's published opinion.

    Mirrors ``TradeDecision`` from ``python_signal_bot`` closely enough that a
    caller can forward its own fields with almost no mapping.
    """

    symbol: str = Field(min_length=1, max_length=12, examples=["XAUUSD"])
    direction: SignalDirection = Field(examples=["BUY"])
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    source: str = Field(min_length=1, max_length=64, examples=["mt5-ea-v3"])
    rationale: str = Field(default="", max_length=2000)
    strategy: str | None = Field(default=None, max_length=64)

    @field_validator("direction", mode="before")
    @classmethod
    def _accept_bot_vocabulary(cls, value: Any) -> SignalDirection:
        return _coerce_direction(value)

    def to_signal(self) -> BotSignal:
        return BotSignal(
            symbol=self.symbol,
            direction=self.direction,
            confidence=self.confidence,
            source=self.source,
            rationale=self.rationale,
            strategy=self.strategy,
        )


class BarInput(BaseModel):
    """One OHLCV candle supplied by the caller.

    Kept separate from the domain ``Bar`` so a malformed candle fails as a 422
    on the request rather than as an exception mid-analysis. Timestamps must
    carry a zone: a naive one is ambiguous, and a bot on a broker's server time
    is exactly the caller most likely to send one.
    """

    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(default=0.0, ge=0)

    @field_validator("timestamp")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            msg = "bar timestamps must include a timezone offset"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _check_ohlc_consistency(self) -> BarInput:
        # Mirrors the domain Bar's own check, deliberately duplicated so the
        # rejection happens here as a 422 naming the bad candle. Without it the
        # identical failure surfaces later, out of the request's validation
        # context, and reaches the caller as a 500.
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            msg = (
                f"inconsistent OHLC bar at {self.timestamp.isoformat()}: "
                f"open={self.open} high={self.high} low={self.low} close={self.close}"
            )
            raise ValueError(msg)
        return self

    def to_bar(self) -> Bar:
        return Bar(
            timestamp=self.timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


class ConsensusRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=12, examples=["XAUUSD"])
    intervals: list[BarInterval] = Field(
        default_factory=lambda: [BarInterval.DAY_1],
        min_length=1,
        max_length=5,
    )
    agents: list[str] = Field(default_factory=list)
    expected_bot: str | None = Field(
        default=None,
        max_length=64,
        description="Bot that should be voting, e.g. 'sniper_bot'. Naming it turns "
        "its silence into a reported status instead of an unnoticed absence.",
        examples=["sniper_bot"],
    )
    signal: SignalRequest | None = Field(
        default=None,
        description="The caller's own vote, submitted inline; omit to use the latest stored one",
    )
    stop_loss: float | None = Field(
        default=None,
        gt=0,
        description="Used to size the proposal; without it the risk gate rejects for safety",
    )
    bars: dict[BarInterval, list[BarInput]] = Field(
        default_factory=dict,
        description="Candles to analyse, per timeframe, supplied by the caller instead of "
        "fetched here. A bot that has already pulled these from its broker should send "
        "them: they are the same feed it decided on, at the moment it decided, and for "
        "spot metals they avoid this platform's fallback to the front-month future "
        "(XAUUSD -> GC=F). Timeframes carrying fewer than 60 usable bars are dropped.",
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="The caller's own reading of the setup - indicator values, "
        "higher-timeframe gate, levels, execution conditions. Reaches the agents as "
        "supplementary evidence they may weigh but must not defer to. Omit any intended "
        "direction: an agent that is told the answer cannot independently check it.",
    )

    def price_histories(self) -> dict[tuple[str, BarInterval], PriceHistory]:
        """Supplied bars as domain histories, keyed for the request scope."""
        symbol = self.symbol.strip().upper()
        return {
            (symbol, interval): PriceHistory(
                symbol=symbol,
                interval=interval,
                bars=tuple(row.to_bar() for row in rows),
            )
            for interval, rows in self.bars.items()
            if rows
        }


class ConsensusResponse(BaseModel):
    decision: ConsensusDecision
    risk: RiskDecision | None = None
    bot_signal: BotSignal | None = None
    stale_signal: BotSignal | None = Field(
        default=None,
        description="A signal too old to vote - the bot went quiet rather than disagreed",
    )
    bot_status: BotStatus
    bot_note: str = Field(description="Plain-language account of the bot's participation")
    agreement: float = Field(description="Share of directional voters siding with the outcome")
    tally: dict[str, int]
    executable: bool = Field(
        description="Consensus wants a trade and the risk gate approved it; still needs a human",
    )


def _signals(request: Request) -> SignalRepository:
    return request.app.state.container.signal_repository  # type: ignore[no-any-return]


def _consensus(request: Request) -> ReachConsensus:
    return request.app.state.container.reach_consensus  # type: ignore[no-any-return]


@router.post("/signals", dependencies=[RequiresSignal])
async def publish_signal(request: SignalRequest, http_request: Request) -> BotSignal:
    """Record a bot's opinion so it can vote in a later consensus run."""
    signal = request.to_signal()
    await _signals(http_request).append(signal)
    return signal


@router.get("/signals", dependencies=[RequiresRead])
async def list_signals(http_request: Request, limit: int = 50) -> dict[str, Any]:
    """The recent audit trail of what bots have claimed."""
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
    signals = await _signals(http_request).list_signals(limit=limit)
    return {"signals": signals, "count": len(signals)}


@router.post("/consensus", response_model=ConsensusResponse, dependencies=[RequiresRead])
async def reach_consensus(request: ConsensusRequest, http_request: Request) -> ConsensusResponse:
    """Run the agents, count them with the bot, and return the verdict.

    Never executes. A caller that wants the trade sends the proposal to
    ``POST /trade``, which is a different scope on purpose.
    """
    # Caller-supplied candles and setup context are visible to the loaders and
    # agents for the duration of this call only, then reset — so one request
    # can never analyse another's bars.
    with supplied_inputs(bars=request.price_histories(), context=request.context):
        result = await _consensus(http_request).execute(
            request.symbol,
            request.intervals,
            agents=request.agents or None,
            signal=request.signal.to_signal() if request.signal else None,
            stop_loss=request.stop_loss,
            expected_bot=request.expected_bot,
        )
    decision = result.decision
    return ConsensusResponse(
        decision=decision,
        risk=result.risk,
        bot_signal=result.bot_signal,
        stale_signal=result.stale_signal,
        bot_status=result.bot_status,
        bot_note=result.bot_note,
        agreement=round(decision.agreement, 4),
        tally=decision.tally,
        executable=result.risk is not None and result.risk.verdict.value == "approved",
    )
