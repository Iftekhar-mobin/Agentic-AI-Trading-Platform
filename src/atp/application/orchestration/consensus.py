"""Use case: run the agents, collect the bot's signal, and count the votes.

This is where the platform's two kinds of judgement meet. The specialist agents
read the situation; an external bot applies a mechanical rule to the same
symbol. Both become votes, a deterministic policy counts them, and the risk
engine then decides whether the winning side is even permissible.

The order matters and is not negotiable: **consensus proposes, risk disposes.**
A unanimous, maximally confident pool still cannot open a position the risk
engine rejects, because sizing and exposure limits are not opinions. The
decision returned here therefore carries both — what the voters wanted, and
what the gate allowed.

Nothing is executed. The caller (a human at the dashboard, or a bot relaying to
its trader) decides whether the proposal becomes an order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import structlog
from pydantic import BaseModel

from atp.application.orchestration.graph import TradingOrchestrator
from atp.application.orchestration.state import TradingState
from atp.application.use_cases.check_trade_risk import CheckTradeRisk
from atp.domain.errors import DomainError
from atp.domain.models.analysis import SignalDirection
from atp.domain.models.market import BarInterval
from atp.domain.models.signals import BotSignal
from atp.domain.models.trading import RiskDecision
from atp.domain.models.voting import (
    BotStatus,
    ConsensusDecision,
    Vote,
    VoterKind,
    VotingPolicy,
)
from atp.domain.ports.signals import SignalRepository
from atp.domain.services.voting import tally

log = structlog.get_logger()

AGENT_ARTIFACTS: dict[str, str] = {
    "technical_analysis": "technical_report",
    "chart_pattern": "chart_pattern_report",
    "market_research": "market_research_report",
    "fundamental_analysis": "fundamental_report",
    "news_analysis": "news_report",
    "sentiment_analysis": "sentiment_report",
}
"""Agents whose reports carry a directional assessment.

``continuous_learning`` is excluded deliberately: it reflects on what the others
concluded, so letting it vote would count the same evidence twice.
"""


class ConsensusResult(BaseModel):
    """What the voters wanted, what the gate allowed, and how it was reached."""

    decision: ConsensusDecision
    risk: RiskDecision | None = None
    """``None`` when the verdict was hold - there is no proposal to assess."""

    bot_signal: BotSignal | None = None
    stale_signal: BotSignal | None = None
    """A signal that existed but was too old to vote. Surfaced rather than
    dropped: "the bot went quiet" is operationally different from "the bot
    disagreed", and a silent drop makes them look identical."""

    bot_status: BotStatus = BotStatus.NOT_CONFIGURED
    bot_note: str = ""
    """A sentence the interface can show verbatim, e.g. "sniper_bot has
    published no signal for XAUUSD - deciding on agent analysis alone"."""

    state: TradingState | None = None
    """The full analysis, so callers can show the reasoning behind the votes."""


class ReachConsensus:
    def __init__(
        self,
        orchestrator: TradingOrchestrator,
        signals: SignalRepository,
        risk_check: CheckTradeRisk,
        policy: VotingPolicy,
        *,
        signal_ttl: timedelta,
        expected_bot: str | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._orchestrator = orchestrator
        self._signals = signals
        self._risk_check = risk_check
        self._policy = policy
        self._signal_ttl = signal_ttl
        self._expected_bot = expected_bot
        self._clock = clock

    async def execute(
        self,
        symbol: str,
        intervals: Sequence[BarInterval] | BarInterval = BarInterval.DAY_1,
        *,
        agents: Sequence[str] | None = None,
        signal: BotSignal | None = None,
        stop_loss: float | None = None,
        expected_bot: str | None = None,
    ) -> ConsensusResult:
        """Reach a verdict on one symbol.

        ``signal`` lets a bot submit its vote inline, which saves it a round
        trip; when omitted the most recent stored signal is used if it is still
        fresh. ``expected_bot`` names the bot that *should* be taking part, so
        its absence can be reported rather than silently tolerated.
        """
        symbol = symbol.strip().upper()
        selected = list(agents) if agents else list(AGENT_ARTIFACTS)
        expected = expected_bot if expected_bot is not None else self._expected_bot

        state = await self._orchestrator.run(symbol, intervals, agents=selected)
        votes = list(_agent_votes(state))

        bot_signal, stale = await self._resolve_signal(symbol, signal)
        if bot_signal is not None:
            votes.append(_bot_vote(bot_signal))

        status, note = _bot_availability(
            symbol, bot_signal, stale, expected, self._signal_ttl, self._clock()
        )

        decision = tally(symbol, votes, self._policy)
        log.info(
            "consensus.decided",
            symbol=symbol,
            action=decision.action.value,
            score=decision.score,
            voters=len(votes),
            bot_status=status.value,
        )

        risk = await self._assess(decision, stop_loss=stop_loss)
        return ConsensusResult(
            decision=decision,
            risk=risk,
            bot_signal=bot_signal,
            stale_signal=stale,
            bot_status=status,
            bot_note=note,
            state=state,
        )

    async def _resolve_signal(
        self, symbol: str, inline: BotSignal | None
    ) -> tuple[BotSignal | None, BotSignal | None]:
        """The signal that may vote, and the one that was too old to."""
        if inline is not None:
            # Submitted with the request: it is by definition current.
            return inline, None

        stored = await self._signals.latest(symbol)
        if stored is None:
            return None, None
        if stored.is_fresh(self._clock(), self._signal_ttl):
            return stored, None

        log.info(
            "consensus.signal_stale",
            symbol=symbol,
            source=stored.source,
            age_seconds=round(stored.age(self._clock()).total_seconds()),
        )
        return None, stored

    async def _assess(
        self, decision: ConsensusDecision, *, stop_loss: float | None
    ) -> RiskDecision | None:
        """Size the winning side and put it through the gate.

        A rejection here is not an error: it is the gate doing its job, and the
        caller needs to see the violations. Only an unusable market price is
        allowed to leave this as ``None``.
        """
        side = decision.action.side
        if side is None:
            return None
        try:
            return await self._risk_check.execute(
                decision.symbol,
                side=side,
                stop_loss=Decimal(str(stop_loss)) if stop_loss is not None else None,
                strategy_name="consensus",
            )
        except DomainError as exc:
            log.warning("consensus.risk_check_failed", symbol=decision.symbol, error=str(exc))
            return None


def _bot_availability(
    symbol: str,
    voted: BotSignal | None,
    stale: BotSignal | None,
    expected: str | None,
    ttl: timedelta,
    now: datetime,
) -> tuple[BotStatus, str]:
    """Classify the bot's participation, and say so in plain language.

    The wording is the product here. "Deciding on agent analysis alone" is what
    a trader needs to read before acting on a verdict that looks unanimous but
    was reached without the mechanical system they believe is watching.
    """
    fallback = "Deciding on the agents' own analysis alone."

    if voted is not None:
        return BotStatus.VOTED, (
            f"{voted.source} voted {voted.direction.value} at {voted.confidence:.0%} confidence."
        )

    if stale is not None:
        minutes = stale.age(now).total_seconds() / 60
        limit = ttl.total_seconds() / 60
        return BotStatus.STALE, (
            f"{stale.source}'s last signal for {symbol} is {minutes:.0f} minutes old "
            f"(limit {limit:.0f}), so it was not counted. {fallback}"
        )

    if expected:
        return BotStatus.MISSING, (
            f"{expected} is not available - it has published no signal for {symbol}. {fallback}"
        )

    return BotStatus.NOT_CONFIGURED, (
        "No external bot is configured, so this verdict is the agents' alone. "
        "Point one at POST /signals to add its vote."
    )


def _agent_votes(state: TradingState) -> list[Vote]:
    """One vote per agent that produced a directional assessment.

    Agents that failed simply do not vote. That is the correct behaviour: a
    missing opinion should shrink the pool, never count as a neutral one, which
    would quietly drag every score toward zero.
    """
    votes: list[Vote] = []
    for agent, field in AGENT_ARTIFACTS.items():
        report = getattr(state, field, None)
        if report is None:
            continue
        assessment = getattr(report, "assessment", None)
        direction = getattr(assessment, "direction", None)
        confidence = getattr(assessment, "confidence", None)
        if not isinstance(direction, SignalDirection) or not isinstance(confidence, int | float):
            continue
        votes.append(
            Vote(
                voter=agent,
                kind=VoterKind.AGENT,
                direction=direction,
                confidence=float(confidence),
                rationale=(getattr(assessment, "reasoning", "") or "")[:400],
            )
        )
    return votes


def _bot_vote(signal: BotSignal) -> Vote:
    return Vote(
        voter=signal.source,
        kind=VoterKind.BOT,
        direction=signal.direction,
        confidence=signal.confidence,
        rationale=signal.rationale[:400],
    )
