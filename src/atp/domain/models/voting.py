"""Decision fusion: many opinions, one deterministic verdict.

The platform has two very different kinds of decision-maker. Specialist agents
read a situation and produce a calibrated view; an external bot applies a
mechanical rule and produces a signal. Neither should be trusted alone — the
agents can be eloquent about a weak setup, and the bot cannot see a news
catalyst at all.

So both vote, and a **deterministic, unit-tested policy** counts the votes. That
split matters: an LLM contributes a vote, never the tally. A consensus that a
model could talk its way into is not a control, and the arithmetic here is the
part that has to be reproducible and auditable months later.

What voting is *not* is permission to trade. It answers "do we want this?" The
risk engine separately answers "are we allowed?" and holds a veto that no
majority can override — see ``domain.services.risk_engine``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from atp.domain.models.analysis import SignalDirection
from atp.domain.models.trading import OrderSide


class VoterKind(StrEnum):
    AGENT = "agent"
    """An LLM specialist from the supervisor graph."""

    BOT = "bot"
    """An external mechanical system reporting through ``POST /signals``."""


class BotStatus(StrEnum):
    """Why the external bot did or did not vote.

    Silence and disagreement are completely different events, and a system that
    renders both as "no bot vote" will eventually get someone hurt: a trader who
    believes a mechanical system endorsed a trade, when in fact it was offline,
    is acting on evidence that does not exist. Every consensus therefore states
    which of these happened.
    """

    VOTED = "voted"
    """A fresh signal was counted."""

    STALE = "stale"
    """A signal exists but is older than the freshness window."""

    MISSING = "missing"
    """A bot was expected by name and has published nothing for this symbol."""

    NOT_CONFIGURED = "not_configured"
    """No bot was expected. Agents decide alone by design, not by accident."""

    @property
    def voted(self) -> bool:
        return self is BotStatus.VOTED

    @property
    def degraded(self) -> bool:
        """Whether a bot was meant to take part but could not."""
        return self in {BotStatus.STALE, BotStatus.MISSING}


class ConsensusAction(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    """The default. Nothing forces a trade out of an inconclusive tally."""

    @property
    def side(self) -> OrderSide | None:
        """The order side this implies, or ``None`` for hold."""
        if self is ConsensusAction.BUY:
            return OrderSide.BUY
        if self is ConsensusAction.SELL:
            return OrderSide.SELL
        return None


DIRECTION_WEIGHTS: dict[SignalDirection, float] = {
    SignalDirection.BULLISH: 1.0,
    SignalDirection.BEARISH: -1.0,
    SignalDirection.NEUTRAL: 0.0,
}
"""Direction as a signed number, so votes can be summed.

Neutral is deliberately 0 rather than excluded: a confident "nothing here"
should dilute conviction, and dropping it would let one bull outvote five
shrugs.
"""


class Vote(BaseModel):
    """One participant's opinion, with the confidence to weight it by."""

    model_config = ConfigDict(frozen=True)

    voter: str = Field(min_length=1, description="e.g. 'technical_analysis' or 'mt5-ea-v3'")
    kind: VoterKind
    direction: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""

    @property
    def weight(self) -> float:
        """Signed contribution: direction scaled by how sure the voter is."""
        return DIRECTION_WEIGHTS[self.direction] * self.confidence


class VotingPolicy(BaseModel):
    """How votes become a decision. Configured once, applied identically to every
    tally so two runs with the same votes cannot disagree."""

    model_config = ConfigDict(frozen=True)

    threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum |score| to act; below this the verdict is hold",
    )
    min_voters: int = Field(
        default=3,
        ge=1,
        description="Quorum - fewer participants than this can never trade",
    )
    require_bot_signal: bool = Field(
        default=False,
        description="Whether a fresh bot signal is mandatory for a non-hold verdict",
    )


class ConsensusDecision(BaseModel):
    """The tally, with everything needed to re-derive it by hand.

    Carries the votes rather than just the outcome: "why did it not trade" is
    the question people actually ask, and it is unanswerable from a score alone.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1, max_length=12)
    action: ConsensusAction
    score: float = Field(ge=-1.0, le=1.0, description="Confidence-weighted net direction")
    votes: tuple[Vote, ...]
    policy: VotingPolicy
    quorum_met: bool
    reason: str = Field(min_length=1, description="Plain-language account of the outcome")
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @property
    def agreement(self) -> float:
        """Share of non-neutral voters siding with the outcome.

        Distinct from ``score``: five hesitant bulls and one certain bear can
        produce a weak score with near-total agreement, and the two readings
        together say more than either alone.
        """
        directional = [vote for vote in self.votes if vote.direction is not SignalDirection.NEUTRAL]
        if not directional or self.action is ConsensusAction.HOLD:
            return 0.0
        wanted = (
            SignalDirection.BULLISH
            if self.action is ConsensusAction.BUY
            else SignalDirection.BEARISH
        )
        return sum(1 for vote in directional if vote.direction is wanted) / len(directional)

    @property
    def tally(self) -> dict[str, int]:
        """Head-count per direction, for display alongside the weighted score."""
        counts = {direction.value: 0 for direction in SignalDirection}
        for vote in self.votes:
            counts[vote.direction.value] += 1
        return counts
