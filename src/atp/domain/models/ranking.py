"""Ranking a basket of verdicts: which of these is worth the next position?

Consensus answers "what do we think about this symbol?" one symbol at a time.
It cannot answer the question a trader with finite capital actually has, which
is comparative: *given all of these, which few deserve attention today?*

The split here mirrors the one in ``voting``. A deterministic composite
(``ScoreComponents``) does the arithmetic - conviction, agreement, calibrated
confidence and how much of the agent pool actually reported - and an LLM then
does the judgement the arithmetic cannot: whether a 0.62 on gold with every
timeframe aligned is a better use of risk than a 0.66 on a single-catalyst
equity. The model reorders and explains; it never recomputes the numbers, and
its ordering is validated against the candidates it was given.

**What "profit potential" is not.** Nothing here forecasts a return. There is no
expected value, no win rate, no target. The composite measures how strongly and
how coherently the pool believes something, which is the only honest input the
platform has. A high-ranked row means "the evidence here is the strongest of
the set", never "this will pay".
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from atp.domain.models.explainability import Explanation
from atp.domain.models.universe import AssetClass
from atp.domain.models.voting import BotStatus, ConsensusAction, Vote


class ScoreComponents(BaseModel):
    """The deterministic composite, kept in pieces so it can be re-derived.

    Carried into the interface unchanged. "Why is EURUSD above NVDA?" has to be
    answerable by reading four numbers, not by trusting a single one.
    """

    model_config = ConfigDict(frozen=True)

    conviction: float = Field(
        ge=0.0, le=1.0, description="|consensus score| - how far off the fence the pool is"
    )
    agreement: float = Field(
        ge=0.0, le=1.0, description="Share of directional voters siding with the outcome"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Mean calibrated confidence across the voters"
    )
    coverage: float = Field(
        ge=0.0,
        le=1.0,
        description="Share of the expected voter pool that actually reported; "
        "a symbol half its agents could not analyse is weaker evidence, not neutral evidence",
    )
    quorum_penalty: float = Field(ge=0.0, le=1.0)
    risk_penalty: float = Field(
        ge=0.0,
        le=1.0,
        description="Applied when the risk gate rejected the proposal - a trade "
        "that cannot be placed is not an opportunity, however convinced the pool is",
    )
    composite: float = Field(ge=0.0, le=1.0, description="The ranked number")


class Candidate(BaseModel):
    """One symbol's consensus outcome, flattened for comparison.

    Everything the ranking agent is allowed to reason about is in here. The full
    analysis stays with the caller: it is far too much to put in one prompt, and
    the agent's job is comparison, not re-analysis.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1, max_length=12)
    name: str
    asset_class: AssetClass
    action: ConsensusAction
    score: float = Field(ge=-1.0, le=1.0)
    agreement: float = Field(ge=0.0, le=1.0)
    quorum_met: bool
    votes: tuple[Vote, ...] = ()
    components: ScoreComponents
    bot_status: BotStatus = BotStatus.NOT_CONFIGURED
    risk_verdict: str | None = Field(
        default=None, description="'approved' / 'rejected', or None when nothing was proposed"
    )
    risk_violations: tuple[str, ...] = ()
    highlights: tuple[str, ...] = Field(
        default=(),
        description="Short factual notes lifted from the reports, e.g. timeframe "
        "alignment or the patterns detected - texture the raw score cannot carry",
    )


class OpportunityVerdict(Explanation):
    """The ranking agent's comparative take on one candidate.

    Inherits the explainability envelope, so a ranking cannot be produced
    without reasoning, calibrated confidence, cited evidence and the conditions
    that would refute it - the same bar every other agent output meets.
    """

    symbol: str = Field(min_length=1, max_length=12)
    profit_potential: float = Field(
        ge=0.0,
        le=1.0,
        description="How much of the opportunity in this basket sits in this name. "
        "Relative to the other candidates, not an expected return",
    )
    key_driver: str = Field(min_length=1, description="The one thing carrying this idea")
    primary_risk: str = Field(min_length=1, description="The one thing most likely to break it")
    horizon: str = Field(
        min_length=1, description="Rough holding period the case implies, e.g. 'days'"
    )


class RankedOpportunity(BaseModel):
    """A row in the leaderboard: the numbers, and the agent's read of them."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)
    candidate: Candidate
    verdict: OpportunityVerdict | None = Field(
        default=None,
        description="None when the agent did not rank this one and it was placed "
        "by composite alone - shown as such rather than given borrowed reasoning",
    )

    @property
    def symbol(self) -> str:
        return self.candidate.symbol

    @property
    def composite(self) -> float:
        return self.candidate.components.composite


class OpportunityRanking(BaseModel):
    """The full leaderboard for one screen."""

    model_config = ConfigDict(frozen=True)

    ranked: tuple[RankedOpportunity, ...]
    narrative: str = Field(
        default="",
        description="What the basket looks like taken together - correlation "
        "between the leaders, and whether anything is worth acting on at all",
    )
    considered: int = Field(ge=0, description="Candidates that reached the ranker")
    ranked_by_agent: bool = Field(
        default=True,
        description="False when the agent failed and the order is the deterministic "
        "composite alone. The screen still returns a ranking; it says which kind",
    )
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("ranked")
    @classmethod
    def _ranks_are_sequential(
        cls, value: tuple[RankedOpportunity, ...]
    ) -> tuple[RankedOpportunity, ...]:
        expected = list(range(1, len(value) + 1))
        if [row.rank for row in value] != expected:
            msg = "ranked rows must be ordered and numbered from 1 without gaps"
            raise ValueError(msg)
        return value

    @property
    def top(self) -> RankedOpportunity | None:
        return self.ranked[0] if self.ranked else None
