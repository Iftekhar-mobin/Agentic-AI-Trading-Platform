"""Tests for the deterministic composite and the ranking agent's guardrails."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from atp.application.agents.opportunity_ranking import (
    OpportunityRankingAgent,
    RankingAssessment,
)
from atp.domain.models.analysis import SignalDirection
from atp.domain.models.explainability import Evidence
from atp.domain.models.ranking import Candidate, OpportunityVerdict, ScoreComponents
from atp.domain.models.trading import (
    OrderSide,
    RiskDecision,
    RiskLimits,
    RiskVerdict,
    RiskViolation,
    TradeProposal,
)
from atp.domain.models.universe import AssetClass
from atp.domain.models.voting import (
    ConsensusAction,
    ConsensusDecision,
    Vote,
    VoterKind,
    VotingPolicy,
)
from atp.domain.services import ranking

POLICY = VotingPolicy(threshold=0.35, min_voters=3)


def _vote(direction: SignalDirection, confidence: float, voter: str = "technical_analysis") -> Vote:
    return Vote(voter=voter, kind=VoterKind.AGENT, direction=direction, confidence=confidence)


def _decision(
    *,
    action: ConsensusAction = ConsensusAction.BUY,
    score: float = 0.8,
    votes: tuple[Vote, ...] | None = None,
    quorum_met: bool = True,
    symbol: str = "XAUUSD",
) -> ConsensusDecision:
    return ConsensusDecision(
        symbol=symbol,
        action=action,
        score=score,
        votes=votes
        if votes is not None
        else tuple(
            _vote(SignalDirection.BULLISH, 0.8, name)
            for name in ("technical_analysis", "chart_pattern", "news_analysis")
        ),
        policy=POLICY,
        quorum_met=quorum_met,
        reason="test",
    )


def _risk(verdict: RiskVerdict) -> RiskDecision:
    return RiskDecision(
        verdict=verdict,
        proposal=TradeProposal(
            symbol="XAUUSD",
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            entry_price=Decimal("4400"),
        ),
        limits=RiskLimits(),
        violations=(RiskViolation(rule="max_position_pct", detail="too big"),)
        if verdict is RiskVerdict.REJECTED
        else (),
        metrics={},
    )


# --- the composite ---------------------------------------------------------


def test_components_are_re_derivable_from_their_parts() -> None:
    """The published formula must reproduce the published number."""
    components = ranking.score(_decision(), expected_voters=6)

    base = (
        ranking.WEIGHTS["conviction"] * components.conviction
        + ranking.WEIGHTS["agreement"] * components.agreement
        + ranking.WEIGHTS["confidence"] * components.confidence
        + ranking.WEIGHTS["coverage"] * components.coverage
    )
    expected = base * components.quorum_penalty * components.risk_penalty
    assert components.composite == pytest.approx(expected, abs=1e-3)


def test_weights_sum_to_one_so_the_base_reads_as_a_share() -> None:
    assert sum(ranking.WEIGHTS.values()) == pytest.approx(1.0)


def test_thin_coverage_ranks_below_a_full_pool_at_the_same_conviction() -> None:
    """Half the agents failing is weaker evidence, not neutral evidence."""
    full = _decision()
    thin = _decision(votes=(_vote(SignalDirection.BULLISH, 0.8),) * 1)

    strong = ranking.score(full, expected_voters=6)
    weak = ranking.score(thin, expected_voters=6)

    assert strong.coverage > weak.coverage
    assert strong.composite > weak.composite


def test_a_rejected_risk_verdict_cannot_be_outvoted_by_conviction() -> None:
    approved = ranking.score(_decision(), risk=_risk(RiskVerdict.APPROVED), expected_voters=3)
    rejected = ranking.score(_decision(), risk=_risk(RiskVerdict.REJECTED), expected_voters=3)

    assert rejected.risk_penalty == ranking.RISK_REJECTED_PENALTY
    assert rejected.composite < approved.composite


def test_missing_quorum_halves_the_composite_without_erasing_it() -> None:
    met = ranking.score(_decision(), expected_voters=3)
    missed = ranking.score(_decision(quorum_met=False), expected_voters=3)

    assert missed.quorum_penalty == ranking.NO_QUORUM_PENALTY
    assert 0 < missed.composite < met.composite


def test_neutral_voters_do_not_inflate_mean_confidence() -> None:
    """A confident abstention must not raise an opportunity's rank."""
    directional = _decision(votes=(_vote(SignalDirection.BULLISH, 0.6, "a"),))
    with_neutral = _decision(
        votes=(
            _vote(SignalDirection.BULLISH, 0.6, "a"),
            _vote(SignalDirection.NEUTRAL, 1.0, "b"),
        )
    )

    assert ranking.score(directional).confidence == pytest.approx(0.6)
    assert ranking.score(with_neutral).confidence == pytest.approx(0.6)


def test_expected_voters_of_zero_treats_coverage_as_complete() -> None:
    assert ranking.score(_decision(), expected_voters=0).coverage == 1.0


def test_a_hold_ranks_below_an_otherwise_identical_actionable_verdict() -> None:
    buy = ranking.score(_decision(action=ConsensusAction.BUY))
    hold = ranking.score(_decision(action=ConsensusAction.HOLD))

    assert hold.risk_penalty == ranking.HOLD_PENALTY
    assert hold.composite < buy.composite


# --- the agent -------------------------------------------------------------


def _candidate(symbol: str, composite: float) -> Candidate:
    return Candidate(
        symbol=symbol,
        name=symbol,
        asset_class=AssetClass.STOCK,
        action=ConsensusAction.BUY,
        score=0.7,
        agreement=1.0,
        quorum_met=True,
        votes=(_vote(SignalDirection.BULLISH, 0.8),),
        components=ScoreComponents(
            conviction=0.7,
            agreement=1.0,
            confidence=0.8,
            coverage=1.0,
            quorum_penalty=1.0,
            risk_penalty=1.0,
            composite=composite,
        ),
    )


def _verdict(symbol: str) -> OpportunityVerdict:
    return OpportunityVerdict(
        symbol=symbol,
        profit_potential=0.7,
        key_driver="trend",
        primary_risk="reversal",
        horizon="days",
        reasoning="because",
        confidence=0.6,
        evidence=(Evidence(source=symbol, statement="strong"),),
        invalidation_conditions=("it breaks",),
    )


class StubLLM:
    """Returns a canned assessment, or raises."""

    def __init__(self, assessment: RankingAssessment | None = None, error: str = "") -> None:
        self._assessment = assessment
        self._error = error
        self.prompts: list[str] = []

    async def generate_structured(self, response_model: Any, *, system: str, prompt: str) -> Any:
        self.prompts.append(prompt)
        if self._error:
            raise RuntimeError(self._error)
        return self._assessment


CANDIDATES = [_candidate("AAPL", 0.5), _candidate("NVDA", 0.9), _candidate("MSFT", 0.7)]


async def test_the_agent_may_reorder_the_deterministic_baseline() -> None:
    """The composite is a prior, not an instruction."""
    agent = OpportunityRankingAgent(
        StubLLM(RankingAssessment(ordering=(_verdict("AAPL"), _verdict("NVDA")), narrative="mixed"))
    )
    result = await agent.rank(CANDIDATES)

    assert [row.symbol for row in result.ranked] == ["AAPL", "NVDA", "MSFT"]
    assert result.ranked[0].verdict is not None
    assert result.narrative == "mixed"
    assert result.ranked_by_agent


async def test_candidates_the_agent_skipped_are_appended_not_lost() -> None:
    agent = OpportunityRankingAgent(StubLLM(RankingAssessment(ordering=(_verdict("AAPL"),))))
    result = await agent.rank(CANDIDATES)

    assert [row.symbol for row in result.ranked] == ["AAPL", "NVDA", "MSFT"]
    # ...and the unranked ones say so rather than borrowing reasoning.
    assert result.ranked[1].verdict is None


async def test_a_hallucinated_symbol_is_dropped() -> None:
    agent = OpportunityRankingAgent(
        StubLLM(RankingAssessment(ordering=(_verdict("TSLA"), _verdict("NVDA"))))
    )
    result = await agent.rank(CANDIDATES)

    assert "TSLA" not in [row.symbol for row in result.ranked]
    assert result.ranked[0].symbol == "NVDA"


async def test_a_duplicated_symbol_is_counted_once() -> None:
    agent = OpportunityRankingAgent(
        StubLLM(RankingAssessment(ordering=(_verdict("NVDA"), _verdict("NVDA"))))
    )
    result = await agent.rank(CANDIDATES)

    symbols = [row.symbol for row in result.ranked]
    assert symbols.count("NVDA") == 1
    assert len(symbols) == len(CANDIDATES)


async def test_a_failed_ranker_still_returns_the_deterministic_order() -> None:
    """A screen that cost sixty LLM calls must not be lost to the sixty-first."""
    agent = OpportunityRankingAgent(StubLLM(error="provider is down"))
    result = await agent.rank(CANDIDATES)

    assert [row.symbol for row in result.ranked] == ["NVDA", "MSFT", "AAPL"]
    assert not result.ranked_by_agent
    assert all(row.verdict is None for row in result.ranked)
    assert "deterministic composite" in result.narrative


async def test_the_limit_truncates_the_board_but_not_the_prompt() -> None:
    """You cannot pick the top two of a set you were only shown two of."""
    llm = StubLLM(RankingAssessment(ordering=tuple(_verdict(c.symbol) for c in CANDIDATES)))
    result = await OpportunityRankingAgent(llm).rank(CANDIDATES, limit=2)

    assert len(result.ranked) == 2
    assert result.considered == 3
    for candidate in CANDIDATES:
        assert candidate.symbol in llm.prompts[0]


async def test_ranking_an_empty_basket_is_not_an_error() -> None:
    agent = OpportunityRankingAgent(StubLLM())
    result = await agent.rank([])

    assert result.ranked == ()
    assert result.considered == 0


async def test_the_model_is_told_how_many_rows_the_shortlist_will_show() -> None:
    """Left to guess, it answers "the ones worth ranking" — and on a quiet day
    that is none, which empties the table the trader asked for."""
    llm = StubLLM(RankingAssessment(ordering=(_verdict("NVDA"),)))
    await OpportunityRankingAgent(llm).rank(CANDIDATES, limit=2)

    assert '"shortlist_size": 2' in llm.prompts[0]


async def test_the_shortlist_size_never_exceeds_the_basket() -> None:
    llm = StubLLM(RankingAssessment(ordering=()))
    await OpportunityRankingAgent(llm).rank(CANDIDATES[:1], limit=5)

    assert '"shortlist_size": 1' in llm.prompts[0]
