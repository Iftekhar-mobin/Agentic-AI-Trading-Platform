"""Tests for the multi-symbol screen: fan-out, isolation, and the shortlist."""

from __future__ import annotations

import asyncio
from typing import Any, cast

from atp.application.agents.opportunity_ranking import OpportunityRankingAgent, RankingAssessment
from atp.application.orchestration.consensus import ConsensusResult, ReachConsensus
from atp.application.orchestration.screening import ScreenOpportunities
from atp.domain.models.analysis import SignalDirection
from atp.domain.models.ranking import OpportunityRanking
from atp.domain.models.voting import (
    BotStatus,
    ConsensusAction,
    ConsensusDecision,
    Vote,
    VoterKind,
    VotingPolicy,
)

POLICY = VotingPolicy(threshold=0.35, min_voters=3)


def _result(symbol: str, score: float) -> ConsensusResult:
    direction = SignalDirection.BULLISH if score >= 0 else SignalDirection.BEARISH
    action = ConsensusAction.BUY if score >= POLICY.threshold else ConsensusAction.HOLD
    return ConsensusResult(
        decision=ConsensusDecision(
            symbol=symbol,
            action=action,
            score=score,
            votes=tuple(
                Vote(
                    voter=name,
                    kind=VoterKind.AGENT,
                    direction=direction,
                    confidence=abs(score),
                )
                for name in ("technical_analysis", "chart_pattern", "news_analysis")
            ),
            policy=POLICY,
            quorum_met=True,
            reason="test",
        ),
        bot_status=BotStatus.NOT_CONFIGURED,
        bot_note="none configured",
    )


class StubConsensus:
    """Records how it was called, and can be told to fail for one symbol."""

    def __init__(self, scores: dict[str, float], *, broken: str | None = None) -> None:
        self._scores = scores
        self._broken = broken
        self.calls: list[str] = []
        self.concurrent = 0
        self.peak = 0

    async def execute(self, symbol: str, intervals: Any = None, **kwargs: Any) -> ConsensusResult:
        self.calls.append(symbol)
        self.concurrent += 1
        self.peak = max(self.peak, self.concurrent)
        try:
            await asyncio.sleep(0)
            if symbol == self._broken:
                msg = f"no price data for {symbol}"
                raise RuntimeError(msg)
            return _result(symbol, self._scores.get(symbol, 0.5))
        finally:
            self.concurrent -= 1


class StubRanker:
    """Ranks by composite, so the tests assert on the screen, not the agent."""

    def __init__(self) -> None:
        self.candidates: list[Any] = []

    async def rank(self, candidates: Any, *, limit: int | None = None) -> OpportunityRanking:
        from atp.domain.models.ranking import RankedOpportunity

        self.candidates = list(candidates)
        ordered = sorted(self.candidates, key=lambda c: -c.components.composite)
        rows = [
            RankedOpportunity(rank=index, candidate=candidate)
            for index, candidate in enumerate(ordered, start=1)
        ]
        return OpportunityRanking(
            ranked=tuple(rows if limit is None else rows[:limit]),
            considered=len(ordered),
        )


def build(
    scores: dict[str, float],
    *,
    broken: str | None = None,
    concurrency: int = 2,
    max_symbols: int = 15,
    top_n: int = 5,
) -> tuple[ScreenOpportunities, StubConsensus, StubRanker]:
    consensus = StubConsensus(scores, broken=broken)
    ranker = StubRanker()
    screener = ScreenOpportunities(
        cast(ReachConsensus, consensus),
        cast(OpportunityRankingAgent, ranker),
        concurrency=concurrency,
        max_symbols=max_symbols,
        default_top_n=top_n,
    )
    return screener, consensus, ranker


SCORES = {"NVDA": 0.9, "EURUSD": 0.8, "XAUUSD": 0.7, "AAPL": 0.6, "BTCUSD": 0.5, "^GSPC": 0.4}


async def test_a_basket_is_screened_and_shortlisted() -> None:
    screener, consensus, _ = build(SCORES)
    result = await screener.execute(list(SCORES), top_n=3)

    assert sorted(consensus.calls) == sorted(SCORES)
    assert len(result.outcomes) == len(SCORES)
    assert [row.symbol for row in result.ranking.ranked] == ["NVDA", "EURUSD", "XAUUSD"]
    # Every symbol is still returned, so "why is mine not listed?" is answerable.
    assert result.ranking.considered == len(SCORES)


async def test_one_bad_symbol_costs_only_itself() -> None:
    """Vendors delist things. That must not void a screen someone paid for."""
    screener, _, _ = build(SCORES, broken="XAUUSD")
    result = await screener.execute(list(SCORES))

    assert [failure.symbol for failure in result.failures] == ["XAUUSD"]
    assert "no price data" in result.failures[0].error
    assert "XAUUSD" not in result.outcomes
    assert len(result.outcomes) == len(SCORES) - 1
    assert result.ranking.ranked


async def test_concurrency_is_bounded() -> None:
    """The cost of a screen has to be predictable, not merely large."""
    screener, consensus, _ = build(SCORES, concurrency=2)
    await screener.execute(list(SCORES))

    assert consensus.peak <= 2


async def test_duplicate_selections_are_screened_once() -> None:
    screener, consensus, _ = build(SCORES)
    result = await screener.execute(["nvda", "NVDA", " nvda ", "AAPL"])

    assert consensus.calls.count("NVDA") == 1
    assert result.requested == ["NVDA", "AAPL"]


async def test_the_basket_is_capped() -> None:
    screener, consensus, _ = build(SCORES, max_symbols=2)
    result = await screener.execute(list(SCORES))

    assert len(consensus.calls) == 2
    assert result.requested == list(SCORES)[:2]


async def test_coverage_reflects_the_agents_that_were_asked_for() -> None:
    """Three voters out of six requested is thinner evidence than three of three."""
    screener, _, ranker = build({"NVDA": 0.9})
    await screener.execute(["NVDA"], agents=["technical_analysis", "chart_pattern"])

    # The stub always returns three votes; two agents were requested, so the
    # pool is complete and coverage is capped at 1.0 rather than exceeding it.
    assert ranker.candidates[0].components.coverage == 1.0

    screener, _, ranker = build({"NVDA": 0.9})
    await screener.execute(["NVDA"])  # all six agents
    assert ranker.candidates[0].components.coverage == 0.5


async def test_candidates_carry_the_catalog_name_and_class() -> None:
    screener, _, ranker = build({"EURUSD": 0.8})
    await screener.execute(["EURUSD"])

    candidate = ranker.candidates[0]
    assert candidate.name == "Euro / US Dollar"
    assert candidate.asset_class.value == "forex"


async def test_an_empty_basket_produces_an_empty_screen_not_an_error() -> None:
    screener, consensus, _ = build(SCORES)
    result = await screener.execute(["", "   "])

    assert consensus.calls == []
    assert result.outcomes == {}
    assert result.ranking.ranked == ()


class FailingLLM:
    async def generate_structured(self, response_model: Any, *, system: str, prompt: str) -> Any:
        msg = "provider is down"
        raise RuntimeError(msg)


async def test_a_failed_ranker_still_yields_a_usable_screen() -> None:
    """End to end: sixty LLM calls must not be thrown away by the sixty-first."""
    consensus = StubConsensus(SCORES)
    screener = ScreenOpportunities(
        cast(ReachConsensus, consensus),
        OpportunityRankingAgent(cast(Any, FailingLLM())),
        concurrency=3,
        default_top_n=3,
    )
    result = await screener.execute(list(SCORES))

    assert len(result.ranking.ranked) == 3
    assert not result.ranking.ranked_by_agent
    assert len(result.outcomes) == len(SCORES)


class RecordingLLM:
    def __init__(self) -> None:
        self.prompt = ""

    async def generate_structured(self, response_model: Any, *, system: str, prompt: str) -> Any:
        self.prompt = prompt
        return RankingAssessment(ordering=(), narrative="nothing compelling")


async def test_the_ranker_sees_every_symbol_and_its_votes() -> None:
    llm = RecordingLLM()
    consensus = StubConsensus(SCORES)
    screener = ScreenOpportunities(
        cast(ReachConsensus, consensus),
        OpportunityRankingAgent(cast(Any, llm)),
        concurrency=3,
    )
    await screener.execute(["NVDA", "EURUSD"])

    assert "NVDA" in llm.prompt
    assert "EURUSD" in llm.prompt
    assert "technical_analysis" in llm.prompt
