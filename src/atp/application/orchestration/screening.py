"""Use case: run consensus across a basket, then rank what came back.

This is the multi-symbol entry point. It fans consensus out over every selected
instrument, scores each outcome with the deterministic composite, and hands the
whole set to the ranking agent for the comparative call.

Two properties are load-bearing:

**A failing symbol costs only itself.** Vendors delist things, indices carry no
fundamentals, a network call times out. Each symbol is run independently and a
failure is recorded against that symbol; the other nine still rank. Losing an
entire screen because one ticker went bad would be an expensive way to learn
that ticker went bad.

**Concurrency is bounded.** A screen is the most expensive thing this platform
does - one LLM call per agent per symbol, plus one to rank - so ten symbols with
six agents is sixty round trips. Running them all at once would flood provider
rate limits and, on a local Ollama, simply thrash. The semaphore is what makes
the cost predictable rather than merely large.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import structlog
from pydantic import BaseModel, Field

from atp.application.agents.opportunity_ranking import OpportunityRankingAgent
from atp.application.orchestration.consensus import AGENT_ARTIFACTS, ConsensusResult, ReachConsensus
from atp.application.orchestration.state import TradingState
from atp.domain.llm_trace import attributed_to
from atp.domain.models.market import BarInterval
from atp.domain.models.ranking import Candidate, OpportunityRanking
from atp.domain.models.trading import RiskDecision
from atp.domain.services import ranking
from atp.domain.universe import lookup, normalize

log = structlog.get_logger()

MAX_HIGHLIGHTS = 4
"""Per symbol. Enough texture for the comparison, short enough that fifteen
candidates still fit in one prompt."""


class SymbolFailure(BaseModel):
    """A symbol that could not be screened at all."""

    symbol: str
    error: str


class ScreenResult(BaseModel):
    """The leaderboard, and every underlying verdict behind it."""

    ranking: OpportunityRanking
    outcomes: dict[str, ConsensusResult] = Field(
        default_factory=dict,
        description="Full consensus result per symbol, keyed by symbol - what the "
        "interface expands when a row is opened",
    )
    failures: list[SymbolFailure] = Field(default_factory=list)
    requested: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0


class ScreenOpportunities:
    def __init__(
        self,
        consensus: ReachConsensus,
        ranker: OpportunityRankingAgent,
        *,
        concurrency: int = 3,
        max_symbols: int = 15,
        default_top_n: int = 5,
    ) -> None:
        self._consensus = consensus
        self._ranker = ranker
        self._concurrency = max(1, concurrency)
        self._max_symbols = max(1, max_symbols)
        self._default_top_n = max(1, default_top_n)

    async def execute(
        self,
        symbols: Sequence[str],
        intervals: Sequence[BarInterval] | BarInterval = BarInterval.DAY_1,
        *,
        agents: Sequence[str] | None = None,
        top_n: int | None = None,
        stop_loss: float | None = None,
        expected_bot: str | None = None,
    ) -> ScreenResult:
        """Screen ``symbols`` and return the ranked shortlist.

        ``stop_loss`` is applied to every symbol, which is only meaningful for a
        basket of one; it is here so a single-symbol screen can still reach a
        sized proposal. Left out, the risk gate rejects for safety and the
        composite is penalised accordingly - correctly, since an unsizeable idea
        is not yet an opportunity.
        """
        requested = _unique(symbols)[: self._max_symbols]
        if len(_unique(symbols)) > self._max_symbols:
            log.warning(
                "screen.truncated",
                requested=len(_unique(symbols)),
                kept=self._max_symbols,
                detail="a screen costs one LLM call per agent per symbol",
            )

        started = time.perf_counter()
        selected = list(agents) if agents else list(AGENT_ARTIFACTS)
        expected_voters = len(selected) + (1 if expected_bot else 0)

        gate = asyncio.Semaphore(self._concurrency)

        async def run(symbol: str) -> ConsensusResult:
            async with gate:
                return await self._consensus.execute(
                    symbol,
                    intervals,
                    agents=selected,
                    stop_loss=stop_loss,
                    expected_bot=expected_bot,
                )

        settled = await asyncio.gather(
            *(run(symbol) for symbol in requested), return_exceptions=True
        )

        outcomes: dict[str, ConsensusResult] = {}
        failures: list[SymbolFailure] = []
        candidates: list[Candidate] = []
        for symbol, outcome in zip(requested, settled, strict=True):
            if isinstance(outcome, BaseException):
                log.warning("screen.symbol_failed", symbol=symbol, error=str(outcome))
                failures.append(SymbolFailure(symbol=symbol, error=str(outcome)))
                continue
            outcomes[symbol] = outcome
            candidates.append(_candidate(symbol, outcome, expected_voters=expected_voters))

        limit = top_n if top_n is not None else self._default_top_n
        # The comparative call is the ranking agent's, not any one symbol's;
        # naming it keeps it distinguishable in the run's LLM trace.
        with attributed_to("opportunity_ranking"):
            ranked = await self._ranker.rank(candidates, limit=limit)

        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        log.info(
            "screen.completed",
            requested=len(requested),
            screened=len(outcomes),
            failed=len(failures),
            ranked=len(ranked.ranked),
            duration_ms=duration_ms,
        )
        return ScreenResult(
            ranking=ranked,
            outcomes=outcomes,
            failures=failures,
            requested=requested,
            duration_ms=duration_ms,
        )


def _unique(symbols: Sequence[str]) -> list[str]:
    """Normalized, de-duplicated, in the order given.

    Ticking a box twice must not buy two runs of the same expensive workflow.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for symbol in symbols:
        normalized = normalize(symbol)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        kept.append(normalized)
    return kept


def _candidate(symbol: str, result: ConsensusResult, *, expected_voters: int) -> Candidate:
    asset = lookup(symbol)
    components = ranking.score(result.decision, risk=result.risk, expected_voters=expected_voters)
    return Candidate(
        symbol=symbol,
        name=asset.name,
        asset_class=asset.asset_class,
        action=result.decision.action,
        score=result.decision.score,
        agreement=result.decision.agreement,
        quorum_met=result.decision.quorum_met,
        votes=result.decision.votes,
        components=components,
        bot_status=result.bot_status,
        risk_verdict=result.risk.verdict.value if result.risk else None,
        risk_violations=_violations(result.risk),
        highlights=_highlights(result.state),
    )


def _violations(risk: RiskDecision | None) -> tuple[str, ...]:
    if risk is None:
        return ()
    return tuple(violation.rule for violation in risk.violations)


def _highlights(state: TradingState | None) -> tuple[str, ...]:
    """A few facts the score cannot carry, lifted verbatim from the reports.

    Verbatim matters: these go into a prompt, and paraphrasing an agent's
    conclusion on the way to another agent is how a platform ends up with
    findings nobody wrote.
    """
    if state is None:
        return ()

    notes: list[str] = []
    if technical := state.technical_report:
        notes.append(f"timeframe alignment: {technical.assessment.timeframe_alignment}")
    if patterns := state.chart_pattern_report:
        detected = [
            pattern.kind
            for frame in patterns.timeframes
            for pattern in frame.patterns
            if pattern.confirmed
        ]
        if detected:
            notes.append("confirmed patterns: " + ", ".join(sorted(set(detected))[:5]))
    if news := state.news_report:
        themes = list(news.assessment.key_themes)[:3]
        if themes:
            notes.append("news themes: " + ", ".join(themes))
    if state.failures:
        notes.append(
            "agents that could not report: "
            + ", ".join(sorted({failure.agent for failure in state.failures}))
        )
    return tuple(notes[:MAX_HIGHLIGHTS])
