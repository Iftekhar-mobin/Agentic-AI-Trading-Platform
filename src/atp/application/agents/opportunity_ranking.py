"""Opportunity Ranking agent.

The only agent that reads across symbols. Every other specialist answers "what
about this instrument?"; this one answers "of these, which few?" - the question
capital allocation actually turns on, and the one no single-symbol agent can
reach because it never sees a second symbol.

It is given each candidate's consensus outcome, the deterministic composite and
its parts, and a few factual highlights lifted from the reports. It is not given
the full analyses: the comparison is the job, and a prompt carrying six complete
agent reports per symbol would be both enormous and an invitation to re-derive
conclusions that have already been reached.

**The ordering is checked, not trusted.** Symbols the model invents are dropped,
duplicates collapse to their first appearance, and candidates it omitted are
appended in composite order without a verdict rather than being lost. If the
call fails outright, the screen still returns a leaderboard - the deterministic
one - flagged as such. A ranking that silently changes shape when a model
misbehaves would be worse than no ranking at all.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import structlog
from pydantic import BaseModel, Field

from atp.domain.models.ranking import (
    Candidate,
    OpportunityRanking,
    OpportunityVerdict,
    RankedOpportunity,
)
from atp.domain.ports.llm import LLMClient

log = structlog.get_logger()

AGENT_NAME = "opportunity_ranking"

SYSTEM_PROMPT = """\
You are the Opportunity Ranking Agent of an AI trading platform. Traders use \
your output as decision support, never as guaranteed predictions.

You receive several candidates. Each has already been analysed by a pool of \
specialist agents and, where configured, an external trading bot; their votes \
have been counted by a deterministic policy, the winning side has been put \
through a risk gate, and a deterministic composite score has been computed with \
its components exposed. Your job is comparative: order these candidates by how \
much they deserve a trader's limited attention and capital today.

Rules:
- Never invent or recompute numbers. Cite only values present in the input.
- Rank only the symbols provided, each at most once, strongest first. Write up \
at least "shortlist_size" of them, or all of them when fewer were given.
- A weak basket is expressed through low "profit_potential" scores and a blunt \
narrative, never by returning a short list. The trader chose these instruments \
and is owed a read on each of the leaders, including "this is the best of a bad \
set and still not worth taking".
- The composite is a strong prior, not an instruction. Depart from its order \
when the evidence justifies it, and say plainly in your reasoning why - for \
example a marginally lower composite whose timeframes all align, against a \
higher one resting on a single agent.
- "profit_potential" is relative to this basket only. It is not a forecast \
return and must not be presented as one.
- Prefer independent ideas. Two candidates driven by the same underlying move \
(two dollar pairs, an index and its largest constituent) are close to one \
position held twice; note the overlap rather than ranking both highly in \
silence.
- Weigh what could not be analysed. Low coverage means agents abstained or \
failed, so the evidence is thinner than the headline score suggests.
- A rejected risk verdict means the trade cannot be placed as proposed. It may \
still be an interesting observation; it is not an actionable opportunity.
- Every evidence item's "source" must be a candidate symbol from the input.
- Confidence is calibrated 0-1. A basket of weak, mutually contradictory or \
thinly covered candidates must lower it.
- Invalidation conditions must be concrete and observable.
- The narrative describes the basket as a whole: what it is collectively \
positioned for, any concentration among the leaders, and whether anything here \
is worth acting on at all. "Nothing in this set is compelling" is a valid and \
often correct answer - say it there, and still rank the shortlist.
"""


class RankingAssessment(BaseModel):
    """What the model returns: an ordering, plus a read of the basket."""

    ordering: tuple[OpportunityVerdict, ...] = Field(
        default=(), description="Strongest first; a subset of the candidates given"
    )
    narrative: str = Field(default="", description="The basket taken as a whole")


class OpportunityRankingAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def rank(
        self, candidates: Sequence[Candidate], *, limit: int | None = None
    ) -> OpportunityRanking:
        """Order ``candidates``, best first.

        ``limit`` truncates the returned leaderboard. The model still sees every
        candidate: it cannot tell you the top five of a set it was only shown
        three of.
        """
        ordered = _by_composite(candidates)
        if not ordered:
            return OpportunityRanking(ranked=(), considered=0, ranked_by_agent=False)

        try:
            assessment = await self._llm.generate_structured(
                RankingAssessment,
                system=SYSTEM_PROMPT,
                prompt=json.dumps(_payload(ordered, limit), sort_keys=True, default=str),
            )
        except Exception as exc:  # a failed ranker must never fail the screen
            log.warning(
                "opportunity_ranking.failed",
                candidates=len(ordered),
                error=str(exc),
                detail="falling back to the deterministic composite order",
            )
            return _deterministic(ordered, limit)

        verdicts = _validate(assessment.ordering, ordered)
        rows = _merge(verdicts, ordered)
        log.info(
            "opportunity_ranking.completed",
            candidates=len(ordered),
            ranked_by_agent=len(verdicts),
            leader=rows[0].symbol if rows else None,
        )
        return OpportunityRanking(
            ranked=_truncate(rows, limit),
            narrative=assessment.narrative,
            considered=len(ordered),
            ranked_by_agent=True,
        )


def _by_composite(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Deterministic baseline order: composite descending, symbol as tiebreak.

    The tiebreak is alphabetical rather than input order so two screens over the
    same set produce the same list whatever order the boxes were ticked in.
    """
    return sorted(candidates, key=lambda c: (-c.components.composite, c.symbol))


def _payload(candidates: Sequence[Candidate], limit: int | None) -> dict[str, object]:
    return {
        "candidate_count": len(candidates),
        # How many the interface will show. The model is told rather than
        # left to guess, because "rank the ones worth ranking" is how you get
        # an empty table on a quiet day - the shortlist is the deliverable.
        "shortlist_size": min(limit, len(candidates)) if limit else len(candidates),
        "candidates": [
            {
                "symbol": candidate.symbol,
                "name": candidate.name,
                "asset_class": candidate.asset_class.value,
                "consensus_action": candidate.action.value,
                "consensus_score": candidate.score,
                "agreement": candidate.agreement,
                "quorum_met": candidate.quorum_met,
                "voter_count": len(candidate.votes),
                "composite": candidate.components.composite,
                "composite_components": candidate.components.model_dump(mode="json"),
                "risk_verdict": candidate.risk_verdict,
                "risk_violations": list(candidate.risk_violations),
                "bot_status": candidate.bot_status.value,
                "highlights": list(candidate.highlights),
                "votes": [
                    {
                        "voter": vote.voter,
                        "kind": vote.kind.value,
                        "direction": vote.direction.value,
                        "confidence": vote.confidence,
                        "rationale": vote.rationale[:300],
                    }
                    for vote in candidate.votes
                ],
            }
            for candidate in candidates
        ],
    }


def _validate(
    ordering: Sequence[OpportunityVerdict], candidates: Sequence[Candidate]
) -> list[OpportunityVerdict]:
    """Keep the verdicts that name a real candidate, once each, in model order."""
    known = {candidate.symbol for candidate in candidates}
    seen: set[str] = set()
    kept: list[OpportunityVerdict] = []
    for verdict in ordering:
        symbol = verdict.symbol.strip().upper()
        if symbol not in known:
            log.warning(
                "opportunity_ranking.unknown_symbol",
                symbol=verdict.symbol,
                detail="ranked a symbol that was not a candidate; dropped",
            )
            continue
        if symbol in seen:
            log.warning("opportunity_ranking.duplicate_symbol", symbol=symbol)
            continue
        seen.add(symbol)
        kept.append(verdict.model_copy(update={"symbol": symbol}))
    return kept


def _merge(
    verdicts: Sequence[OpportunityVerdict], candidates: Sequence[Candidate]
) -> list[RankedOpportunity]:
    """The agent's order first, then whatever it left out, by composite."""
    by_symbol = {candidate.symbol: candidate for candidate in candidates}
    ranked_symbols = {verdict.symbol for verdict in verdicts}

    rows: list[RankedOpportunity] = [
        RankedOpportunity(rank=index, candidate=by_symbol[verdict.symbol], verdict=verdict)
        for index, verdict in enumerate(verdicts, start=1)
    ]
    unranked = [candidate for candidate in candidates if candidate.symbol not in ranked_symbols]
    rows.extend(
        RankedOpportunity(rank=len(verdicts) + offset, candidate=candidate)
        for offset, candidate in enumerate(unranked, start=1)
    )
    return rows


def _deterministic(candidates: Sequence[Candidate], limit: int | None) -> OpportunityRanking:
    rows = [
        RankedOpportunity(rank=index, candidate=candidate)
        for index, candidate in enumerate(candidates, start=1)
    ]
    return OpportunityRanking(
        ranked=_truncate(rows, limit),
        narrative=(
            "The ranking agent was unavailable, so this order is the deterministic "
            "composite alone - conviction, agreement, confidence and coverage. "
            "No comparative judgement has been applied."
        ),
        considered=len(candidates),
        ranked_by_agent=False,
    )


def _truncate(
    rows: Sequence[RankedOpportunity], limit: int | None
) -> tuple[RankedOpportunity, ...]:
    return tuple(rows if limit is None else rows[:limit])
