"""Continuous Learning agent.

Runs after the analysis pool, with two inputs the other agents never see: the
deterministic regime tag for right now, and the episodes recalled from memory
about this symbol. Its job is the one thing a stateless analysis cannot do —
compare the present situation to what actually happened before.

The deterministic/interpretive split holds here too. Regime tagging, retrieval
and similarity ranking are code; the agent receives precomputed matches with
their scores and may not claim a precedent that is not in the list.

Its output is written back to memory, which is what closes the loop: today's
reflection is tomorrow's recalled episode.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import structlog

from atp.application.agents.evidence import warn_on_unknown_sources
from atp.domain.models.memory import (
    EpisodeMatch,
    JournalEntry,
    LearningReport,
    RegimeTag,
)
from atp.domain.ports.llm import LLMClient

log = structlog.get_logger()

AGENT_NAME = "continuous_learning"

SYSTEM_PROMPT = """\
You are the Continuous Learning Agent of an AI trading platform. Traders use \
your output as decision support, never as guaranteed predictions.

You receive: the current situation for one symbol (what the analysis agents \
concluded), a deterministically computed market regime, and past episodes \
recalled from the platform's own journal, each with a similarity score and the \
regime it happened in.

Rules:
- Every evidence item's "source" must be the id of one provided episode. If no \
episodes were recalled, cite the source "current_situation" instead and say \
plainly that you are reasoning without precedent.
- Never invent precedents. If the recalled episodes are weak matches (low \
scores) or few, say so and lower your confidence accordingly.
- regime_note must state how the current regime compares to the regimes the \
recalled episodes happened in, and what that does to their relevance. A \
precedent from a calm uptrend is weak evidence in a volatile downtrend.
- lessons must be transferable and actionable — what to do differently, or \
what to keep doing, stated so it would still make sense read six months from \
now against a different symbol. "Be careful" is not a lesson; "entries taken \
against the slow-SMA slope gave back gains in volatile regimes" is.
- Do not restate the analysis agents' conclusions. Your value is the \
comparison across time, not a summary of the present.
- Confidence is calibrated 0-1. Little or irrelevant history must keep it \
below 0.4.
- Invalidation conditions must be concrete and observable.
"""


class ContinuousLearningAgent:
    def __init__(
        self,
        llm: LLMClient,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._llm = llm
        self._clock = clock

    async def reflect(
        self,
        symbol: str,
        *,
        situation: str,
        regime: RegimeTag | None,
        recalled: Sequence[EpisodeMatch],
    ) -> LearningReport:
        symbol = symbol.strip().upper()
        now = self._clock()

        payload = {
            "symbol": symbol,
            "as_of": now.isoformat(),
            "current_situation": situation,
            "current_regime": (
                {
                    "label": regime.label,
                    "trend": regime.trend.value,
                    "volatility": regime.volatility.value,
                    "metrics": regime.metrics,
                }
                if regime is not None
                else None
            ),
            "recalled_episode_count": len(recalled),
            "recalled_episodes": [
                {
                    "id": match.episode.id,
                    "kind": match.episode.kind.value,
                    "occurred_at": match.episode.occurred_at.isoformat(),
                    "age_days": round(
                        (now - match.episode.occurred_at).total_seconds() / 86_400.0, 1
                    ),
                    "regime": match.episode.regime.label if match.episode.regime else None,
                    "similarity": round(match.score, 4),
                    "summary": match.episode.summary,
                    "metadata": match.episode.metadata,
                }
                for match in recalled
            ],
        }
        entry = await self._llm.generate_structured(
            JournalEntry,
            system=SYSTEM_PROMPT,
            prompt=json.dumps(payload, sort_keys=True, default=str),
        )

        known = {match.episode.id for match in recalled} | {"current_situation"}
        warn_on_unknown_sources(AGENT_NAME, symbol, entry.evidence, known)
        log.info(
            "continuous_learning.completed",
            symbol=symbol,
            regime=regime.label if regime else None,
            recalled=len(recalled),
            lessons=len(entry.lessons),
            confidence=entry.confidence,
        )
        return LearningReport(
            symbol=symbol,
            as_of=now,
            regime=regime,
            recalled=tuple(recalled),
            entry=entry,
        )
