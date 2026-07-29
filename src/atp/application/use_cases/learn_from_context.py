"""Use case: reflect on the current situation against remembered episodes.

Closes the learning loop in one pass: tag the regime, recall what happened
before, reflect, then write both the situation and the reflection back so the
next run has more to compare against.

Memory is treated as an enhancement, not a dependency. A regime that cannot be
computed, a store that is unreachable, a write that fails — each degrades the
result and is logged, but none of them costs the caller the reflection. The
alternative would be an analysis workflow that fails because a vector database
is down, which is a worse trade for a decision-support tool.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import structlog

from atp.application.agents.continuous_learning import ContinuousLearningAgent
from atp.application.use_cases.load_price_history import LoadPriceHistory
from atp.domain.errors import DomainError
from atp.domain.models.market import BarInterval
from atp.domain.models.memory import (
    EpisodeKind,
    EpisodeMatch,
    LearningReport,
    MemoryEpisode,
    RegimeTag,
)
from atp.domain.ports.memory import EpisodicMemory
from atp.domain.services.regime_detection import MIN_REGIME_BARS, detect_regime

log = structlog.get_logger()

DEFAULT_RECALL_LIMIT = 5

RECALL_KINDS = (EpisodeKind.ANALYSIS, EpisodeKind.TRADE, EpisodeKind.LESSON)


class LearnFromContext:
    def __init__(
        self,
        history: LoadPriceHistory,
        memory: EpisodicMemory,
        agent: ContinuousLearningAgent,
        *,
        recall_limit: int = DEFAULT_RECALL_LIMIT,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._history = history
        self._memory = memory
        self._agent = agent
        self._recall_limit = recall_limit
        self._clock = clock

    async def execute(
        self,
        symbol: str,
        interval: BarInterval = BarInterval.DAY_1,
        *,
        situation: str,
    ) -> LearningReport:
        symbol = symbol.strip().upper()
        regime = await self._tag_regime(symbol, interval)
        recalled = await self._recall(symbol, situation)

        report = await self._agent.reflect(
            symbol, situation=situation, regime=regime, recalled=recalled
        )
        await self._remember(report, situation)
        return report

    async def _tag_regime(self, symbol: str, interval: BarInterval) -> RegimeTag | None:
        try:
            history = await self._history.execute(symbol, interval, min_bars=MIN_REGIME_BARS)
            return detect_regime(history)
        except DomainError as exc:
            log.warning("learning.regime_unavailable", symbol=symbol, error=str(exc))
            return None

    async def _recall(self, symbol: str, situation: str) -> tuple[EpisodeMatch, ...]:
        try:
            return await self._memory.recall(
                situation, symbol=symbol, kinds=RECALL_KINDS, limit=self._recall_limit
            )
        except DomainError as exc:
            log.warning("learning.recall_failed", symbol=symbol, error=str(exc))
            return ()

    async def _remember(self, report: LearningReport, situation: str) -> None:
        """Write the situation and the reflection back as two separate episodes.

        Separate on purpose: a future recall looking for "what did the market
        look like" should not have to read past a reflection, and vice versa.
        Ids are deterministic, so a re-run overwrites instead of duplicating.
        """
        stamp = report.as_of.isoformat()
        entry = report.entry
        episodes = [
            MemoryEpisode(
                id=f"analysis:{report.symbol}:{stamp}",
                symbol=report.symbol,
                kind=EpisodeKind.ANALYSIS,
                occurred_at=report.as_of,
                summary=situation,
                regime=report.regime,
                metadata={"recalled": len(report.recalled)},
            ),
            MemoryEpisode(
                id=f"lesson:{report.symbol}:{stamp}",
                symbol=report.symbol,
                kind=EpisodeKind.LESSON,
                occurred_at=report.as_of,
                summary=" ".join([entry.reasoning, *entry.lessons]),
                regime=report.regime,
                metadata={
                    "confidence": entry.confidence,
                    "lessons": len(entry.lessons),
                    "regime_note": entry.regime_note,
                },
            ),
        ]
        for episode in episodes:
            try:
                await self._memory.remember(episode)
            except DomainError as exc:
                log.warning("learning.remember_failed", episode_id=episode.id, error=str(exc))
