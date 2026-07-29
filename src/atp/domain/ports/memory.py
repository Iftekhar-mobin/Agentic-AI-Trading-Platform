"""Port for episodic memory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from atp.domain.models.memory import EpisodeKind, EpisodeMatch, MemoryEpisode


class EpisodicMemory(Protocol):
    """Stores and semantically recalls past episodes.

    ``remember`` is idempotent on ``episode.id``: re-remembering the same
    episode overwrites rather than duplicating, so a retried workflow cannot
    stuff memory with copies of one event.

    ``recall`` filters structurally first (symbol, kind, regime) and ranks
    semantically within that filter, returning matches strongest-first.
    """

    async def remember(self, episode: MemoryEpisode) -> None: ...

    async def recall(
        self,
        query: str,
        *,
        symbol: str | None = None,
        kinds: Sequence[EpisodeKind] | None = None,
        regime_label: str | None = None,
        limit: int = 5,
    ) -> tuple[EpisodeMatch, ...]: ...
