"""File-backed episodic memory — the dev/offline default.

Brute-force cosine over every stored episode. That is O(n) per recall and
obviously not what a production memory looks like, which is exactly why the
Qdrant adapter exists behind the same port. What this buys is a memory layer
that works with no services running: the tests, the CLI and a laptop demo all
exercise the same recall semantics the production adapter implements.

Vectors are stored alongside the episode with the name of the model that
produced them, and a recall ignores rows embedded by a different model — mixing
vector spaces would return confident nonsense.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import structlog

from atp.domain.models.memory import EpisodeKind, EpisodeMatch, MemoryEpisode
from atp.domain.ports.embeddings import EmbeddingModel

log = structlog.get_logger()

Row = dict[str, Any]
"""One stored line: the episode, its vector, and the model that produced it."""


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Dot product; both sides are already L2-normalized by the embedding port."""
    if len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))


class JsonEpisodicMemory:
    def __init__(self, path: Path, embeddings: EmbeddingModel) -> None:
        self._path = path
        self._embeddings = embeddings
        self._lock = asyncio.Lock()

    async def remember(self, episode: MemoryEpisode) -> None:
        (vector,) = await self._embeddings.embed([episode.summary])
        record: Row = {
            "episode": episode.model_dump(mode="json"),
            "vector": list(vector),
            "model": self._embeddings.name,
        }
        async with self._lock:
            rows = self._read()
            # Upsert by id: a retried workflow must not duplicate its episode.
            rows = [row for row in rows if row.get("episode", {}).get("id") != episode.id]
            rows.append(record)
            self._write(rows)
        log.debug("memory.remembered", episode_id=episode.id, symbol=episode.symbol)

    async def recall(
        self,
        query: str,
        *,
        symbol: str | None = None,
        kinds: Sequence[EpisodeKind] | None = None,
        regime_label: str | None = None,
        limit: int = 5,
    ) -> tuple[EpisodeMatch, ...]:
        async with self._lock:
            rows = self._read()
        if not rows:
            return ()

        (query_vector,) = await self._embeddings.embed([query])
        wanted_kinds = {kind.value for kind in kinds} if kinds else None
        wanted_symbol = symbol.strip().upper() if symbol else None

        matches: list[EpisodeMatch] = []
        for row in rows:
            if row.get("model") != self._embeddings.name:
                continue  # different vector space; not comparable
            try:
                episode = MemoryEpisode.model_validate(row["episode"])
            except (KeyError, ValueError):
                log.warning("memory.unreadable_row_skipped", path=str(self._path))
                continue
            if wanted_symbol is not None and episode.symbol != wanted_symbol:
                continue
            if wanted_kinds is not None and episode.kind.value not in wanted_kinds:
                continue
            if regime_label is not None and (
                episode.regime is None or episode.regime.label != regime_label
            ):
                continue
            matches.append(EpisodeMatch(episode=episode, score=cosine(query_vector, row["vector"])))

        matches.sort(key=lambda match: (match.score, match.episode.occurred_at), reverse=True)
        return tuple(matches[:limit])

    def _read(self) -> list[Row]:
        if not self._path.exists():
            return []
        rows: list[Row] = []
        for line in self._path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("memory.corrupt_line_skipped", path=str(self._path))
        return rows

    def _write(self, rows: list[Row]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = "\n".join(json.dumps(row) for row in rows)
        self._path.write_text(payload + "\n" if payload else "", encoding="utf-8")
