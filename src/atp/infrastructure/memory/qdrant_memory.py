"""Qdrant-backed episodic memory — the production adapter.

Same port, same semantics as the JSON adapter, but the filtering and the
nearest-neighbour search happen in the database rather than in this process, so
recall stays fast as the journal grows.

Two details worth knowing:

- Qdrant point ids must be integers or UUIDs, while episode ids are readable
  strings. They are mapped through a deterministic UUID5, which is what makes
  ``remember`` an upsert rather than an append.
- Payloads carry the embedding model name, and recall filters on it. A model
  swap changes the vector space, and comparing across spaces returns confident
  nonsense; the old rows simply stop matching until they are re-embedded.

The collection is created on first use, sized from the embedding port.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient, models

from atp.domain.errors import RepositoryUnavailableError
from atp.domain.models.memory import EpisodeKind, EpisodeMatch, MemoryEpisode
from atp.domain.ports.embeddings import EmbeddingModel

log = structlog.get_logger()

DEFAULT_COLLECTION = "atp_episodes"

_EPISODE_NAMESPACE = uuid.UUID("6f6a1f9c-4a41-5f1e-9d2b-0c9a5b7e3d18")
"""Fixed namespace so an episode id always maps to the same point id."""


def point_id(episode_id: str) -> str:
    return str(uuid.uuid5(_EPISODE_NAMESPACE, episode_id))


class QdrantEpisodicMemory:
    def __init__(
        self,
        client: AsyncQdrantClient,
        embeddings: EmbeddingModel,
        *,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        self._client = client
        self._embeddings = embeddings
        self._collection = collection
        self._ready = False

    async def remember(self, episode: MemoryEpisode) -> None:
        (vector,) = await self._embeddings.embed([episode.summary])
        await self._ensure_collection()
        payload: dict[str, Any] = {
            "episode": episode.model_dump(mode="json"),
            "symbol": episode.symbol,
            "kind": episode.kind.value,
            "regime_label": episode.regime.label if episode.regime else None,
            "occurred_at": episode.occurred_at.isoformat(),
            "model": self._embeddings.name,
        }
        try:
            await self._client.upsert(
                collection_name=self._collection,
                points=[
                    models.PointStruct(
                        id=point_id(episode.id), vector=list(vector), payload=payload
                    )
                ],
            )
        except Exception as exc:  # qdrant raises transport-specific errors
            msg = f"could not write to Qdrant collection '{self._collection}': {exc}"
            raise RepositoryUnavailableError(msg) from exc
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
        (query_vector,) = await self._embeddings.embed([query])
        await self._ensure_collection()

        conditions: list[models.FieldCondition] = [
            models.FieldCondition(key="model", match=models.MatchValue(value=self._embeddings.name))
        ]
        if symbol is not None:
            conditions.append(
                models.FieldCondition(
                    key="symbol", match=models.MatchValue(value=symbol.strip().upper())
                )
            )
        if kinds:
            conditions.append(
                models.FieldCondition(
                    key="kind", match=models.MatchAny(any=[kind.value for kind in kinds])
                )
            )
        if regime_label is not None:
            conditions.append(
                models.FieldCondition(
                    key="regime_label", match=models.MatchValue(value=regime_label)
                )
            )

        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=list(query_vector),
                query_filter=models.Filter(must=conditions),
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:  # qdrant raises transport-specific errors
            msg = f"could not query Qdrant collection '{self._collection}': {exc}"
            raise RepositoryUnavailableError(msg) from exc

        matches: list[EpisodeMatch] = []
        for point in response.points:
            payload = point.payload or {}
            try:
                episode = MemoryEpisode.model_validate(payload["episode"])
            except (KeyError, ValueError):
                log.warning("memory.unreadable_point_skipped", point_id=str(point.id))
                continue
            matches.append(EpisodeMatch(episode=episode, score=_clamp(point.score)))
        return tuple(matches)

    async def _ensure_collection(self) -> None:
        if self._ready:
            return
        try:
            if not await self._client.collection_exists(self._collection):
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=models.VectorParams(
                        size=self._embeddings.dimensions,
                        distance=models.Distance.COSINE,
                    ),
                )
                log.info(
                    "memory.collection_created",
                    collection=self._collection,
                    dimensions=self._embeddings.dimensions,
                )
        except Exception as exc:  # qdrant raises transport-specific errors
            msg = f"Qdrant is unreachable (collection '{self._collection}'): {exc}"
            raise RepositoryUnavailableError(msg) from exc
        self._ready = True


def _clamp(score: float) -> float:
    """Cosine can drift a hair outside [-1, 1] through float error."""
    return max(-1.0, min(1.0, float(score)))
