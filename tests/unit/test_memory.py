"""Tests for the embedding model and the file-backed episodic memory."""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from factories import make_regime

from atp.domain.models.memory import EpisodeKind, MemoryEpisode, RegimeTrend
from atp.infrastructure.embeddings import HashingEmbeddingModel
from atp.infrastructure.memory import JsonEpisodicMemory
from atp.infrastructure.memory.json_memory import cosine
from atp.infrastructure.memory.qdrant_memory import point_id

NOW = datetime(2026, 7, 22, tzinfo=UTC)


def episode(
    identifier: str,
    summary: str,
    *,
    symbol: str = "AAPL",
    kind: EpisodeKind = EpisodeKind.ANALYSIS,
    trend: RegimeTrend = RegimeTrend.UPTREND,
    age_days: float = 0.0,
) -> MemoryEpisode:
    return MemoryEpisode(
        id=identifier,
        symbol=symbol,
        kind=kind,
        occurred_at=NOW - timedelta(days=age_days),
        summary=summary,
        regime=make_regime(trend=trend),
    )


@pytest.fixture
def memory(tmp_path: Path) -> JsonEpisodicMemory:
    return JsonEpisodicMemory(tmp_path / "memory.jsonl", HashingEmbeddingModel(128))


# --- Hashing embeddings -----------------------------------------------------


async def test_embeddings_are_deterministic_across_instances() -> None:
    """Vectors must survive a restart, so hashing cannot use the salted builtin."""
    first = await HashingEmbeddingModel(64).embed(["guidance cut on weak demand"])
    second = await HashingEmbeddingModel(64).embed(["guidance cut on weak demand"])
    assert first == second


async def test_embeddings_are_unit_length() -> None:
    (vector,) = await HashingEmbeddingModel(64).embed(["revenue beat expectations"])
    assert math.sqrt(sum(value**2 for value in vector)) == pytest.approx(1.0)


async def test_embedding_dimensions_are_honoured() -> None:
    model = HashingEmbeddingModel(32)
    (vector,) = await model.embed(["text"])
    assert model.dimensions == 32
    assert len(vector) == 32


async def test_similar_text_scores_higher_than_unrelated_text() -> None:
    model = HashingEmbeddingModel(512)
    base, similar, unrelated = await model.embed(
        [
            "Apple beats earnings estimates as revenue surges",
            "Apple beats earnings estimates as revenue climbs",
            "Central bank holds interest rates steady in Europe",
        ]
    )
    assert cosine(base, similar) > cosine(base, unrelated)


async def test_empty_text_yields_a_zero_vector() -> None:
    (vector,) = await HashingEmbeddingModel(32).embed([""])
    assert all(value == 0.0 for value in vector)
    assert cosine(vector, vector) == 0.0


async def test_model_name_encodes_dimensions() -> None:
    assert HashingEmbeddingModel(128).name == "hashing-v1-128"


def test_tiny_dimension_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        HashingEmbeddingModel(4)


# --- JSON episodic memory ---------------------------------------------------


async def test_recall_on_empty_memory_returns_nothing(memory: JsonEpisodicMemory) -> None:
    assert await memory.recall("anything") == ()


async def test_remembered_episode_is_recalled(memory: JsonEpisodicMemory) -> None:
    await memory.remember(episode("e1", "Technical bullish on a moving-average crossover"))
    matches = await memory.recall("moving-average crossover")

    assert len(matches) == 1
    assert matches[0].episode.id == "e1"
    assert matches[0].score > 0


async def test_recall_ranks_by_similarity(memory: JsonEpisodicMemory) -> None:
    await memory.remember(episode("close", "Technical bullish on a moving average crossover"))
    await memory.remember(episode("far", "Regulatory probe into advertising practices"))

    matches = await memory.recall("moving average crossover signal")

    assert [match.episode.id for match in matches] == ["close", "far"]
    assert matches[0].score > matches[1].score


async def test_remember_is_idempotent_on_id(memory: JsonEpisodicMemory) -> None:
    """A retried workflow must not stuff memory with copies of one event."""
    await memory.remember(episode("same", "First version of the summary"))
    await memory.remember(episode("same", "Second version of the summary"))

    matches = await memory.recall("summary")
    assert len(matches) == 1
    assert matches[0].episode.summary == "Second version of the summary"


async def test_recall_filters_by_symbol(memory: JsonEpisodicMemory) -> None:
    await memory.remember(episode("a", "Bullish crossover", symbol="AAPL"))
    await memory.remember(episode("m", "Bullish crossover", symbol="MSFT"))

    matches = await memory.recall("crossover", symbol="msft")
    assert [match.episode.id for match in matches] == ["m"]


async def test_recall_filters_by_kind(memory: JsonEpisodicMemory) -> None:
    await memory.remember(episode("a", "Bullish crossover", kind=EpisodeKind.ANALYSIS))
    await memory.remember(episode("t", "Bullish crossover", kind=EpisodeKind.TRADE))

    matches = await memory.recall("crossover", kinds=[EpisodeKind.TRADE])
    assert [match.episode.id for match in matches] == ["t"]


async def test_recall_filters_by_regime(memory: JsonEpisodicMemory) -> None:
    await memory.remember(episode("up", "Bullish crossover", trend=RegimeTrend.UPTREND))
    await memory.remember(episode("down", "Bullish crossover", trend=RegimeTrend.DOWNTREND))

    matches = await memory.recall("crossover", regime_label="downtrend/normal")
    assert [match.episode.id for match in matches] == ["down"]


async def test_recall_respects_the_limit(memory: JsonEpisodicMemory) -> None:
    for index in range(6):
        await memory.remember(episode(f"e{index}", f"Bullish crossover number {index}"))

    assert len(await memory.recall("crossover", limit=3)) == 3


async def test_rows_from_another_embedding_model_are_ignored(tmp_path: Path) -> None:
    """Comparing across vector spaces would return confident nonsense."""
    path = tmp_path / "memory.jsonl"
    await JsonEpisodicMemory(path, HashingEmbeddingModel(64)).remember(
        episode("old", "Bullish crossover")
    )

    rebuilt = JsonEpisodicMemory(path, HashingEmbeddingModel(128))
    assert await rebuilt.recall("crossover") == ()


async def test_corrupt_lines_are_skipped_not_fatal(
    tmp_path: Path, memory: JsonEpisodicMemory
) -> None:
    await memory.remember(episode("good", "Bullish crossover"))
    path = tmp_path / "memory.jsonl"
    path.write_text(path.read_text("utf-8") + "{not json\n", encoding="utf-8")

    matches = await memory.recall("crossover")
    assert [match.episode.id for match in matches] == ["good"]


async def test_concurrent_writes_do_not_lose_episodes(memory: JsonEpisodicMemory) -> None:
    await asyncio.gather(
        *(memory.remember(episode(f"e{index}", f"Episode {index}")) for index in range(8))
    )
    matches = await memory.recall("Episode", limit=20)
    assert len(matches) == 8


async def test_regime_survives_a_round_trip(memory: JsonEpisodicMemory) -> None:
    await memory.remember(episode("e1", "Bullish crossover"))
    (match,) = await memory.recall("crossover")

    assert match.episode.regime is not None
    assert match.episode.regime.label == "uptrend/normal"
    assert match.episode.regime.metrics["sma_fast"] == 320.0


# --- Qdrant id mapping ------------------------------------------------------


def test_qdrant_point_ids_are_deterministic_uuids() -> None:
    """The upsert semantics depend on one episode id always mapping to one point."""
    assert point_id("trade:AAPL:order-1") == point_id("trade:AAPL:order-1")
    assert point_id("trade:AAPL:order-1") != point_id("trade:AAPL:order-2")
    assert len(point_id("x").split("-")) == 5
