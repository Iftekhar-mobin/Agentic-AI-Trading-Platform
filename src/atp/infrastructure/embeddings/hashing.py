"""Deterministic hashing embeddings — the offline default.

A hashed bag of unigrams and bigrams, sign-trick folded into a fixed number of
dimensions and L2-normalized. It captures lexical overlap and nothing deeper:
"guidance cut" and "lowered outlook" are near-orthogonal to it, where a real
sentence-embedding model would place them close together.

That is a deliberate trade. Semantic recall behind a port means the storage,
filtering and ranking machinery can be built and tested end to end with zero
network and zero model downloads; swapping in a hosted embedding provider is a
new adapter and one line in the composition root. What must not happen is the
memory layer silently depending on a vendor being reachable.

Hashing uses blake2b rather than the builtin ``hash``, whose seed is randomized
per process — vectors have to stay comparable across restarts.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator, Sequence
from itertools import pairwise

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'\-\.]*")

DEFAULT_DIMENSIONS = 256


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _features(text: str) -> Iterator[str]:
    """Unigrams plus bigrams; bigrams give the vector a little word-order sense."""
    tokens = _tokens(text)
    yield from tokens
    for earlier, later in pairwise(tokens):
        yield f"{earlier}_{later}"


class HashingEmbeddingModel:
    """EmbeddingModel implementation with no dependencies and no network."""

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        if dimensions < 8:
            msg = f"dimensions must be at least 8, got {dimensions}"
            raise ValueError(msg)
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return f"hashing-v1-{self._dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        # Pure CPU string work on short summaries; no thread offload needed.
        return tuple(self.embed_text(text) for text in texts)

    def embed_text(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self._dimensions
        for feature in _features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # Empty or unhashable text: a zero vector scores 0 against everything,
            # which is the correct "no signal" answer rather than a false match.
            return tuple(vector)
        return tuple(value / norm for value in vector)
