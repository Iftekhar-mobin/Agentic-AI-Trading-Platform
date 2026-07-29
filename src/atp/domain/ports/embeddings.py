"""Port for text embedding models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingModel(Protocol):
    """Turns text into fixed-length vectors for semantic recall.

    Implementations must return one vector per input text, in the same order,
    each of length ``dimensions`` and L2-normalized — the memory adapters
    compare with a dot product and rely on that normalization.
    """

    @property
    def name(self) -> str:
        """Identifier recorded alongside stored vectors; changing it invalidates them."""
        ...

    @property
    def dimensions(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...
