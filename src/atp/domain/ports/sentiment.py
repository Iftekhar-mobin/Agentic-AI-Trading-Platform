"""Port for text sentiment classification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from atp.domain.models.sentiment import SentimentScore


class SentimentModel(Protocol):
    """Classifies financial text. Deterministic numbers only — no prose.

    ``score`` returns one ``SentimentScore`` per input text, in the same order,
    so callers can zip results back onto their sources.
    """

    @property
    def name(self) -> str:
        """Identifier recorded on reports for reproducibility (e.g. 'ProsusAI/finbert')."""
        ...

    async def score(self, texts: Sequence[str]) -> tuple[SentimentScore, ...]: ...
