"""Episodic memory domain models.

The system's long-term memory is a set of **episodes**: what the situation was,
what was decided, why, and — when it is known — how it turned out. Episodes are
tagged with the market regime they happened in, because "this setup worked" is
only useful alongside "in a calm uptrend".

Retrieval is semantic (embedded ``summary`` text) but filtered structurally
(symbol, kind, regime), so a recall can ask a precise question: *what did we
conclude about this name the last time volatility looked like this?*
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from atp.domain.models.explainability import Explanation


class RegimeTrend(StrEnum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"


class RegimeVolatility(StrEnum):
    CALM = "calm"
    NORMAL = "normal"
    VOLATILE = "volatile"


class RegimeTag(BaseModel):
    """A deterministic, reproducible label for market conditions at a point in time."""

    model_config = ConfigDict(frozen=True)

    trend: RegimeTrend
    volatility: RegimeVolatility
    as_of: datetime
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="The numbers behind the label, so it can be audited",
    )

    @field_validator("as_of")
    @classmethod
    def _normalize_to_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            msg = "regime timestamps must be timezone-aware"
            raise ValueError(msg)
        return value.astimezone(UTC)

    @property
    def label(self) -> str:
        """Compact form used for filtering and display, e.g. ``uptrend/volatile``."""
        return f"{self.trend.value}/{self.volatility.value}"


class EpisodeKind(StrEnum):
    ANALYSIS = "analysis"
    """A completed analysis workflow: what the agents concluded."""

    TRADE = "trade"
    """An order that reached the broker, with the risk decision behind it."""

    LESSON = "lesson"
    """The learning agent's own reflection, written back for future recall."""


class MemoryEpisode(BaseModel):
    """One remembered event. ``summary`` is the text that gets embedded."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    symbol: str = Field(min_length=1, max_length=12)
    kind: EpisodeKind
    occurred_at: datetime
    summary: str = Field(min_length=1, description="Natural-language text; this is embedded")
    regime: RegimeTag | None = None
    metadata: dict[str, str | float | int | bool | None] = Field(default_factory=dict)

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("occurred_at")
    @classmethod
    def _normalize_to_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            msg = "episode timestamps must be timezone-aware"
            raise ValueError(msg)
        return value.astimezone(UTC)


class EpisodeMatch(BaseModel):
    """A recalled episode with its similarity score (1.0 == identical direction)."""

    model_config = ConfigDict(frozen=True)

    episode: MemoryEpisode
    score: float = Field(ge=-1.0, le=1.0)


class JournalEntry(Explanation):
    """The learning agent's reflection, with the envelope every agent carries.

    ``lessons`` are the transferable part — what to do differently next time —
    and are deliberately separate from ``reasoning``, which explains this
    specific situation.
    """

    lessons: tuple[str, ...] = Field(min_length=1)
    regime_note: str = Field(
        min_length=1,
        description="How the current regime should qualify the recalled precedents",
    )


class LearningReport(BaseModel):
    """Full output of the Continuous Learning agent."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    as_of: datetime
    regime: RegimeTag | None
    recalled: tuple[EpisodeMatch, ...]
    entry: JournalEntry
