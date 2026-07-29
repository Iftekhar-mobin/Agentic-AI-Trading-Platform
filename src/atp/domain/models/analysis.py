"""Technical analysis domain models.

Analysis is multi-timeframe: a report holds one entry per requested timeframe,
ordered highest to lowest. That order is the point — the higher timeframe sets
the context, and a signal on the lower one is only worth taking when it does
not fight the tide above it. The assessment is therefore singular and drawn
across all of them, not one verdict per timeframe glued together.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from atp.domain.models.explainability import Explanation
from atp.domain.models.market import BarInterval

MIN_HISTORY_BARS = 60
"""Minimum bars required for a meaningful indicator snapshot (longest lookback + warmup)."""


class SignalDirection(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class IndicatorReading(BaseModel):
    """One indicator's latest values plus its deterministic, rule-based classification."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, description="Stable identifier, e.g. 'rsi_14'")
    values: dict[str, float]
    direction: SignalDirection
    summary: str = Field(min_length=1, description="Human-readable one-liner with the numbers")


def net_direction(directions: list[SignalDirection]) -> SignalDirection:
    """Majority vote, with ties resolving to neutral.

    Deterministic on purpose: it gives every timeframe a machine-readable read
    that does not depend on an LLM, so timeframe agreement can be checked in
    code and the model's own call can be compared against it.
    """
    counts = Counter(
        direction for direction in directions if direction is not SignalDirection.NEUTRAL
    )
    if not counts:
        return SignalDirection.NEUTRAL
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return SignalDirection.NEUTRAL
    return ranked[0][0]


class TimeframeReadings(BaseModel):
    """Every indicator reading computed on one timeframe."""

    model_config = ConfigDict(frozen=True)

    interval: BarInterval
    as_of: datetime
    latest_close: float
    bars: int = Field(ge=0)
    readings: tuple[IndicatorReading, ...]
    signal_counts: dict[SignalDirection, int]
    direction: SignalDirection = Field(
        description="Deterministic majority read of this timeframe's indicators"
    )


class TechnicalAssessment(Explanation):
    """The LLM's interpretation of the deterministic readings (explainability enforced)."""

    direction: SignalDirection
    timeframe_alignment: str = Field(
        min_length=1,
        description="Whether the timeframes agree, and what to do when they do not",
    )


class TechnicalReport(BaseModel):
    """Full output of the Technical Analysis agent.

    ``timeframes`` is deterministic and reproducible; ``assessment`` is the LLM
    interpretation layered on top of it.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    timeframes: tuple[TimeframeReadings, ...] = Field(min_length=1)
    assessment: TechnicalAssessment

    @property
    def primary(self) -> TimeframeReadings:
        """The highest timeframe — the context every lower one is read against."""
        return self.timeframes[0]

    @property
    def intervals(self) -> tuple[BarInterval, ...]:
        return tuple(timeframe.interval for timeframe in self.timeframes)

    @property
    def as_of(self) -> datetime:
        return self.primary.as_of

    @property
    def latest_close(self) -> float:
        return self.primary.latest_close

    @property
    def aligned(self) -> bool:
        """True when every timeframe's deterministic read points the same way."""
        directions = {timeframe.direction for timeframe in self.timeframes}
        return len(directions) == 1
