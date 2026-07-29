"""Chart pattern domain models.

Chart patterns are geometry over swing points, and geometry is arithmetic — so
detection is deterministic code and the LLM only interprets what was found.
That matters more here than anywhere else in the platform: pattern reading is
the part of technical analysis most prone to seeing faces in clouds, and a
model asked to "find patterns" in raw prices will always find some.

Every detected pattern carries the price levels that define it, so a reader can
check the claim against the chart, and a ``confirmed`` flag separating a shape
that has completed from one that is merely forming.

Detection runs per timeframe; a report holds one entry per requested timeframe,
ordered highest to lowest.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from atp.domain.models.analysis import SignalDirection
from atp.domain.models.explainability import Explanation
from atp.domain.models.market import BarInterval

MIN_PATTERN_BARS = 60
"""Fewer bars than this cannot hold enough swing points to form a pattern."""


class SwingKind(StrEnum):
    HIGH = "high"
    LOW = "low"


class LevelKind(StrEnum):
    SUPPORT = "support"
    RESISTANCE = "resistance"


class PatternKind(StrEnum):
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    UPTREND_STRUCTURE = "uptrend_structure"
    DOWNTREND_STRUCTURE = "downtrend_structure"
    RANGE_BREAKOUT = "range_breakout"
    RANGE_BREAKDOWN = "range_breakdown"


class SwingPoint(BaseModel):
    """A local extreme in price — the raw material every pattern is built from."""

    model_config = ConfigDict(frozen=True)

    kind: SwingKind
    index: int = Field(ge=0, description="Position in the bar series")
    timestamp: datetime
    price: float = Field(gt=0)


class PriceLevel(BaseModel):
    """A price that has been tested repeatedly, clustered from nearby swings."""

    model_config = ConfigDict(frozen=True)

    kind: LevelKind
    price: float = Field(gt=0)
    touches: int = Field(ge=2, description="Swing points that formed this level")
    last_touch: datetime
    distance_pct: float = Field(description="Signed distance from the latest close, in percent")


class ChartPattern(BaseModel):
    """One detected formation, with the levels that define it."""

    model_config = ConfigDict(frozen=True)

    kind: PatternKind
    direction: SignalDirection
    start: datetime
    end: datetime
    levels: dict[str, float] = Field(
        description="Named prices defining the shape (peaks, troughs, neckline, ...)"
    )
    confirmed: bool = Field(
        description="True when price has completed the pattern, e.g. closed through the neckline"
    )
    quality: float = Field(
        ge=0.0,
        le=1.0,
        description="How cleanly the shape fits its ideal (symmetry, touch count)",
    )
    summary: str = Field(min_length=1)


class TimeframePatterns(BaseModel):
    """Everything detected on one timeframe."""

    model_config = ConfigDict(frozen=True)

    interval: BarInterval
    as_of: datetime
    latest_close: float
    bars: int = Field(ge=0)
    swings: tuple[SwingPoint, ...]
    levels: tuple[PriceLevel, ...]
    patterns: tuple[ChartPattern, ...]
    signal_counts: dict[SignalDirection, int]
    direction: SignalDirection = Field(
        description="Deterministic net read of this timeframe's confirmed patterns"
    )


class PatternAssessment(Explanation):
    """The LLM's reading of the detected formations across timeframes."""

    direction: SignalDirection
    timeframe_alignment: str = Field(
        min_length=1,
        description="Whether the timeframes agree, and what to do when they do not",
    )


class ChartPatternReport(BaseModel):
    """Full output of the Chart Pattern agent.

    ``timeframes`` is deterministic and reproducible; ``assessment`` is the
    interpretation layered on top.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    timeframes: tuple[TimeframePatterns, ...] = Field(min_length=1)
    assessment: PatternAssessment

    @property
    def primary(self) -> TimeframePatterns:
        """The highest timeframe — the context every lower one is read against."""
        return self.timeframes[0]

    @property
    def intervals(self) -> tuple[BarInterval, ...]:
        return tuple(timeframe.interval for timeframe in self.timeframes)
