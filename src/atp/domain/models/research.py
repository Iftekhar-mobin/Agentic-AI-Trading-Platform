"""Market research domain models.

Where the technical agent asks "what is this instrument doing?", market
research asks "what is it doing *relative to the market*?" — a stock up 4% in a
month that rose 9% is not strong, and no amount of single-symbol analysis will
notice that.

Every comparison is computed against an explicit benchmark over an explicit
window, and both are recorded on the report, because "outperforming" is
meaningless without saying versus what and since when.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from atp.domain.models.analysis import SignalDirection
from atp.domain.models.explainability import Explanation
from atp.domain.models.market import BarInterval

MIN_RESEARCH_BARS = 70
"""Enough overlapping bars for a quarter-length comparison plus warmup."""


class MarketContextReading(BaseModel):
    """One market-relative measurement with its deterministic classification."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, description="Stable identifier, e.g. 'relative_strength_3m'")
    values: dict[str, float]
    direction: SignalDirection
    summary: str = Field(min_length=1, description="Human-readable one-liner with the numbers")


class MarketResearchAssessment(Explanation):
    """The LLM's reading of where this instrument sits within the market."""

    direction: SignalDirection


class MarketResearchReport(BaseModel):
    """Full output of the Market Research agent."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    benchmark: str
    interval: BarInterval
    as_of: datetime
    overlapping_bars: int = Field(ge=0)
    readings: tuple[MarketContextReading, ...]
    signal_counts: dict[SignalDirection, int]
    assessment: MarketResearchAssessment
