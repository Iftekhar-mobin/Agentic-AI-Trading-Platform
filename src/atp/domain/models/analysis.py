"""Technical analysis domain models."""

from __future__ import annotations

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


class TechnicalAssessment(Explanation):
    """The LLM's interpretation of the deterministic readings (explainability enforced)."""

    direction: SignalDirection


class TechnicalReport(BaseModel):
    """Full output of the Technical Analysis agent.

    ``readings`` and ``signal_counts`` are deterministic and reproducible;
    ``assessment`` is the LLM interpretation layered on top of them.
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    interval: BarInterval
    as_of: datetime
    latest_close: float
    readings: tuple[IndicatorReading, ...]
    signal_counts: dict[SignalDirection, int]
    assessment: TechnicalAssessment
