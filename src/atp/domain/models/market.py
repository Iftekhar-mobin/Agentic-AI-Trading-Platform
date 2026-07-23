"""Market data value objects.

Prices are ``float`` (market data feeds analytical/vectorized code, where float
is the norm); monetary amounts on orders and portfolios will use ``Decimal``
when those models arrive. All timestamps are timezone-aware and normalized to
UTC at the boundary — naive datetimes are rejected outright.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BarInterval(StrEnum):
    """Supported bar aggregation intervals."""

    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    HOUR_1 = "1h"
    DAY_1 = "1d"
    WEEK_1 = "1wk"


class Bar(BaseModel):
    """A single OHLCV bar. Immutable value object."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @field_validator("timestamp")
    @classmethod
    def _normalize_to_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            msg = "bar timestamps must be timezone-aware"
            raise ValueError(msg)
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _check_ohlc_consistency(self) -> Bar:
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            msg = (
                f"inconsistent OHLC bar at {self.timestamp.isoformat()}: "
                f"open={self.open} high={self.high} low={self.low} close={self.close}"
            )
            raise ValueError(msg)
        return self


class PriceHistory(BaseModel):
    """An ordered, gap-tolerant series of bars for one symbol at one interval."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1)
    interval: BarInterval
    bars: tuple[Bar, ...] = ()

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def _check_sorted_and_unique(self) -> PriceHistory:
        for earlier, later in pairwise(self.bars):
            if earlier.timestamp >= later.timestamp:
                msg = (
                    f"bars must be strictly ascending by timestamp; "
                    f"{earlier.timestamp.isoformat()} >= {later.timestamp.isoformat()}"
                )
                raise ValueError(msg)
        return self

    def __len__(self) -> int:
        return len(self.bars)

    @property
    def latest(self) -> Bar | None:
        return self.bars[-1] if self.bars else None
