"""Signals published by external trading bots.

An inbound signal is a claim from a system this platform does not control and
cannot audit. It is therefore treated as evidence, not instruction: it is
stored, it expires, and it becomes exactly one vote alongside the agents'.

Staleness is the property that matters most. A mechanical bot emits a signal for
the bar it fired on; acting on it twenty minutes later is acting on a different
market. Every signal therefore carries its own arrival time and is checked
against a configured freshness window before it is allowed to vote.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, field_validator

from atp.domain.models.analysis import SignalDirection


class BotSignal(BaseModel):
    """One decision published by an external bot."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1, max_length=12)
    direction: SignalDirection
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="How sure the bot is; weights its vote like any other",
    )
    source: str = Field(
        min_length=1,
        max_length=64,
        description="Which bot said it, e.g. 'mt5-ea-v3' - appears in the audit trail",
    )
    rationale: str = Field(default="", max_length=2000)
    strategy: str | None = Field(default=None, max_length=64)
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("received_at")
    @classmethod
    def _normalize_to_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            msg = "signal timestamps must be timezone-aware"
            raise ValueError(msg)
        return value.astimezone(UTC)

    def age(self, now: datetime) -> timedelta:
        return now - self.received_at

    def is_fresh(self, now: datetime, ttl: timedelta) -> bool:
        """Whether this signal still describes the current market.

        A signal from the future is rejected too: clock skew on a remote bot is
        common, and silently trusting it would extend the window indefinitely.
        """
        age = self.age(now)
        return timedelta(0) <= age <= ttl
