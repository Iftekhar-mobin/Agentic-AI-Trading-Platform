"""The explainability envelope every agent output must carry.

This is the structural enforcement of the platform's no-black-box rule: agent
conclusions are Pydantic models that *require* reasoning, calibrated
confidence, cited evidence, and invalidation conditions. An agent that cannot
produce these cannot produce output at all.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Evidence(BaseModel):
    """A single supporting fact, citing the data source it came from."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(description="Name of the input the fact comes from (e.g. 'rsi_14')")
    statement: str = Field(min_length=1)


class Explanation(BaseModel):
    """Base envelope: why, how confident, based on what, and what would refute it."""

    model_config = ConfigDict(frozen=True)

    reasoning: str = Field(min_length=1)
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Calibrated 0-1; mixed or weak signals must lower this",
    )
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    invalidation_conditions: tuple[str, ...] = Field(
        min_length=1,
        description="Concrete observable conditions under which this conclusion is wrong",
    )
