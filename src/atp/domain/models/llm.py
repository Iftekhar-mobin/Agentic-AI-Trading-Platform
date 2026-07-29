"""Language-model provider domain models.

The platform treats the model backend as a swappable detail — agents depend on
the ``LLMClient`` port and never learn which vendor answered. These models exist
so the *choice* of backend is itself first-class data: something an operator can
list, compare on cost, and switch at runtime without redeploying.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"
    """Hosted, paid, highest quality. The production default."""

    OPENROUTER = "openrouter"
    """Aggregator with a free tier — models whose id ends in ``:free``."""

    OLLAMA = "ollama"
    """Local inference. No API key, no per-token cost, no data leaving the box."""


class ModelInfo(BaseModel):
    """One selectable model."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    id: str = Field(min_length=1, description="Provider-specific model identifier")
    provider: LLMProvider
    name: str = Field(min_length=1, description="Human-readable label")
    free: bool = Field(
        default=False,
        description="No per-token cost: an OpenRouter ':free' model or a local one",
    )
    installed: bool | None = Field(
        default=None,
        description="Local providers only: whether the weights are downloaded",
    )
    context_length: int | None = Field(default=None, ge=1)
    description: str | None = None


class ActiveModel(BaseModel):
    """Which model the agents are currently using."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    provider: LLMProvider
    model: str = Field(min_length=1)

    def __str__(self) -> str:
        return f"{self.provider.value}:{self.model}"


class ModelPullResult(BaseModel):
    """Outcome of downloading a local model."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model: str = Field(min_length=1)
    installed: bool
    detail: str | None = None
