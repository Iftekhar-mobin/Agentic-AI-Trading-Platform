"""Port for discovering and installing language models."""

from __future__ import annotations

from typing import Protocol

from atp.domain.models.llm import LLMProvider, ModelInfo, ModelPullResult


class ModelCatalog(Protocol):
    """Lists the models one provider offers, and installs them where that applies.

    Kept separate from ``LLMClient`` deliberately: generating a completion and
    administering which weights exist are different concerns with different
    permissions, and most providers implement only the first.
    """

    @property
    def provider(self) -> LLMProvider: ...

    @property
    def supports_pull(self) -> bool:
        """True for local providers, where a model must be downloaded before use."""
        ...

    async def list_models(self) -> tuple[ModelInfo, ...]:
        """Available models, cheapest-to-run first. Never raises on an empty catalog."""
        ...

    async def pull(self, model: str) -> ModelPullResult:
        """Download a model. Raises ``ModelUnavailableError`` when unsupported."""
        ...
