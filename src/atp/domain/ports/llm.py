"""Port for large language model access."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    """Generates schema-validated structured output from a prompt.

    Implementations must return an instance of ``response_model`` or raise
    ``LLMGenerationError`` — free-text responses are never surfaced, which is
    how the explainability envelope stays code-enforced.
    """

    async def generate_structured(
        self,
        response_model: type[T],
        *,
        system: str,
        prompt: str,
    ) -> T: ...
