"""Anthropic adapter for the LLMClient port.

Uses ``messages.parse`` with a Pydantic ``output_format`` so responses are
schema-validated by the SDK — the agent layer only ever sees typed models.
Vendor exceptions are translated into domain ``LLMGenerationError``s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import anthropic
import structlog
from pydantic import BaseModel, ValidationError

from atp.domain.errors import LLMGenerationError

if TYPE_CHECKING:
    from atp.infrastructure.config.settings import LLMSettings

log = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)


class AnthropicLLMClient:
    def __init__(
        self,
        settings: LLMSettings,
        *,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self._settings = settings
        self._client = client

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            api_key = (
                self._settings.anthropic_api_key.get_secret_value()
                if self._settings.anthropic_api_key
                else None  # SDK falls back to ANTHROPIC_API_KEY / auth profile
            )
            try:
                self._client = anthropic.AsyncAnthropic(api_key=api_key)
            except anthropic.AnthropicError as exc:
                msg = (
                    "Anthropic client could not be created - set ANTHROPIC_API_KEY "
                    "or ATP_LLM__ANTHROPIC_API_KEY"
                )
                raise LLMGenerationError(msg) from exc
        return self._client

    async def generate_structured(
        self,
        response_model: type[T],
        *,
        system: str,
        prompt: str,
    ) -> T:
        try:
            response = await self._get_client().messages.parse(
                model=self._settings.model,
                max_tokens=self._settings.max_tokens,
                system=system,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
                output_format=response_model,
            )
        except anthropic.APIError as exc:
            msg = f"LLM request failed: {exc}"
            raise LLMGenerationError(msg) from exc
        except TypeError as exc:
            # The SDK raises TypeError at request time when no credentials resolve.
            msg = (
                "Anthropic credentials missing - set ANTHROPIC_API_KEY or "
                f"ATP_LLM__ANTHROPIC_API_KEY ({exc})"
            )
            raise LLMGenerationError(msg) from exc
        except ValidationError as exc:
            msg = f"LLM response failed schema validation for {response_model.__name__}"
            raise LLMGenerationError(msg) from exc

        if response.stop_reason == "refusal":
            msg = "the model declined the request (stop_reason=refusal)"
            raise LLMGenerationError(msg)
        parsed = response.parsed_output
        if parsed is None:
            msg = f"no structured output returned (stop_reason={response.stop_reason})"
            raise LLMGenerationError(msg)

        log.debug(
            "llm.structured_response",
            model=self._settings.model,
            response_model=response_model.__name__,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return parsed
