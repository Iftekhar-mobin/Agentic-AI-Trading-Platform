"""OpenRouter adapter for the LLMClient port.

OpenRouter aggregates dozens of vendors behind one OpenAI-compatible API and,
usefully here, exposes a genuinely free tier: models whose id ends in
``:free``. That makes the whole platform runnable end to end — every agent, real
LLM reasoning — for the price of an account, which matters a great deal for
anyone evaluating it before committing to Anthropic spend.

The trade is quality and reliability. Free models are rate-limited, sometimes
queued behind paid traffic, and materially worse at holding to a schema. This
adapter therefore leans on ``structured`` to enforce the contract rather than
trusting ``response_format``, and translates every failure into
``LLMGenerationError`` so a weak model surfaces as a recorded agent failure.

Talks HTTP directly rather than pulling in the OpenAI SDK: the surface used
here is one endpoint, and httpx is already a dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import httpx
import structlog
from pydantic import BaseModel

from atp.domain.errors import LLMGenerationError
from atp.infrastructure.llm.structured import parse_structured, schema_instructions

if TYPE_CHECKING:
    from atp.infrastructure.config.settings import LLMSettings

log = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)

FREE_SUFFIX = ":free"


class OpenRouterLLMClient:
    def __init__(
        self,
        settings: LLMSettings,
        model: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._model = model
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            key = self._settings.openrouter_api_key
            if key is None:
                msg = (
                    "OpenRouter needs an API key - set ATP_LLM__OPENROUTER_API_KEY "
                    "(free to create at openrouter.ai/keys)"
                )
                raise LLMGenerationError(msg)
            self._client = httpx.AsyncClient(
                base_url=self._settings.openrouter_base_url.rstrip("/"),
                headers={
                    "Authorization": f"Bearer {key.get_secret_value()}",
                    # OpenRouter attributes traffic with these; they are optional
                    # but keep this app identifiable in a shared account.
                    "HTTP-Referer": "https://github.com/atp/agentic-trading-platform",
                    "X-Title": "Agentic AI Trading Platform",
                },
                timeout=httpx.Timeout(self._settings.request_timeout_seconds),
            )
        return self._client

    async def generate_structured(
        self,
        response_model: type[T],
        *,
        system: str,
        prompt: str,
    ) -> T:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": f"{system}\n\n{schema_instructions(response_model)}"},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self._settings.max_tokens,
            # Honoured where supported and ignored elsewhere; the real guarantee
            # is validation on the way out.
            "response_format": {"type": "json_object"},
        }

        try:
            response = await self._http().post("/chat/completions", json=payload)
        except httpx.RequestError as exc:
            msg = f"could not reach OpenRouter: {exc}"
            raise LLMGenerationError(msg) from exc

        if response.status_code == 429:
            msg = (
                f"OpenRouter rate-limited model '{self._model}'. Free models share a "
                "small quota; wait, or switch to a paid model or Ollama."
            )
            raise LLMGenerationError(msg)
        if response.status_code >= 400:
            msg = f"OpenRouter returned HTTP {response.status_code}: {response.text[:300]}"
            raise LLMGenerationError(msg)

        body = response.json()
        if error := body.get("error"):
            msg = f"OpenRouter error for '{self._model}': {error.get('message', error)}"
            raise LLMGenerationError(msg)

        choices = body.get("choices") or []
        if not choices:
            msg = f"OpenRouter returned no choices for '{self._model}'"
            raise LLMGenerationError(msg)
        content = choices[0].get("message", {}).get("content") or ""

        usage = body.get("usage") or {}
        log.debug(
            "llm.structured_response",
            provider="openrouter",
            model=self._model,
            response_model=response_model.__name__,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )
        return parse_structured(response_model, content, model=self._model)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
