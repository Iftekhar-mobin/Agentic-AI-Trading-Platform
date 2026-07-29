"""Ollama adapter for the LLMClient port — local inference, no API key.

The strongest privacy story available to this platform: prompts contain a
trader's positions and reasoning, and with Ollama none of it leaves the
machine. It is also the only backend with no marginal cost, which changes what
is reasonable to run — a six-agent workflow on every symbol in a watchlist is
untenable at hosted prices and fine locally.

Ollama supports genuine structured output: passing a JSON Schema as ``format``
constrains decoding, so schema adherence is far better than free hosted models
of similar size. Output is still validated on the way in.

Generation timeouts are generous on purpose. A 7B model on CPU can take a
minute or more per agent, which is slow but not an error.
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


class OllamaLLMClient:
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
            self._client = httpx.AsyncClient(
                base_url=self._settings.ollama_base_url.rstrip("/"),
                timeout=httpx.Timeout(self._settings.ollama_timeout_seconds),
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
            "stream": False,
            # Constrains decoding to the schema rather than merely asking for it.
            "format": response_model.model_json_schema(),
            "options": {"num_predict": self._settings.max_tokens},
        }

        try:
            response = await self._http().post("/api/chat", json=payload)
        except httpx.RequestError as exc:
            msg = (
                f"could not reach Ollama at {self._settings.ollama_base_url}: {exc}. "
                "Is `ollama serve` running?"
            )
            raise LLMGenerationError(msg) from exc

        if response.status_code == 404:
            msg = (
                f"Ollama does not have model '{self._model}'. Download it first "
                f"(`ollama pull {self._model}`, or use the Models page)."
            )
            raise LLMGenerationError(msg)
        if response.status_code >= 400:
            msg = f"Ollama returned HTTP {response.status_code}: {response.text[:300]}"
            raise LLMGenerationError(msg)

        body = response.json()
        if error := body.get("error"):
            msg = f"Ollama error for '{self._model}': {error}"
            raise LLMGenerationError(msg)

        content = (body.get("message") or {}).get("content") or ""
        log.debug(
            "llm.structured_response",
            provider="ollama",
            model=self._model,
            response_model=response_model.__name__,
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
        )
        return parse_structured(response_model, content, model=self._model)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
