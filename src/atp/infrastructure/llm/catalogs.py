"""Model catalogs: what each provider offers, and installing it where that applies.

Three implementations of one port:

- **Anthropic** is a static list. There is no public "list models I can afford"
  endpoint worth calling per page load, and the useful set is small and known.
- **OpenRouter** is fetched live and filtered. Its catalog runs to hundreds of
  entries, so free models are surfaced first — that is the reason most people
  reach for it.
- **Ollama** lists what is actually downloaded, and can download more. It is
  the only provider where "available" and "installed" differ, which is why
  ``ModelInfo.installed`` exists.

Every catalog degrades to an empty list rather than raising: a Models page that
cannot reach one provider should still show the others.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import structlog

from atp.domain.errors import ModelUnavailableError
from atp.domain.models.llm import LLMProvider, ModelInfo, ModelPullResult
from atp.infrastructure.llm.openrouter_client import FREE_SUFFIX

if TYPE_CHECKING:
    from atp.infrastructure.config.settings import LLMSettings

log = structlog.get_logger()

ANTHROPIC_MODELS: tuple[tuple[str, str, int], ...] = (
    ("claude-opus-4-8", "Claude Opus 4.8", 200_000),
    ("claude-sonnet-4-5", "Claude Sonnet 4.5", 200_000),
    ("claude-haiku-4-5", "Claude Haiku 4.5", 200_000),
)

SUGGESTED_OLLAMA_MODELS: tuple[str, ...] = (
    "llama3.1:8b",
    "qwen2.5:7b",
    "mistral:7b",
    "gemma2:9b",
    "phi3:mini",
)
"""Sensible starting points: small enough to run on a laptop, good enough to
follow a JSON schema. Shown as download suggestions when nothing is installed."""


class AnthropicModelCatalog:
    """Static catalog — nothing to fetch, nothing to install."""

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.ANTHROPIC

    @property
    def supports_pull(self) -> bool:
        return False

    async def list_models(self) -> tuple[ModelInfo, ...]:
        return tuple(
            ModelInfo(
                id=identifier,
                provider=LLMProvider.ANTHROPIC,
                name=name,
                free=False,
                context_length=context,
            )
            for identifier, name, context in ANTHROPIC_MODELS
        )

    async def pull(self, model: str) -> ModelPullResult:
        msg = "Anthropic models are hosted; there is nothing to download"
        raise ModelUnavailableError(msg)


class OpenRouterModelCatalog:
    def __init__(self, settings: LLMSettings, *, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.OPENROUTER

    @property
    def supports_pull(self) -> bool:
        return False

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            # Listing needs no credentials, so the Models page works before a key
            # is configured — which is exactly when someone is shopping for one.
            self._client = httpx.AsyncClient(
                base_url=self._settings.openrouter_base_url.rstrip("/"),
                timeout=httpx.Timeout(30.0),
            )
        return self._client

    async def list_models(self) -> tuple[ModelInfo, ...]:
        try:
            response = await self._http().get("/models")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("models.openrouter_list_failed", error=str(exc))
            return ()

        entries: list[dict[str, Any]] = response.json().get("data") or []
        models = [self._to_model(entry) for entry in entries]
        usable = [model for model in models if model is not None]
        # Free first, then by name: the free tier is the reason to be here.
        usable.sort(key=lambda model: (not model.free, model.name.lower()))
        log.debug(
            "models.openrouter_listed",
            total=len(usable),
            free=sum(1 for model in usable if model.free),
        )
        return tuple(usable)

    @staticmethod
    def _to_model(entry: dict[str, Any]) -> ModelInfo | None:
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier:
            return None
        context = entry.get("context_length")
        return ModelInfo(
            id=identifier,
            provider=LLMProvider.OPENROUTER,
            name=str(entry.get("name") or identifier),
            free=identifier.endswith(FREE_SUFFIX),
            context_length=int(context) if isinstance(context, int | float) and context else None,
            description=(str(entry.get("description"))[:300] or None)
            if entry.get("description")
            else None,
        )

    async def pull(self, model: str) -> ModelPullResult:
        msg = "OpenRouter models are hosted; there is nothing to download"
        raise ModelUnavailableError(msg)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class OllamaModelCatalog:
    def __init__(self, settings: LLMSettings, *, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.OLLAMA

    @property
    def supports_pull(self) -> bool:
        return True

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._settings.ollama_base_url.rstrip("/"),
                timeout=httpx.Timeout(self._settings.ollama_pull_timeout_seconds),
            )
        return self._client

    async def list_models(self) -> tuple[ModelInfo, ...]:
        installed = await self._installed()
        installed_ids = {model.id for model in installed}
        # Suggestions are shown as not-installed entries so the page is useful
        # on a fresh machine, where the local list is empty.
        suggestions = tuple(
            ModelInfo(
                id=name,
                provider=LLMProvider.OLLAMA,
                name=name,
                free=True,
                installed=False,
                description="Suggested - not downloaded yet",
            )
            for name in SUGGESTED_OLLAMA_MODELS
            if name not in installed_ids
        )
        return installed + suggestions

    async def _installed(self) -> tuple[ModelInfo, ...]:
        try:
            response = await self._http().get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("models.ollama_list_failed", error=str(exc))
            return ()

        entries: list[dict[str, Any]] = response.json().get("models") or []
        models: list[ModelInfo] = []
        for entry in entries:
            name = entry.get("model") or entry.get("name")
            if not isinstance(name, str) or not name:
                continue
            size = entry.get("size")
            detail = f"{size / 1e9:.1f} GB" if isinstance(size, int | float) and size else None
            models.append(
                ModelInfo(
                    id=name,
                    provider=LLMProvider.OLLAMA,
                    name=name,
                    free=True,
                    installed=True,
                    description=detail,
                )
            )
        models.sort(key=lambda model: model.name.lower())
        return tuple(models)

    async def pull(self, model: str) -> ModelPullResult:
        """Download a model. Blocking and slow — gigabytes over the network."""
        model = model.strip()
        if not model:
            msg = "no model name given to pull"
            raise ModelUnavailableError(msg)

        log.info("models.ollama_pull_started", model=model)
        try:
            response = await self._http().post("/api/pull", json={"model": model, "stream": False})
        except httpx.RequestError as exc:
            msg = (
                f"could not reach Ollama at {self._settings.ollama_base_url}: {exc}. "
                "Is `ollama serve` running?"
            )
            raise ModelUnavailableError(msg) from exc

        if response.status_code >= 400:
            msg = (
                f"Ollama could not pull '{model}': HTTP {response.status_code} "
                f"{response.text[:200]}"
            )
            raise ModelUnavailableError(msg)

        body = response.json()
        status = str(body.get("status") or "")
        if error := body.get("error"):
            msg = f"Ollama could not pull '{model}': {error}"
            raise ModelUnavailableError(msg)

        installed = status.lower() == "success"
        log.info("models.ollama_pull_finished", model=model, installed=installed, status=status)
        return ModelPullResult(model=model, installed=installed, detail=status or None)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
