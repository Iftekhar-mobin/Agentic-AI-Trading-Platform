"""Tests for the OpenRouter and Ollama adapters, catalogs, and the runtime router."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from atp.domain.errors import LLMGenerationError, ModelUnavailableError
from atp.domain.models.llm import ActiveModel, LLMProvider
from atp.infrastructure.config import LLMSettings
from atp.infrastructure.llm import (
    AnthropicModelCatalog,
    LLMRouter,
    OllamaLLMClient,
    OllamaModelCatalog,
    OpenRouterLLMClient,
    OpenRouterModelCatalog,
)
from atp.infrastructure.llm.structured import extract_json, parse_structured


class Answer(BaseModel):
    verdict: str
    score: float


def settings(**overrides: Any) -> LLMSettings:
    defaults: dict[str, Any] = {"openrouter_api_key": "test-key", "max_tokens": 512}
    return LLMSettings(**{**defaults, **overrides})


def transport(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url="http://testserver", transport=httpx.MockTransport(handler))


def openrouter_reply(content: str, *, status: int = 200) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, text=content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    return handler


def ollama_reply(content: str, *, status: int = 200) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, text=content)
        return httpx.Response(200, json={"message": {"content": content}})

    return handler


# --- Structured-output enforcement ------------------------------------------


def test_plain_json_is_extracted() -> None:
    assert extract_json('{"a": 1}') == '{"a": 1}'


def test_markdown_fences_are_stripped() -> None:
    """Small models fence their JSON no matter how firmly you ask them not to."""
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('```\n{"a": 1}\n```') == '{"a": 1}'


def test_surrounding_prose_is_discarded() -> None:
    assert extract_json('Sure! Here is the JSON:\n{"a": 1}\nHope that helps.') == '{"a": 1}'


def test_valid_json_parses_into_the_model() -> None:
    answer = parse_structured(Answer, '{"verdict": "buy", "score": 0.7}', model="m")
    assert answer.verdict == "buy"


def test_non_json_output_is_a_domain_error() -> None:
    with pytest.raises(LLMGenerationError, match="did not return JSON"):
        parse_structured(Answer, "I think you should buy it.", model="tiny-model")


def test_an_empty_response_is_a_domain_error() -> None:
    with pytest.raises(LLMGenerationError, match="empty response"):
        parse_structured(Answer, "   ", model="tiny-model")


def test_json_missing_required_fields_is_rejected() -> None:
    """A model that drops `evidence` must fail, not produce a half-report."""
    with pytest.raises(LLMGenerationError, match="does not match Answer"):
        parse_structured(Answer, '{"verdict": "buy"}', model="tiny-model")


# --- OpenRouter client ------------------------------------------------------


async def test_openrouter_returns_a_validated_model() -> None:
    client = OpenRouterLLMClient(
        settings(),
        "deepseek/deepseek-chat-v3.1:free",
        client=transport(openrouter_reply('{"verdict": "buy", "score": 0.7}')),
    )

    answer = await client.generate_structured(Answer, system="s", prompt="p")
    assert answer.score == 0.7


async def test_openrouter_sends_the_model_and_schema() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"verdict": "b", "score": 1}'}}]}
        )

    client = OpenRouterLLMClient(
        settings(), "meta-llama/llama-3.3-70b-instruct:free", client=transport(handler)
    )
    await client.generate_structured(Answer, system="SYSTEM", prompt="PROMPT")

    assert captured["model"] == "meta-llama/llama-3.3-70b-instruct:free"
    assert captured["response_format"] == {"type": "json_object"}
    # The schema is restated in the prompt for models that ignore response_format.
    assert "JSON Schema" in captured["messages"][0]["content"]
    assert "SYSTEM" in captured["messages"][0]["content"]
    assert captured["messages"][1]["content"] == "PROMPT"


async def test_openrouter_fenced_output_still_validates() -> None:
    client = OpenRouterLLMClient(
        settings(),
        "free/model:free",
        client=transport(openrouter_reply('```json\n{"verdict": "sell", "score": 0.2}\n```')),
    )
    answer = await client.generate_structured(Answer, system="s", prompt="p")
    assert answer.verdict == "sell"


async def test_openrouter_repairs_a_schema_violation_on_a_second_call() -> None:
    """The common free-model failure - one dropped field - is worth one retry."""
    requests: list[dict[str, Any]] = []
    replies = iter(['{"verdict": "buy"}', '{"verdict": "buy", "score": 0.7}'])

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": next(replies)}}]})

    client = OpenRouterLLMClient(settings(), "free/model:free", client=transport(handler))
    answer = await client.generate_structured(Answer, system="s", prompt="p")

    assert answer.score == 0.7
    assert len(requests) == 2
    # The retry replays the bad answer and names the field that was missing,
    # rather than just asking again and hoping.
    correction = requests[1]["messages"]
    assert correction[2] == {"role": "assistant", "content": '{"verdict": "buy"}'}
    assert "score" in correction[3]["content"]
    assert "Field required" in correction[3]["content"]


async def test_openrouter_gives_up_after_one_repair() -> None:
    """A model that cannot hold the schema twice is a recorded agent failure,
    not an unbounded retry loop on a rate-limited quota."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"verdict": "b"}'}}]})

    client = OpenRouterLLMClient(settings(), "free/model:free", client=transport(handler))
    with pytest.raises(LLMGenerationError, match="does not match Answer"):
        await client.generate_structured(Answer, system="s", prompt="p")
    assert len(requests) == 2


async def test_openrouter_rate_limit_explains_the_free_tier() -> None:
    client = OpenRouterLLMClient(
        settings(), "free/model:free", client=transport(openrouter_reply("slow down", status=429))
    )
    with pytest.raises(LLMGenerationError, match="rate-limited"):
        await client.generate_structured(Answer, system="s", prompt="p")


async def test_openrouter_error_body_is_surfaced() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"message": "model not found"}})

    client = OpenRouterLLMClient(settings(), "bad/model", client=transport(handler))
    with pytest.raises(LLMGenerationError, match="model not found"):
        await client.generate_structured(Answer, system="s", prompt="p")


async def test_openrouter_without_a_key_fails_with_guidance() -> None:
    client = OpenRouterLLMClient(settings(openrouter_api_key=None), "free/model:free")
    with pytest.raises(LLMGenerationError, match="ATP_LLM__OPENROUTER_API_KEY"):
        await client.generate_structured(Answer, system="s", prompt="p")


# --- Ollama client ----------------------------------------------------------


async def test_ollama_returns_a_validated_model() -> None:
    client = OllamaLLMClient(
        settings(),
        "llama3.1:8b",
        client=transport(ollama_reply('{"verdict": "hold", "score": 0.5}')),
    )
    answer = await client.generate_structured(Answer, system="s", prompt="p")
    assert answer.verdict == "hold"


async def test_ollama_constrains_decoding_with_the_schema() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"verdict": "b", "score": 1}'}})

    client = OllamaLLMClient(settings(), "qwen2.5:7b", client=transport(handler))
    await client.generate_structured(Answer, system="s", prompt="p")

    assert captured["model"] == "qwen2.5:7b"
    assert captured["stream"] is False
    # Ollama supports real structured output; the schema goes in `format`.
    assert captured["format"]["properties"].keys() == {"verdict", "score"}


async def test_a_missing_local_model_says_how_to_get_it() -> None:
    client = OllamaLLMClient(
        settings(), "llama3.1:8b", client=transport(ollama_reply("not found", status=404))
    )
    with pytest.raises(LLMGenerationError, match="ollama pull"):
        await client.generate_structured(Answer, system="s", prompt="p")


async def test_an_unreachable_ollama_says_so() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = OllamaLLMClient(settings(), "llama3.1:8b", client=transport(handler))
    with pytest.raises(LLMGenerationError, match="ollama serve"):
        await client.generate_structured(Answer, system="s", prompt="p")


# --- Catalogs ---------------------------------------------------------------


async def test_anthropic_catalog_is_static_and_paid() -> None:
    catalog = AnthropicModelCatalog()
    models = await catalog.list_models()

    assert models
    assert all(model.provider is LLMProvider.ANTHROPIC for model in models)
    assert not any(model.free for model in models)
    assert catalog.supports_pull is False


async def test_anthropic_catalog_cannot_pull() -> None:
    with pytest.raises(ModelUnavailableError, match="nothing to download"):
        await AnthropicModelCatalog().pull("claude-opus-4-8")


async def test_openrouter_catalog_flags_free_models_and_lists_them_first() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "paid/model", "name": "Zebra Paid", "context_length": 8000},
                    {"id": "free/model:free", "name": "Alpha Free", "context_length": 32000},
                ]
            },
        )

    models = await OpenRouterModelCatalog(settings(), client=transport(handler)).list_models()

    assert [model.id for model in models] == ["free/model:free", "paid/model"]
    assert models[0].free is True
    assert models[1].free is False
    assert models[0].context_length == 32000


async def test_openrouter_catalog_survives_an_unreachable_api() -> None:
    """One provider being down must not break a page listing the others."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    assert await OpenRouterModelCatalog(settings(), client=transport(handler)).list_models() == ()


async def test_ollama_catalog_lists_installed_then_suggestions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"models": [{"model": "llama3.1:8b", "size": 4_700_000_000}]}
        )

    models = await OllamaModelCatalog(settings(), client=transport(handler)).list_models()
    installed = [model for model in models if model.installed]
    suggested = [model for model in models if not model.installed]

    assert [model.id for model in installed] == ["llama3.1:8b"]
    assert installed[0].description == "4.7 GB"
    assert suggested  # fresh-machine suggestions
    assert "llama3.1:8b" not in {model.id for model in suggested}  # no duplicates
    assert all(model.free for model in models)


async def test_ollama_catalog_still_suggests_when_nothing_is_installed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("not running")

    models = await OllamaModelCatalog(settings(), client=transport(handler)).list_models()
    assert models
    assert all(model.installed is False for model in models)


async def test_ollama_pull_reports_success() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"status": "success"})

    catalog = OllamaModelCatalog(settings(), client=transport(handler))
    result = await catalog.pull("llama3.2:3b")

    assert catalog.supports_pull is True
    assert captured == {"model": "llama3.2:3b", "stream": False}
    assert result.installed is True
    assert result.model == "llama3.2:3b"


async def test_ollama_pull_failure_is_a_domain_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="disk full")

    with pytest.raises(ModelUnavailableError, match="could not pull"):
        await OllamaModelCatalog(settings(), client=transport(handler)).pull("llama3.2:3b")


async def test_pulling_an_empty_name_is_rejected() -> None:
    with pytest.raises(ModelUnavailableError, match="no model name"):
        await OllamaModelCatalog(settings()).pull("   ")


# --- Runtime router ---------------------------------------------------------


class RecordingClient:
    def __init__(self, label: str) -> None:
        self.label = label

    async def generate_structured(
        self, response_model: type[Answer], *, system: str, prompt: str
    ) -> Answer:
        return Answer(verdict=self.label, score=1.0)


async def test_router_delegates_to_the_active_backend() -> None:
    router = LLMRouter(
        lambda active: RecordingClient(str(active)),  # type: ignore[arg-type,return-value]
        ActiveModel(provider=LLMProvider.ANTHROPIC, model="claude-opus-4-8"),
    )

    answer = await router.generate_structured(Answer, system="s", prompt="p")
    assert answer.verdict == "anthropic:claude-opus-4-8"


async def test_selecting_a_model_changes_every_later_call() -> None:
    """Agents hold the router, so a switch reaches all of them at once."""
    router = LLMRouter(
        lambda active: RecordingClient(str(active)),  # type: ignore[arg-type,return-value]
        ActiveModel(provider=LLMProvider.ANTHROPIC, model="claude-opus-4-8"),
    )

    await router.select(LLMProvider.OLLAMA, "llama3.1:8b")

    assert router.active.provider is LLMProvider.OLLAMA
    answer = await router.generate_structured(Answer, system="s", prompt="p")
    assert answer.verdict == "ollama:llama3.1:8b"


async def test_selecting_an_empty_model_is_rejected() -> None:
    router = LLMRouter(
        lambda active: RecordingClient(str(active)),  # type: ignore[arg-type,return-value]
        ActiveModel(provider=LLMProvider.ANTHROPIC, model="claude-opus-4-8"),
    )
    with pytest.raises(ValueError, match="model id is required"):
        await router.select(LLMProvider.OLLAMA, "  ")


# --- Settings ---------------------------------------------------------------


def test_each_provider_keeps_its_own_model() -> None:
    """Switching back and forth must not lose the other provider's choice."""
    config = LLMSettings(
        model="claude-opus-4-8",
        openrouter_model="free/model:free",
        ollama_model="llama3.1:8b",
    )

    assert config.model_for(LLMProvider.ANTHROPIC) == "claude-opus-4-8"
    assert config.model_for(LLMProvider.OPENROUTER) == "free/model:free"
    assert config.model_for(LLMProvider.OLLAMA) == "llama3.1:8b"


def test_active_model_follows_the_configured_provider() -> None:
    config = LLMSettings(provider=LLMProvider.OLLAMA, ollama_model="qwen2.5:7b")
    assert config.active == ActiveModel(provider=LLMProvider.OLLAMA, model="qwen2.5:7b")


def test_api_keys_are_masked() -> None:
    config = LLMSettings(openrouter_api_key="sk-or-secret")
    assert "sk-or-secret" not in repr(config)
