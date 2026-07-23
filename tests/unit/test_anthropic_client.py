"""Tests for the Anthropic LLM adapter, with the SDK client stubbed."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import anthropic
import pytest
from pydantic import BaseModel

from atp.domain.errors import LLMGenerationError
from atp.infrastructure.config import LLMSettings
from atp.infrastructure.llm import AnthropicLLMClient


class Verdict(BaseModel):
    direction: str
    confidence: float


class StubSDK:
    """Mimics AsyncAnthropic().messages.parse."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.kwargs: dict[str, Any] = {}
        self.messages = SimpleNamespace(parse=self._parse)

    async def _parse(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self._error is not None:
            raise self._error
        return self._response


def make_client(stub: StubSDK) -> AnthropicLLMClient:
    return AnthropicLLMClient(LLMSettings(), client=cast(anthropic.AsyncAnthropic, stub))


def make_response(*, parsed: Verdict | None, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
    )


async def test_returns_parsed_model_and_sends_expected_request() -> None:
    verdict = Verdict(direction="bullish", confidence=0.7)
    stub = StubSDK(response=make_response(parsed=verdict))

    result = await make_client(stub).generate_structured(Verdict, system="sys", prompt="prompt")

    assert result is verdict
    assert stub.kwargs["model"] == "claude-opus-4-8"
    assert stub.kwargs["system"] == "sys"
    assert stub.kwargs["thinking"] == {"type": "adaptive"}
    assert stub.kwargs["output_format"] is Verdict
    assert stub.kwargs["messages"] == [{"role": "user", "content": "prompt"}]


async def test_refusal_raises_domain_error() -> None:
    stub = StubSDK(response=make_response(parsed=None, stop_reason="refusal"))
    with pytest.raises(LLMGenerationError, match="refusal"):
        await make_client(stub).generate_structured(Verdict, system="s", prompt="p")


async def test_missing_parsed_output_raises_domain_error() -> None:
    stub = StubSDK(response=make_response(parsed=None, stop_reason="max_tokens"))
    with pytest.raises(LLMGenerationError, match="max_tokens"):
        await make_client(stub).generate_structured(Verdict, system="s", prompt="p")


async def test_api_errors_are_wrapped() -> None:
    error = anthropic.APIConnectionError(request=cast(Any, SimpleNamespace()))
    stub = StubSDK(error=error)
    with pytest.raises(LLMGenerationError, match="request failed"):
        await make_client(stub).generate_structured(Verdict, system="s", prompt="p")


async def test_missing_credentials_give_actionable_error() -> None:
    # The SDK raises TypeError at request time when no auth method resolves.
    stub = StubSDK(error=TypeError("Could not resolve authentication method."))
    with pytest.raises(LLMGenerationError, match="ANTHROPIC_API_KEY"):
        await make_client(stub).generate_structured(Verdict, system="s", prompt="p")
