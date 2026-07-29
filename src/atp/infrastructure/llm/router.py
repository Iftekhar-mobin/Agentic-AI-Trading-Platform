"""Runtime-switchable LLM client.

Implements ``LLMClient`` by delegating to whichever backend is currently
selected. Every agent holds a reference to this one object, so changing the
model takes effect on the next call — no restart, no rewiring, no agent aware
that anything happened.

The delegate is rebuilt on selection rather than held per provider. Building a
client is cheap (an httpx client and some config), and rebuilding means the
model id lives in exactly one place instead of being threaded through every
call signature.

Selection is guarded by a lock so a switch mid-workflow cannot leave two agents
reading a half-updated pair of provider and model — they would otherwise be
able to observe an Ollama model id pointed at the Anthropic client.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

import structlog
from pydantic import BaseModel

from atp.domain.models.llm import ActiveModel, LLMProvider
from atp.domain.ports.llm import LLMClient

log = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)

ClientFactory = Callable[[ActiveModel], LLMClient]


class LLMRouter:
    """An ``LLMClient`` whose backend can be swapped at runtime."""

    def __init__(self, factory: ClientFactory, initial: ActiveModel) -> None:
        self._factory = factory
        self._active = initial
        self._delegate = factory(initial)
        self._lock = asyncio.Lock()

    @property
    def active(self) -> ActiveModel:
        return self._active

    async def select(self, provider: LLMProvider, model: str) -> ActiveModel:
        """Point every agent at a different model."""
        model = model.strip()
        if not model:
            msg = "a model id is required"
            raise ValueError(msg)

        requested = ActiveModel(provider=provider, model=model)
        async with self._lock:
            previous = self._active
            self._delegate = self._factory(requested)
            self._active = requested
        log.info(
            "llm.model_selected",
            provider=provider.value,
            model=model,
            previous=str(previous),
        )
        return requested

    async def generate_structured(
        self,
        response_model: type[T],
        *,
        system: str,
        prompt: str,
    ) -> T:
        # Read the delegate once: a concurrent switch must not swap the backend
        # out from under a call already in flight.
        delegate = self._delegate
        return await delegate.generate_structured(response_model, system=system, prompt=prompt)
