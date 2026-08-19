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

Because the backend can move underneath a run, the router is also the only
place that knows, for certain, which model served any given call. It therefore
records each round-trip into the request's trace (see ``domain.llm_trace``) so
a caller can be told what actually answered rather than what is configured now.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TypeVar

import structlog
from pydantic import BaseModel

from atp.domain.llm_trace import current_agent, record_call
from atp.domain.models.llm import ActiveModel, LLMCall, LLMProvider
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
        # Read the delegate and the active model together, before awaiting: a
        # concurrent switch must not swap the backend out from under a call
        # already in flight, nor credit this call to the model that replaced it.
        delegate, active = self._delegate, self._active
        agent = current_agent()
        started_at = datetime.now(UTC)
        started = time.perf_counter()

        def record(status: str, detail: str = "") -> None:
            record_call(
                LLMCall(
                    provider=active.provider,
                    model=active.model,
                    agent=agent,
                    response_model=response_model.__name__,
                    started_at=started_at,
                    duration_ms=round((time.perf_counter() - started) * 1000, 1),
                    status=status,
                    detail=detail,
                )
            )

        try:
            result = await delegate.generate_structured(
                response_model, system=system, prompt=prompt
            )
        except Exception as exc:
            # A failed call is the one most worth attributing: "the free model
            # rate-limited us" and "the agent had no data" look identical from
            # the outside otherwise. Recorded, then re-raised untouched.
            record("failed", str(exc))
            raise
        record("ok")
        return result
