"""Which model answered, recorded per call rather than inferred after the fact.

The platform's whole promise is that a recommendation can be interrogated, and
"which model produced this" is part of that. Reporting the *configured* model
alongside a finished run is close enough to look right and wrong in exactly the
cases that matter: an operator switches model mid-run from the Models page, or
a provider is swapped between the analysis pool and the ranking call, and the
report then credits work to a backend that never saw it.

So the router records each round-trip as it happens, and the endpoint returns
the list. Two context variables carry it:

- ``llm_trace`` opens a recording scope for one request and yields the list the
  router appends to. Outside a scope, recording is a no-op — the router is used
  by the CLI and by tests too, and neither should have to opt out.
- ``attributed_to`` names the agent currently running, so a call can be blamed
  on ``news_analysis`` rather than on a schema name. The orchestrator sets it
  once per dispatch instead of every agent passing its own name down through
  the port, which would put a reporting concern into the one interface that
  exists to keep vendors out of the agents.

Context variables rather than parameters, for the same reason and with the same
guard rails as ``application.request_scope``: both scopes reset on exit, so one
request can never append to another's trace, and ``asyncio`` copies the active
context into each task, so the concurrently dispatched agents each keep their
own attribution.

This lives in ``domain`` because both sides need it — ``application`` sets the
agent name, ``infrastructure`` records the call — and domain is the only layer
they may both import.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from atp.domain.models.llm import LLMCall

_TRACE: ContextVar[list[LLMCall] | None] = ContextVar("atp_llm_trace", default=None)
_AGENT: ContextVar[str | None] = ContextVar("atp_llm_agent", default=None)


@contextmanager
def llm_trace() -> Iterator[list[LLMCall]]:
    """Record every LLM call made inside this block, in call order."""
    calls: list[LLMCall] = []
    token = _TRACE.set(calls)
    try:
        yield calls
    finally:
        _TRACE.reset(token)


@contextmanager
def attributed_to(agent: str) -> Iterator[None]:
    """Blame LLM calls made inside this block on ``agent``."""
    token = _AGENT.set(agent)
    try:
        yield
    finally:
        _AGENT.reset(token)


def current_agent() -> str | None:
    """The agent whose work is running, or None outside a dispatch."""
    return _AGENT.get()


def record_call(call: LLMCall) -> None:
    """Append to the active trace; a no-op when nothing is recording."""
    trace = _TRACE.get()
    if trace is None:
        return
    # Concurrent agents share one list. list.append is atomic under the GIL and
    # these tasks never yield inside it, so no lock is needed.
    trace.append(call)
