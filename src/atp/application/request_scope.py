"""Request-scoped inputs supplied by the caller rather than fetched by us.

An external trading bot that has *already* pulled candles from its broker can
send them with a consensus request, along with a description of the setup it is
looking at. Both are properties of one request, and both are needed several
layers below the endpoint — in ``LoadPriceHistory`` and inside the agents — so
they live here as context variables instead of being threaded as parameters
through the orchestrator, the graph state, both use cases and both agent
signatures.

That is a deliberate trade. Passing them explicitly would be purer, and the
cost would be a new parameter on roughly a dozen call sites that have no
interest in the value. Context variables keep the change surgical, and they
propagate correctly into the concurrent agent tasks because ``asyncio`` copies
the active context when a task is created.

Two rules make the side channel safe:

- **Scope it.** Always enter through ``supplied_inputs``; it resets on exit, so
  one request can never leak candles into the next.
- **Never guess.** An absent value means "fetch normally", not "empty". Every
  reader distinguishes the two.

Why a caller supplying bars is the *better* path, not a shortcut: bars fetched
here would come from a different vendor, minutes later, and — for spot metals —
from a different instrument entirely (``XAUUSD`` resolves to ``GC=F``,
front-month gold). When the question is "do you agree with the decision I just
made", the answer has to be computed from the candles that decision was made
on.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from atp.domain.models.market import BarInterval, PriceHistory

_SUPPLIED_BARS: ContextVar[Mapping[tuple[str, BarInterval], PriceHistory] | None] = ContextVar(
    "atp_supplied_bars", default=None
)
_BOT_CONTEXT: ContextVar[Mapping[str, Any] | None] = ContextVar("atp_bot_context", default=None)


@contextmanager
def supplied_inputs(
    *,
    bars: Mapping[tuple[str, BarInterval], PriceHistory] | None = None,
    context: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Make caller-supplied bars and setup context visible for one request."""
    bars_token = _SUPPLIED_BARS.set(bars or None)
    context_token = _BOT_CONTEXT.set(context or None)
    try:
        yield
    finally:
        _SUPPLIED_BARS.reset(bars_token)
        _BOT_CONTEXT.reset(context_token)


def supplied_history(symbol: str, interval: BarInterval) -> PriceHistory | None:
    """Bars the caller sent for this symbol/timeframe, or None to fetch normally."""
    supplied = _SUPPLIED_BARS.get()
    if not supplied:
        return None
    return supplied.get((symbol.strip().upper(), interval))


def bot_context() -> Mapping[str, Any] | None:
    """The caller's description of the setup, or None when it sent none.

    This is an external system's reading of a lower timeframe. It is evidence,
    not instruction: agents may weigh it, and must not defer to it.
    """
    return _BOT_CONTEXT.get()
