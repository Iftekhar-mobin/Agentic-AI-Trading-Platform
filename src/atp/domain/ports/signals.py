"""Port for storing signals published by external bots."""

from __future__ import annotations

from typing import Protocol

from atp.domain.models.signals import BotSignal


class SignalRepository(Protocol):
    """Append-only record of what external bots have claimed.

    Append-only on purpose: a bot changing its mind is a new signal, not an
    edit. When a consensus is later questioned, the sequence of claims that
    produced it has to still be there.
    """

    async def append(self, signal: BotSignal) -> None: ...

    async def latest(self, symbol: str) -> BotSignal | None:
        """The most recent signal for a symbol, regardless of age.

        Freshness is the caller's decision, not the store's — the voting layer
        knows the TTL and needs the option to report "a signal existed but was
        stale", which a store that filtered silently could never express.
        """
        ...

    async def list_signals(self, *, limit: int | None = None) -> list[BotSignal]: ...
