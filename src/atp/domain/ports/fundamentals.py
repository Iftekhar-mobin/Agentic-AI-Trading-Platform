"""Port for company fundamentals sources."""

from __future__ import annotations

from typing import Protocol

from atp.domain.models.fundamentals import Fundamentals


class FundamentalsProvider(Protocol):
    """Fetches a fundamentals snapshot for one symbol.

    Implementations must normalize to the canonical units documented on
    ``Fundamentals`` and leave a metric ``None`` when the vendor does not
    supply it — never zero, never imputed.
    """

    async def get_fundamentals(self, symbol: str) -> Fundamentals: ...
