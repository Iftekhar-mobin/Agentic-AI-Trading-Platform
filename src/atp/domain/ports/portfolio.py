"""Port for portfolio persistence."""

from __future__ import annotations

from typing import Protocol

from atp.domain.models.trading import Portfolio


class PortfolioRepository(Protocol):
    """Loads and saves the (single, for now) paper portfolio.

    ``load`` must return a valid starting portfolio when none exists yet.
    """

    async def load(self) -> Portfolio: ...

    async def save(self, portfolio: Portfolio) -> None: ...
