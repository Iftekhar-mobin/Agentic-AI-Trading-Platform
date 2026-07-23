"""File-backed portfolio store for paper trading.

A single JSON file with atomic writes — deliberately boring. The Postgres
repository replaces this behind the same port when execution arrives and
positions need transactional updates alongside orders.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import structlog

from atp.domain.models.trading import Portfolio

log = structlog.get_logger()


class JsonPortfolioRepository:
    def __init__(self, path: Path, *, starting_cash: Decimal) -> None:
        self._path = path
        self._starting_cash = starting_cash

    async def load(self) -> Portfolio:
        if not self._path.exists():
            log.info(
                "portfolio.seeded",
                path=str(self._path),
                starting_cash=float(self._starting_cash),
            )
            return Portfolio(cash=self._starting_cash)
        return Portfolio.model_validate_json(self._path.read_text("utf-8"))

    async def save(self, portfolio: Portfolio) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp = self._path.with_suffix(".json.tmp")
        temp.write_text(portfolio.model_dump_json(indent=2), "utf-8")
        temp.replace(self._path)  # atomic on the same filesystem
