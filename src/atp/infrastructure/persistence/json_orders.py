"""Append-only JSONL order audit trail."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from atp.domain.models.orders import Order, OrderRecord
from atp.domain.models.trading import RiskDecision


class JsonOrderRepository:
    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._path = path
        self._clock = clock

    async def append(self, order: Order, decision: RiskDecision) -> None:
        record = OrderRecord(order=order, decision=decision, recorded_at=self._clock())
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    async def list_records(self, *, limit: int | None = None) -> list[OrderRecord]:
        if not self._path.exists():
            return []
        lines = [line for line in self._path.read_text("utf-8").splitlines() if line.strip()]
        records = [OrderRecord.model_validate_json(line) for line in lines]
        return records[-limit:] if limit is not None else records
