"""Append-only JSONL store for external bot signals.

Same shape as the order audit trail: a file, one JSON object per line, no
service required. Signal volume is low (one per bot cycle at most) and the
access pattern is "the newest for this symbol", so an index would be effort
spent on a problem this does not have.
"""

from __future__ import annotations

from pathlib import Path

from atp.domain.models.signals import BotSignal


class JsonSignalRepository:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def append(self, signal: BotSignal) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(signal.model_dump_json() + "\n")

    async def latest(self, symbol: str) -> BotSignal | None:
        symbol = symbol.strip().upper()
        # Reverse scan: the newest matching signal is almost always near the end.
        for signal in reversed(self._read()):
            if signal.symbol == symbol:
                return signal
        return None

    async def list_signals(self, *, limit: int | None = None) -> list[BotSignal]:
        signals = self._read()
        return signals[-limit:] if limit is not None else signals

    def _read(self) -> list[BotSignal]:
        if not self._path.exists():
            return []
        lines = [line for line in self._path.read_text("utf-8").splitlines() if line.strip()]
        return [BotSignal.model_validate_json(line) for line in lines]
