"""Port for the deterministic technical indicator engine."""

from __future__ import annotations

from typing import Protocol

from atp.domain.models.analysis import IndicatorReading
from atp.domain.models.market import PriceHistory


class IndicatorEngine(Protocol):
    """Computes indicator readings from price history.

    Implementations must be pure/deterministic: same history in, same readings
    out. Raises ``InsufficientHistoryError`` below ``MIN_HISTORY_BARS``.
    """

    def compute_readings(self, history: PriceHistory) -> tuple[IndicatorReading, ...]: ...
