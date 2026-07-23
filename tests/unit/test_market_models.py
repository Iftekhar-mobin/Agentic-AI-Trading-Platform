"""Unit tests for market data domain models."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from atp.domain.models.market import Bar, BarInterval, PriceHistory


def make_bar(ts: datetime | None = None, **overrides: float) -> Bar:
    values: dict[str, float] = {
        "open": 100.0,
        "high": 105.0,
        "low": 99.0,
        "close": 103.0,
        "volume": 1_000.0,
    }
    values.update(overrides)
    return Bar(timestamp=ts or datetime(2026, 1, 5, tzinfo=UTC), **values)


class TestBar:
    def test_timestamps_are_normalized_to_utc(self) -> None:
        eastern = timezone(timedelta(hours=-5))
        bar = make_bar(ts=datetime(2026, 1, 5, 9, 30, tzinfo=eastern))
        assert bar.timestamp.tzinfo == UTC
        assert bar.timestamp.hour == 14

    def test_naive_timestamps_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            make_bar(ts=datetime(2026, 1, 5))

    def test_high_below_open_or_close_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="inconsistent OHLC"):
            make_bar(high=101.0, close=103.0)

    def test_low_above_open_or_close_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="inconsistent OHLC"):
            make_bar(low=101.0, open=100.0)

    def test_negative_volume_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_bar(volume=-1.0)

    def test_zero_or_negative_prices_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_bar(open=0.0)

    def test_bars_are_immutable(self) -> None:
        bar = make_bar()
        with pytest.raises(ValidationError):
            bar.close = 999.0  # type: ignore[misc]


class TestPriceHistory:
    def test_symbol_is_normalized(self) -> None:
        history = PriceHistory(symbol=" aapl ", interval=BarInterval.DAY_1)
        assert history.symbol == "AAPL"

    def test_unsorted_bars_are_rejected(self) -> None:
        day1 = make_bar(ts=datetime(2026, 1, 5, tzinfo=UTC))
        day2 = make_bar(ts=datetime(2026, 1, 6, tzinfo=UTC))
        with pytest.raises(ValidationError, match="ascending"):
            PriceHistory(symbol="AAPL", interval=BarInterval.DAY_1, bars=(day2, day1))

    def test_duplicate_timestamps_are_rejected(self) -> None:
        bar = make_bar()
        with pytest.raises(ValidationError, match="ascending"):
            PriceHistory(symbol="AAPL", interval=BarInterval.DAY_1, bars=(bar, bar))

    def test_len_and_latest(self) -> None:
        day1 = make_bar(ts=datetime(2026, 1, 5, tzinfo=UTC))
        day2 = make_bar(ts=datetime(2026, 1, 6, tzinfo=UTC))
        history = PriceHistory(symbol="AAPL", interval=BarInterval.DAY_1, bars=(day1, day2))
        assert len(history) == 2
        assert history.latest == day2
        assert PriceHistory(symbol="AAPL", interval=BarInterval.DAY_1).latest is None
