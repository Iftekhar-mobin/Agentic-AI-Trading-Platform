"""Tests for synthetic-timeframe aggregation (4H from 1H) and the provider decorator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from atp.domain.models.market import Bar, BarInterval, PriceHistory, sort_timeframes
from atp.domain.services.resample import resample, source_interval
from atp.infrastructure.market_data import ResamplingMarketDataProvider

START = datetime(2026, 1, 5, 0, 0, tzinfo=UTC)


def hourly(count: int, *, start: datetime = START) -> PriceHistory:
    """Prices are a function of absolute time, not of position in the series.

    That is what lets a test assert two different fetch windows produce
    byte-identical candles for the hours they share.
    """
    offset = int((start - START).total_seconds() // 3600)
    bars = tuple(
        Bar(
            timestamp=start + timedelta(hours=index),
            open=100.0 + offset + index,
            high=100.5 + offset + index,
            low=99.5 + offset + index,
            close=100.2 + offset + index,
            volume=10.0,
        )
        for index in range(count)
    )
    return PriceHistory(symbol="AAPL", interval=BarInterval.HOUR_1, bars=bars)


class SpyProvider:
    def __init__(self, history: PriceHistory) -> None:
        self._history = history
        self.requests: list[tuple[BarInterval, datetime]] = []

    async def get_bars(
        self,
        symbol: str,
        interval: BarInterval,
        *,
        start: datetime,
        end: datetime | None = None,
    ) -> PriceHistory:
        self.requests.append((interval, start))
        return self._history


# --- Timeframe ordering -----------------------------------------------------


def test_timeframes_sort_highest_first_and_deduplicate() -> None:
    ordered = sort_timeframes(
        [BarInterval.HOUR_1, BarInterval.DAY_1, BarInterval.HOUR_4, BarInterval.DAY_1]
    )
    assert ordered == (BarInterval.DAY_1, BarInterval.HOUR_4, BarInterval.HOUR_1)


def test_four_hour_sits_between_one_hour_and_one_day() -> None:
    assert BarInterval.HOUR_1.minutes < BarInterval.HOUR_4.minutes < BarInterval.DAY_1.minutes


# --- Aggregation ------------------------------------------------------------


def test_four_hour_is_built_from_one_hour() -> None:
    assert source_interval(BarInterval.HOUR_4) is BarInterval.HOUR_1
    assert source_interval(BarInterval.DAY_1) is None


def test_aggregation_produces_correct_ohlcv() -> None:
    result = resample(hourly(4), BarInterval.HOUR_4)

    assert len(result) == 1
    bar = result.bars[0]
    assert bar.timestamp == START
    assert bar.open == 100.0  # first bar's open
    assert bar.close == pytest.approx(103.2)  # last bar's close
    assert bar.high == pytest.approx(103.5)  # highest high
    assert bar.low == pytest.approx(99.5)  # lowest low
    assert bar.volume == 40.0  # summed


def test_buckets_align_to_fixed_utc_boundaries() -> None:
    """Starting mid-bucket must not shift every candle that follows."""
    result = resample(hourly(8, start=START + timedelta(hours=2)), BarInterval.HOUR_4)

    assert [bar.timestamp for bar in result.bars] == [
        START,
        START + timedelta(hours=4),
        START + timedelta(hours=8),
    ]


def test_bucket_contents_do_not_depend_on_the_fetch_start() -> None:
    full = resample(hourly(12), BarInterval.HOUR_4)
    later = resample(hourly(8, start=START + timedelta(hours=4)), BarInterval.HOUR_4)

    # The 04:00 and 08:00 candles are identical in both fetches.
    assert full.bars[1] == later.bars[0]
    assert full.bars[2] == later.bars[1]


def test_a_partial_trailing_bucket_is_still_emitted() -> None:
    """The candle in progress is what a trader is looking at."""
    result = resample(hourly(6), BarInterval.HOUR_4)

    assert len(result) == 2
    assert result.bars[-1].volume == 20.0  # only two of four hours so far


def test_the_result_carries_the_target_interval() -> None:
    assert resample(hourly(8), BarInterval.HOUR_4).interval is BarInterval.HOUR_4


def test_resampling_downward_is_rejected() -> None:
    daily = PriceHistory(
        symbol="AAPL",
        interval=BarInterval.DAY_1,
        bars=hourly(3).bars,
    )
    with pytest.raises(ValueError, match="must be a higher timeframe"):
        resample(daily, BarInterval.HOUR_4)


def test_every_supported_interval_pair_divides_cleanly() -> None:
    """Documents why the "not a multiple" guard never fires today.

    Every interval in the enum is a whole multiple of every smaller one, so
    aggregation is always exact. The guard exists for the first interval that
    breaks that (a 90-minute bar, say) and would otherwise silently produce
    ragged candles.
    """
    intervals = sorted(BarInterval, key=lambda interval: interval.minutes)
    for index, finer in enumerate(intervals):
        for coarser in intervals[index + 1 :]:
            assert coarser.minutes % finer.minutes == 0, f"{coarser} % {finer}"


def test_empty_history_aggregates_to_empty() -> None:
    empty = PriceHistory(symbol="AAPL", interval=BarInterval.HOUR_1)
    assert len(resample(empty, BarInterval.HOUR_4)) == 0


# --- Provider decorator -----------------------------------------------------


async def test_native_intervals_pass_straight_through() -> None:
    inner = SpyProvider(hourly(4))
    provider = ResamplingMarketDataProvider(inner)

    result = await provider.get_bars("AAPL", BarInterval.HOUR_1, start=START)

    assert inner.requests == [(BarInterval.HOUR_1, START)]
    assert result.interval is BarInterval.HOUR_1


async def test_four_hour_requests_fetch_one_hour_and_aggregate() -> None:
    inner = SpyProvider(hourly(8))
    provider = ResamplingMarketDataProvider(inner)

    result = await provider.get_bars("AAPL", BarInterval.HOUR_4, start=START)

    requested_interval, requested_start = inner.requests[0]
    assert requested_interval is BarInterval.HOUR_1
    # Widened by one target bucket so the first candle is complete.
    assert requested_start == START - timedelta(hours=4)
    assert result.interval is BarInterval.HOUR_4
    assert len(result) == 2
