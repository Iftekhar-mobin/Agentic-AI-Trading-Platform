"""Aggregation of bars into higher, synthetic timeframes.

Four-hour candles are a standard swing-trading timeframe that most vendors —
yfinance included — simply do not serve. Rather than drop 4H from the platform
or teach every adapter to fake it, bars are aggregated here, once, in the
domain, and a provider decorator applies it to whatever source is configured.

Buckets are aligned to fixed UTC boundaries rather than to the first bar in the
series. That is what makes the output stable: the same 4H candle has the same
timestamp and the same contents no matter which day you started the fetch from.
It also means the buckets do not align to any particular exchange's session —
an honest limitation, and the reason a partial trailing bucket is still
emitted (the current, incomplete candle is what a trader is looking at).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from atp.domain.models.market import Bar, BarInterval, PriceHistory

SYNTHETIC_INTERVALS: dict[BarInterval, BarInterval] = {
    BarInterval.HOUR_4: BarInterval.HOUR_1,
}
"""Target interval -> the source interval it is aggregated from."""

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def source_interval(interval: BarInterval) -> BarInterval | None:
    """The interval to fetch when ``interval`` is synthetic, else ``None``."""
    return SYNTHETIC_INTERVALS.get(interval)


def _bucket_start(timestamp: datetime, minutes: int) -> datetime:
    elapsed = timestamp - _EPOCH
    return _EPOCH + timedelta(minutes=(elapsed // timedelta(minutes=minutes)) * minutes)


def resample(history: PriceHistory, target: BarInterval) -> PriceHistory:
    """Aggregate ``history`` into ``target`` bars.

    Raises ``ValueError`` when the target is not a coarser multiple of the
    source — silently returning the wrong timeframe would be far worse than
    failing.
    """
    if target.minutes <= history.interval.minutes:
        msg = (
            f"cannot resample {history.interval.value} bars into {target.value}: "
            "the target must be a higher timeframe"
        )
        raise ValueError(msg)
    if target.minutes % history.interval.minutes != 0:
        msg = (
            f"cannot resample {history.interval.value} into {target.value}: "
            f"{target.minutes} is not a multiple of {history.interval.minutes}"
        )
        raise ValueError(msg)

    buckets: dict[datetime, list[Bar]] = {}
    for bar in history.bars:
        buckets.setdefault(_bucket_start(bar.timestamp, target.minutes), []).append(bar)

    bars = tuple(
        Bar(
            timestamp=start,
            open=group[0].open,
            high=max(bar.high for bar in group),
            low=min(bar.low for bar in group),
            close=group[-1].close,
            volume=sum(bar.volume for bar in group),
        )
        # `history.bars` is already ascending, so each group is too.
        for start, group in sorted(buckets.items())
    )
    return PriceHistory(symbol=history.symbol, interval=target, bars=bars)
