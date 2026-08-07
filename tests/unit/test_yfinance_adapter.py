"""Unit tests for the yfinance adapter (network calls faked)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest
import yfinance

from atp.domain.models.market import BarInterval
from atp.infrastructure.market_data import YFinanceMarketDataProvider


def yahoo_frame(*, multiindex: bool = True) -> pd.DataFrame:
    """A frame mimicking yfinance.download output for a single ticker."""
    index = pd.DatetimeIndex(
        [datetime(2026, 1, 5), datetime(2026, 1, 6), datetime(2026, 1, 7), datetime(2026, 1, 8)]
    )
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0],
            "High": [105.0, 106.0, 90.0, float("nan")],  # day 3 inconsistent, day 4 NaN
            "Low": [99.0, 100.0, 101.0, 102.0],
            "Close": [103.0, 104.0, 105.0, 106.0],
            "Volume": [1_000.0, 2_000.0, 3_000.0, 4_000.0],
        },
        index=index,
    )
    if multiindex:
        frame.columns = pd.MultiIndex.from_product([frame.columns, ["AAPL"]])
    return frame


@pytest.fixture
def fake_download(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def download(**kwargs: Any) -> pd.DataFrame:
        calls.update(kwargs)
        return yahoo_frame()

    monkeypatch.setattr(yfinance, "download", download)
    return calls


async def test_normalizes_vendor_frame_to_domain_bars(fake_download: dict[str, Any]) -> None:
    provider = YFinanceMarketDataProvider()
    history = await provider.get_bars(
        " aapl ", BarInterval.DAY_1, start=datetime(2026, 1, 1, tzinfo=UTC)
    )

    assert history.symbol == "AAPL"
    assert fake_download["tickers"] == "AAPL"
    assert fake_download["interval"] == "1d"
    # 4 vendor rows: one NaN row dropped, one OHLC-inconsistent row dropped.
    assert len(history) == 2
    assert history.bars[0].close == 103.0
    assert all(bar.timestamp.tzinfo == UTC for bar in history.bars)


async def test_downloads_vendor_ticker_but_reports_the_symbol_asked_for(
    fake_download: dict[str, Any],
) -> None:
    """Gold is fetched as GC=F; the history still says XAUUSD, because that is
    what the caller asked about and what every downstream report will show."""
    provider = YFinanceMarketDataProvider()
    history = await provider.get_bars(
        "xauusd", BarInterval.DAY_1, start=datetime(2026, 1, 1, tzinfo=UTC)
    )

    assert fake_download["tickers"] == "GC=F"
    assert history.symbol == "XAUUSD"


async def test_handles_flat_columns_and_empty_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = YFinanceMarketDataProvider()

    monkeypatch.setattr(yfinance, "download", lambda **_: yahoo_frame(multiindex=False))
    history = await provider.get_bars(
        "AAPL", BarInterval.DAY_1, start=datetime(2026, 1, 1, tzinfo=UTC)
    )
    assert len(history) == 2

    monkeypatch.setattr(yfinance, "download", lambda **_: pd.DataFrame())
    empty = await provider.get_bars(
        "AAPL", BarInterval.DAY_1, start=datetime(2026, 1, 1, tzinfo=UTC)
    )
    assert len(empty) == 0
