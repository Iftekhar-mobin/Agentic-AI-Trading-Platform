"""Unit tests for Yahoo vendor symbol translation."""

from __future__ import annotations

import pytest

from atp.infrastructure.vendor_symbols import is_translated, normalize, to_vendor_symbol


@pytest.mark.parametrize(
    ("typed", "vendor"),
    [
        # Spot metals have no Yahoo spot ticker; they resolve to the future.
        ("XAUUSD", "GC=F"),
        ("xauusd", "GC=F"),
        (" XauUsd ", "GC=F"),
        ("XAGUSD", "SI=F"),
        ("XPTUSD", "PL=F"),
        ("XPDUSD", "PA=F"),
        # Currency pairs take Yahoo's =X suffix.
        ("EURUSD", "EURUSD=X"),
        ("GBPJPY", "GBPJPY=X"),
        ("USDCHF", "USDCHF=X"),
        # Crypto takes -USD, whichever stablecoin was quoted against.
        ("BTCUSD", "BTC-USD"),
        ("BTCUSDT", "BTC-USD"),
        ("ETHUSD", "ETH-USD"),
    ],
)
def test_translates_known_conventions(typed: str, vendor: str) -> None:
    assert to_vendor_symbol(typed) == vendor
    assert is_translated(typed)


@pytest.mark.parametrize(
    "symbol",
    [
        "AAPL",  # ordinary equity
        "GOOGL",
        "MSFT",
        "GC=F",  # already vendor-shaped
        "BTC-USD",
        "^GSPC",
        "BRK.B",
        "ZZZZZZ",  # six letters, but neither half is a currency
        "XAUEUR",  # metal, but not quoted in USD
        "USDUSD",  # degenerate pair
    ],
)
def test_passes_through_what_it_does_not_recognize(symbol: str) -> None:
    assert to_vendor_symbol(symbol) == normalize(symbol)
    assert not is_translated(symbol)


def test_translation_is_idempotent() -> None:
    """Vendor form must survive a second pass, or a retry would corrupt it."""
    for typed in ("XAUUSD", "EURUSD", "BTCUSD", "AAPL"):
        once = to_vendor_symbol(typed)
        assert to_vendor_symbol(once) == once
