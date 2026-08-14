"""Tests for the selectable universe.

Mostly guarding the seam between the curated catalog and the vendor: a symbol
that does not translate returns an empty price frame, and every agent then
reports "insufficient data" — a failure that looks like a broken platform
rather than a bad menu entry.
"""

from __future__ import annotations

import pytest

from atp.domain.models.universe import AssetClass
from atp.domain.universe import (
    CATALOG,
    DEFAULT_SELECTION,
    FOREX_MAJORS,
    by_class,
    groups,
    lookup,
)
from atp.infrastructure.vendor_symbols import to_vendor_symbol

MAJORS = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"}


def test_every_forex_major_is_offered() -> None:
    """ "The majors" is a closed list; a missing one is immediately noticed."""
    assert {asset.symbol for asset in FOREX_MAJORS} == MAJORS


def test_symbols_are_unique() -> None:
    symbols = [asset.symbol for asset in CATALOG]
    assert len(symbols) == len(set(symbols))


def test_every_class_is_represented() -> None:
    for asset_class in AssetClass:
        assert by_class(asset_class), f"nothing offered for {asset_class}"


VENDOR_SHAPE = {
    AssetClass.FOREX: "=X",
    AssetClass.CRYPTO: "-USD",
    AssetClass.COMMODITY: "=F",
}
"""The suffix each non-equity class must carry by the time it reaches Yahoo.
Equities and indices are already written the vendor's way."""


@pytest.mark.parametrize("asset", CATALOG, ids=lambda asset: asset.symbol)
def test_every_catalog_symbol_resolves_to_a_real_vendor_ticker(asset: object) -> None:
    """The silent failure this guards: a pair whose base is missing from
    ``ISO_CURRENCIES`` passes straight through as ``EURPLN``, fetches an empty
    frame, and every agent reports "insufficient data" as if the platform were
    broken."""
    symbol: str = asset.symbol  # type: ignore[attr-defined]
    asset_class: AssetClass = asset.asset_class  # type: ignore[attr-defined]
    vendor = to_vendor_symbol(symbol)

    if suffix := VENDOR_SHAPE.get(asset_class):
        assert vendor.endswith(suffix), f"{symbol} resolves to {vendor}, which is not a {suffix}"
    elif asset_class is AssetClass.INDEX:
        assert vendor.startswith("^")
    else:
        assert vendor == symbol


def test_the_default_selection_spans_several_classes() -> None:
    """A first screen has to have something to compare."""
    classes = {lookup(symbol).asset_class for symbol in DEFAULT_SELECTION}
    assert len(classes) >= 4


def test_the_default_selection_is_all_catalog_symbols() -> None:
    listed = {asset.symbol for asset in CATALOG}
    assert set(DEFAULT_SELECTION) <= listed


def test_groups_preserve_catalog_order() -> None:
    forex = groups(AssetClass.FOREX)
    assert list(forex) == ["Majors", "Popular crosses"]


def test_an_unlisted_symbol_is_still_usable() -> None:
    """The catalog is a menu, not a whitelist."""
    asset = lookup("shel")
    assert asset.symbol == "SHEL"
    assert asset.group == "Custom"


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("^HSI", AssetClass.INDEX),
        ("ZC=F", AssetClass.COMMODITY),
        ("DOGE-USD", AssetClass.CRYPTO),
        ("EURSEK", AssetClass.FOREX),
        ("SHEL", AssetClass.STOCK),
    ],
)
def test_unlisted_symbols_are_classified_by_shape(symbol: str, expected: AssetClass) -> None:
    assert lookup(symbol).asset_class is expected
