"""The curated selectable universe: what the console offers out of the box.

Deliberately a short, opinionated list rather than every instrument a vendor
serves. A screen costs one LLM call per agent per symbol, so an interface that
lets you tick five hundred boxes is an interface that lets you spend a fortune
by accident. These are the instruments a discretionary trader actually watches:
the currency majors, the large-cap US names that set the tone, the headline
indices, the metals and energy contracts, and the two or three cryptocurrencies
with enough history to analyse.

Every symbol here is written the way a trader writes it - ``XAUUSD``, not
``GC=F``. ``infrastructure.vendor_symbols`` translates on the way out to Yahoo,
and reports keep the name that was asked for.

Nothing stops a caller analysing a symbol that is absent: this is a menu, not a
whitelist. ``lookup`` returns a synthesized ``Asset`` for anything unknown so an
off-menu ticker still ranks and renders like the rest.
"""

from __future__ import annotations

from typing import Final

from atp.domain.models.universe import Asset, AssetClass


def _assets(asset_class: AssetClass, group: str, entries: dict[str, str]) -> tuple[Asset, ...]:
    return tuple(
        Asset(symbol=symbol, name=name, asset_class=asset_class, group=group)
        for symbol, name in entries.items()
    )


FOREX_MAJORS: Final[tuple[Asset, ...]] = _assets(
    AssetClass.FOREX,
    "Majors",
    {
        "EURUSD": "Euro / US Dollar",
        "GBPUSD": "British Pound / US Dollar",
        "USDJPY": "US Dollar / Japanese Yen",
        "USDCHF": "US Dollar / Swiss Franc",
        "USDCAD": "US Dollar / Canadian Dollar",
        "AUDUSD": "Australian Dollar / US Dollar",
        "NZDUSD": "New Zealand Dollar / US Dollar",
    },
)
"""The seven majors - every pair with the dollar on one side. This is the whole
set, not a selection: "the majors" is a closed list and a trader will notice a
missing one immediately."""

FOREX_CROSSES: Final[tuple[Asset, ...]] = _assets(
    AssetClass.FOREX,
    "Popular crosses",
    {
        "EURGBP": "Euro / British Pound",
        "EURJPY": "Euro / Japanese Yen",
        "GBPJPY": "British Pound / Japanese Yen",
        "AUDJPY": "Australian Dollar / Japanese Yen",
        "EURAUD": "Euro / Australian Dollar",
    },
)

STOCKS: Final[tuple[Asset, ...]] = _assets(
    AssetClass.STOCK,
    "US large caps",
    {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "NVDA": "NVIDIA",
        "AMZN": "Amazon",
        "GOOGL": "Alphabet",
        "META": "Meta Platforms",
        "TSLA": "Tesla",
        "AMD": "Advanced Micro Devices",
        "NFLX": "Netflix",
        "JPM": "JPMorgan Chase",
        "V": "Visa",
        "XOM": "Exxon Mobil",
        "BRK-B": "Berkshire Hathaway B",
    },
)

INDICES: Final[tuple[Asset, ...]] = _assets(
    AssetClass.INDEX,
    "Headline indices",
    {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones Industrial Average",
        "^IXIC": "Nasdaq Composite",
        "^NDX": "Nasdaq 100",
        "^RUT": "Russell 2000",
        "^VIX": "CBOE Volatility Index",
        "^FTSE": "FTSE 100",
        "^GDAXI": "DAX 40",
        "^N225": "Nikkei 225",
    },
)
"""Indices carry no fundamentals and little company news, so those agents will
usually abstain on them. That is reported as a smaller voter pool rather than
hidden - see ``coverage`` in the ranking components."""

METALS: Final[tuple[Asset, ...]] = _assets(
    AssetClass.COMMODITY,
    "Precious metals",
    {
        "XAUUSD": "Gold spot",
        "XAGUSD": "Silver spot",
        "XPTUSD": "Platinum spot",
    },
)
"""Spot metals resolve to the front-month future at the vendor boundary; the
basis is immaterial for structure and trend. See ``vendor_symbols``."""

ENERGY_AND_INDUSTRIAL: Final[tuple[Asset, ...]] = _assets(
    AssetClass.COMMODITY,
    "Energy & industrial",
    {
        "CL=F": "WTI Crude Oil",
        "BZ=F": "Brent Crude Oil",
        "NG=F": "Natural Gas",
        "HG=F": "Copper",
    },
)

CRYPTO: Final[tuple[Asset, ...]] = _assets(
    AssetClass.CRYPTO,
    "Major coins",
    {
        "BTCUSD": "Bitcoin",
        "ETHUSD": "Ethereum",
        "SOLUSD": "Solana",
        "XRPUSD": "XRP",
    },
)

CATALOG: Final[tuple[Asset, ...]] = (
    *FOREX_MAJORS,
    *FOREX_CROSSES,
    *INDICES,
    *STOCKS,
    *METALS,
    *ENERGY_AND_INDUSTRIAL,
    *CRYPTO,
)
"""Every offered instrument, in the order an interface should present it."""

_BY_SYMBOL: Final[dict[str, Asset]] = {asset.symbol: asset for asset in CATALOG}

DEFAULT_SELECTION: Final[tuple[str, ...]] = (
    "XAUUSD",
    "EURUSD",
    "GBPUSD",
    "^GSPC",
    "NVDA",
    "BTCUSD",
)
"""A first screen that spans every class, so the ranking has something to
compare rather than six variations on the dollar."""


def normalize(symbol: str) -> str:
    return symbol.strip().upper()


def by_class(asset_class: AssetClass) -> tuple[Asset, ...]:
    return tuple(asset for asset in CATALOG if asset.asset_class is asset_class)


def groups(asset_class: AssetClass) -> dict[str, tuple[Asset, ...]]:
    """Sub-headings within one class, in catalog order."""
    grouped: dict[str, list[Asset]] = {}
    for asset in by_class(asset_class):
        grouped.setdefault(asset.group, []).append(asset)
    return {group: tuple(assets) for group, assets in grouped.items()}


def lookup(symbol: str) -> Asset:
    """The catalog entry for ``symbol``, or a plausible stand-in for one that is
    not listed.

    Never raises. An unlisted ticker is a perfectly reasonable thing to screen -
    it just has no curated name - and failing the whole run because one symbol
    is off-menu would be a worse outcome than showing it under its own ticker.
    """
    symbol = normalize(symbol)
    if asset := _BY_SYMBOL.get(symbol):
        return asset
    return Asset(symbol=symbol, name=symbol, asset_class=_guess_class(symbol), group="Custom")


def _guess_class(symbol: str) -> AssetClass:
    """A best-effort class for an unlisted symbol, from its shape alone.

    Only used for grouping and for telling the ranking agent what it is looking
    at. Being wrong here costs a misfiled row, never a wrong number.
    """
    if symbol.startswith("^"):
        return AssetClass.INDEX
    if symbol.endswith("=F"):
        return AssetClass.COMMODITY
    if symbol.endswith("-USD"):
        return AssetClass.CRYPTO
    if len(symbol) == 6 and symbol.isalpha():
        return AssetClass.FOREX
    return AssetClass.STOCK
