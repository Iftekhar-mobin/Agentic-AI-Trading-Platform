"""Translation from the symbols people type to the ones Yahoo serves.

Traders write ``XAUUSD`` for gold and ``EURUSD`` for the euro. Yahoo serves
neither: it wants ``GC=F`` for gold and ``EURUSD=X`` for the pair. Left
untranslated, ``XAUUSD`` returns an empty frame and every agent reports
"insufficient data" — a confusing failure, because nothing is actually broken.

The mapping lives here rather than in the domain because it is a property of one
vendor, not of markets. The domain keeps the symbol the user asked for; only the
outbound HTTP call is rewritten, so reports still say ``XAUUSD``.

Two rules, in order of specificity:

- **Spot metals** have no Yahoo spot ticker at all, so they resolve to the
  front-month COMEX/NYMEX future (``XAUUSD`` -> ``GC=F``). That is a real
  instrument with a real basis against spot, not the same series — see
  ``METAL_FUTURES`` for the caveat.
- **Currency pairs and crypto** follow Yahoo's own suffix conventions
  (``=X`` and ``-USD`` respectively).

Anything already vendor-shaped, and anything unrecognized, passes through
untouched: an unknown ticker is Yahoo's problem to reject, not this module's to
guess at.
"""

from __future__ import annotations

from typing import Final

METAL_FUTURES: Final[dict[str, str]] = {
    "XAU": "GC=F",  # gold, COMEX front-month
    "XAG": "SI=F",  # silver
    "XPT": "PL=F",  # platinum
    "XPD": "PA=F",  # palladium
}
"""Spot metal code -> front-month future.

The future is a proxy, not the spot series: it carries contract roll gaps and a
basis of a few dollars against spot. For trend, structure and levels — what the
agents actually reason about — that difference is immaterial. For anything
settling against a spot fix, it is not.
"""

CRYPTO_QUOTES: Final[frozenset[str]] = frozenset({"USD", "USDT", "USDC"})

CRYPTO_BASES: Final[frozenset[str]] = frozenset(
    {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "MATIC", "LTC", "BCH"}
)
"""Bases quoted as ``BASE-USD`` on Yahoo. Kept explicit: three-letter crypto
tickers collide with equity tickers, and silently rewriting one would be worse
than not resolving it."""

ISO_CURRENCIES: Final[frozenset[str]] = frozenset(
    {
        "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD",
        "CNY", "HKD", "SGD", "SEK", "NOK", "DKK", "PLN", "ZAR",
        "MXN", "TRY", "INR", "BRL", "KRW", "THB", "HUF", "CZK",
    }
)  # fmt: skip

_VENDOR_MARKERS: Final[tuple[str, ...]] = ("=", "-", "^", ".")
"""Characters that only appear in symbols already written in vendor form."""


def normalize(symbol: str) -> str:
    """Canonical form of a user-supplied symbol: trimmed and upper-cased."""
    return symbol.strip().upper()


def to_vendor_symbol(symbol: str) -> str:
    """The Yahoo ticker for a symbol, or the symbol itself when none is known.

    >>> to_vendor_symbol("xauusd")
    'GC=F'
    >>> to_vendor_symbol("EURUSD")
    'EURUSD=X'
    >>> to_vendor_symbol("BTCUSD")
    'BTC-USD'
    >>> to_vendor_symbol("AAPL")
    'AAPL'
    """
    symbol = normalize(symbol)
    if any(marker in symbol for marker in _VENDOR_MARKERS):
        return symbol
    # Three-letter base plus a three- or four-letter quote (USDT, USDC).
    if not 6 <= len(symbol) <= 7 or not symbol.isalpha():
        return symbol

    base, quote = symbol[:3], symbol[3:]
    if quote == "USD" and (future := METAL_FUTURES.get(base)):
        return future
    if base in CRYPTO_BASES and quote in CRYPTO_QUOTES:
        # Yahoo quotes crypto against USD only; a stablecoin pair is close
        # enough to the dollar pair for chart structure to carry over.
        return f"{base}-USD"
    if base in ISO_CURRENCIES and quote in ISO_CURRENCIES and base != quote:
        return f"{symbol}=X"
    return symbol


def is_translated(symbol: str) -> bool:
    """Whether ``symbol`` resolves to a different vendor ticker.

    Callers log this so an operator reading ``XAUUSD`` in a report can see which
    series actually produced the numbers.
    """
    return to_vendor_symbol(symbol) != normalize(symbol)
