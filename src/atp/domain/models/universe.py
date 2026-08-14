"""What an instrument *is*, so a screen can be assembled from categories.

A trader does not think in tickers, they think in markets: "the majors", "the
big US tech names", "the indices". Until now the platform only accepted a
symbol typed by hand, which makes "compare gold against the majors" a
copy-paste exercise and quietly punishes typos with an empty price frame.

An ``Asset`` gives a symbol a name and a class. Nothing here knows about any
data vendor - ``infrastructure.vendor_symbols`` still owns that translation -
so the same catalog would survive swapping Yahoo out.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AssetClass(StrEnum):
    """The market an instrument trades in.

    Coarse on purpose. The class exists so a screen can be built by category
    and so the ranking agent knows it is comparing an index against a currency
    pair - not to drive any analysis logic, which reads prices and nothing else.
    """

    FOREX = "forex"
    STOCK = "stock"
    INDEX = "index"
    COMMODITY = "commodity"
    CRYPTO = "crypto"

    @property
    def label(self) -> str:
        """Heading an interface can show without a lookup table of its own."""
        return {
            AssetClass.FOREX: "Forex",
            AssetClass.STOCK: "Stocks",
            AssetClass.INDEX: "Indices",
            AssetClass.COMMODITY: "Commodities",
            AssetClass.CRYPTO: "Crypto",
        }[self]


class Asset(BaseModel):
    """One selectable instrument."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1, max_length=12, description="As the trader writes it")
    name: str = Field(min_length=1, description="e.g. 'Euro / US Dollar'")
    asset_class: AssetClass
    group: str = Field(
        default="",
        description="Sub-heading within the class, e.g. 'Majors' or 'Precious metals'",
    )

    @property
    def display(self) -> str:
        return f"{self.symbol} — {self.name}"
