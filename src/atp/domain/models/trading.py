"""Trading domain models: proposals, positions, portfolio, risk limits.

Monetary values are ``Decimal`` throughout — market-data floats stop at this
boundary (convert via ``Decimal(str(x))``). Models are frozen; state changes
produce new instances.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class TradeProposal(BaseModel):
    """An intended trade — the input to the risk gate, never yet an order."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1, max_length=12)
    side: OrderSide
    quantity: Decimal = Field(gt=0)
    entry_price: Decimal = Field(gt=0)
    stop_loss: Decimal | None = Field(default=None, gt=0)
    take_profit: Decimal | None = Field(default=None, gt=0)
    strategy_name: str | None = None

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @property
    def notional(self) -> Decimal:
        return self.quantity * self.entry_price


class Position(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1)
    quantity: Decimal = Field(gt=0)
    avg_entry_price: Decimal = Field(gt=0)
    current_price: Decimal = Field(gt=0)

    @property
    def market_value(self) -> Decimal:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> Decimal:
        return (self.current_price - self.avg_entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> Decimal:
        return (self.current_price / self.avg_entry_price - 1) * 100


class Portfolio(BaseModel):
    model_config = ConfigDict(frozen=True)

    cash: Decimal = Field(ge=0)
    positions: tuple[Position, ...] = ()
    equity_peak: Decimal | None = Field(default=None, gt=0)
    realized_pnl_today: Decimal = Decimal("0")

    @model_validator(mode="after")
    def _check_unique_symbols(self) -> Portfolio:
        symbols = [position.symbol for position in self.positions]
        if len(symbols) != len(set(symbols)):
            msg = "portfolio cannot hold duplicate symbols"
            raise ValueError(msg)
        return self

    @property
    def total_market_value(self) -> Decimal:
        return sum((position.market_value for position in self.positions), Decimal("0"))

    @property
    def equity(self) -> Decimal:
        return self.cash + self.total_market_value

    @property
    def exposure_pct(self) -> Decimal:
        if self.equity == 0:
            return Decimal("0")
        return self.total_market_value / self.equity * 100

    @property
    def drawdown_pct(self) -> Decimal:
        """Current drawdown from the equity peak (>= 0)."""
        if self.equity_peak is None or self.equity_peak == 0:
            return Decimal("0")
        drop = (1 - self.equity / self.equity_peak) * 100
        return max(drop, Decimal("0"))

    def position_for(self, symbol: str) -> Position | None:
        symbol = symbol.strip().upper()
        return next((p for p in self.positions if p.symbol == symbol), None)

    def with_prices(self, prices: dict[str, Decimal]) -> Portfolio:
        updated = tuple(
            position.model_copy(update={"current_price": prices[position.symbol]})
            if position.symbol in prices
            else position
            for position in self.positions
        )
        return self.model_copy(update={"positions": updated})

    def with_peak_updated(self) -> Portfolio:
        peak = self.equity if self.equity_peak is None else max(self.equity_peak, self.equity)
        return self.model_copy(update={"equity_peak": peak})


class RiskLimits(BaseModel):
    """Hard limits enforced by the risk engine. Configuration, not suggestion."""

    model_config = ConfigDict(frozen=True)

    max_risk_per_trade_pct: float = Field(default=1.0, gt=0, le=10)
    max_position_pct: float = Field(default=20.0, gt=0, le=100)
    max_total_exposure_pct: float = Field(default=100.0, gt=0, le=200)
    max_positions: int = Field(default=10, ge=1, le=100)
    max_daily_loss_pct: float = Field(default=3.0, gt=0, le=50)
    max_drawdown_pct: float = Field(default=20.0, gt=0, le=90)
    require_stop_loss: bool = True


class RiskViolation(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule: str
    detail: str


class RiskVerdict(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


class RiskDecision(BaseModel):
    """The risk gate's output: verdict, every violation, and the numbers behind it.

    ``metrics`` carries each evaluated quantity so decisions are explainable
    and auditable without re-running the engine.
    """

    model_config = ConfigDict(frozen=True)

    verdict: RiskVerdict
    proposal: TradeProposal
    limits: RiskLimits
    violations: tuple[RiskViolation, ...]
    metrics: dict[str, float]

    @property
    def approved(self) -> bool:
        return self.verdict is RiskVerdict.APPROVED
