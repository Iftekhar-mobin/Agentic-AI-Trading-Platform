"""Tests for the JSON portfolio repository and portfolio/risk use cases."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from atp.application.use_cases import CheckTradeRisk, GetPortfolio
from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.domain.models.trading import Portfolio, Position, RiskLimits
from atp.infrastructure.persistence.json_portfolio import JsonPortfolioRepository

NOW = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)


class StubProvider:
    """Serves a fixed latest close per symbol."""

    def __init__(self, prices: dict[str, float]) -> None:
        self._prices = prices

    async def get_bars(
        self,
        symbol: str,
        interval: BarInterval,
        *,
        start: datetime,
        end: datetime | None = None,
    ) -> PriceHistory:
        price = self._prices[symbol]
        bar = Bar(
            timestamp=NOW - timedelta(days=1),
            open=price,
            high=price * 1.01,
            low=price * 0.99,
            close=price,
            volume=1_000.0,
        )
        return PriceHistory(symbol=symbol, interval=interval, bars=(bar,))


def make_repository(tmp_path: Path) -> JsonPortfolioRepository:
    return JsonPortfolioRepository(tmp_path / "portfolio.json", starting_cash=Decimal("100000"))


class TestJsonPortfolioRepository:
    async def test_seeds_when_missing(self, tmp_path: Path) -> None:
        portfolio = await make_repository(tmp_path).load()
        assert portfolio.cash == Decimal("100000")
        assert portfolio.positions == ()

    async def test_round_trip_preserves_decimals(self, tmp_path: Path) -> None:
        repository = make_repository(tmp_path)
        portfolio = Portfolio(
            cash=Decimal("12345.67"),
            positions=(
                Position(
                    symbol="AAPL",
                    quantity=Decimal("10"),
                    avg_entry_price=Decimal("325.53"),
                    current_price=Decimal("330.10"),
                ),
            ),
            equity_peak=Decimal("20000"),
        )
        await repository.save(portfolio)
        assert await repository.load() == portfolio

    async def test_save_overwrites(self, tmp_path: Path) -> None:
        repository = make_repository(tmp_path)
        await repository.save(Portfolio(cash=Decimal("1")))
        await repository.save(Portfolio(cash=Decimal("2")))
        assert (await repository.load()).cash == Decimal("2")


class TestGetPortfolio:
    async def test_refreshes_prices_and_tracks_peak(self, tmp_path: Path) -> None:
        repository = make_repository(tmp_path)
        await repository.save(
            Portfolio(
                cash=Decimal("1000"),
                positions=(
                    Position(
                        symbol="AAPL",
                        quantity=Decimal("10"),
                        avg_entry_price=Decimal("300"),
                        current_price=Decimal("300"),
                    ),
                ),
            )
        )
        use_case = GetPortfolio(repository, StubProvider({"AAPL": 320.0}), clock=lambda: NOW)
        portfolio = await use_case.execute()

        aapl = portfolio.position_for("AAPL")
        assert aapl is not None
        assert aapl.current_price == Decimal("320.0")
        assert portfolio.equity == Decimal("1000") + Decimal("3200.0")
        assert portfolio.equity_peak == portfolio.equity
        # The refreshed state (incl. peak) was persisted.
        assert (await repository.load()).equity_peak == portfolio.equity


class TestCheckTradeRisk:
    async def test_auto_sizes_from_stop_and_approves(self, tmp_path: Path) -> None:
        repository = make_repository(tmp_path)
        provider = StubProvider({"AAPL": 100.0})
        use_case = CheckTradeRisk(
            GetPortfolio(repository, provider, clock=lambda: NOW),
            provider,
            RiskLimits(),
            clock=lambda: NOW,
        )
        decision = await use_case.execute("aapl", stop_loss=Decimal("95"))

        # 1% of 100k = 1000 budget / $5 per-share risk = 200 shares.
        assert decision.proposal.quantity == Decimal("200")
        assert decision.approved

    async def test_rejection_flows_through(self, tmp_path: Path) -> None:
        repository = make_repository(tmp_path)
        provider = StubProvider({"AAPL": 100.0})
        use_case = CheckTradeRisk(
            GetPortfolio(repository, provider, clock=lambda: NOW),
            provider,
            RiskLimits(),
            clock=lambda: NOW,
        )
        decision = await use_case.execute("AAPL")  # no stop, no quantity

        assert not decision.approved
        assert "missing_stop_loss" in [v.rule for v in decision.violations]
