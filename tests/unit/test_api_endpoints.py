"""Tests for the portfolio, orders, memory and trading endpoints."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from factories import make_episode
from fastapi.testclient import TestClient

from atp.application.use_cases import GetPortfolio, JournalTrade
from atp.composition import Container
from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.domain.models.memory import EpisodeKind, EpisodeMatch, MemoryEpisode
from atp.domain.models.trading import Portfolio, Position
from atp.infrastructure.config import Settings
from atp.infrastructure.persistence.json_portfolio import JsonPortfolioRepository
from atp.interfaces.api import create_app

NOW = datetime(2026, 7, 23, 12, tzinfo=UTC)


class StubProvider:
    def __init__(self, price: float = 320.0) -> None:
        self._price = price

    async def get_bars(
        self,
        symbol: str,
        interval: BarInterval,
        *,
        start: datetime,
        end: datetime | None = None,
    ) -> PriceHistory:
        bar = Bar(
            timestamp=NOW - timedelta(days=1),
            open=self._price,
            high=self._price * 1.01,
            low=self._price * 0.99,
            close=self._price,
            volume=1_000.0,
        )
        return PriceHistory(symbol=symbol, interval=interval, bars=(bar,))


class StubMemory:
    def __init__(self, matches: Sequence[EpisodeMatch] = ()) -> None:
        self._matches = tuple(matches)
        self.calls: list[dict[str, object]] = []

    async def remember(self, episode: MemoryEpisode) -> None:
        return None

    async def recall(
        self,
        query: str,
        *,
        symbol: str | None = None,
        kinds: Sequence[EpisodeKind] | None = None,
        regime_label: str | None = None,
        limit: int = 5,
    ) -> tuple[EpisodeMatch, ...]:
        self.calls.append({"query": query, "symbol": symbol, "kinds": kinds, "limit": limit})
        return self._matches


@pytest.fixture
def container(tmp_path: Path) -> Container:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    provider = StubProvider()
    repository = JsonPortfolioRepository(
        tmp_path / "portfolio.json", starting_cash=Decimal("100000")
    )
    return dataclasses.replace(
        Container.build(settings),
        market_data=provider,
        portfolio_repository=repository,
        get_portfolio=GetPortfolio(repository, provider, clock=lambda: NOW),
        memory=StubMemory((EpisodeMatch(episode=make_episode("episode-0"), score=0.71),)),
    )


@pytest.fixture
def client(container: Container) -> Iterator[TestClient]:
    with TestClient(create_app(container)) as test_client:
        yield test_client


# --- Portfolio --------------------------------------------------------------


def test_empty_portfolio_reports_starting_cash(client: TestClient) -> None:
    body = client.get("/portfolio").json()

    assert Decimal(body["cash"]) == Decimal("100000")
    assert Decimal(body["equity"]) == Decimal("100000")
    assert body["positions"] == []


async def test_positions_carry_server_computed_pnl(container: Container) -> None:
    """The dashboard must not have to reimplement P&L arithmetic."""
    await container.portfolio_repository.save(
        Portfolio(
            cash=Decimal("50000"),
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
    with TestClient(create_app(container)) as client:
        body = client.get("/portfolio").json()

    position = body["positions"][0]
    assert position["symbol"] == "AAPL"
    # Marked to the stub provider's live price of 320, not the stored 300.
    assert Decimal(position["current_price"]) == Decimal("320")
    assert Decimal(position["market_value"]) == Decimal("3200")
    assert Decimal(position["unrealized_pnl"]) == Decimal("200")
    assert float(position["unrealized_pnl_pct"]) == pytest.approx(6.667, abs=0.01)


def test_orders_endpoint_starts_empty(client: TestClient) -> None:
    assert client.get("/orders").json()["records"] == []


def test_orders_limit_is_validated(client: TestClient) -> None:
    assert client.get("/orders", params={"limit": 0}).status_code == 422
    assert client.get("/orders", params={"limit": 1000}).status_code == 422


# --- Memory -----------------------------------------------------------------


def test_memory_recall_defaults_the_query_to_the_symbol(
    client: TestClient, container: Container
) -> None:
    body = client.get("/memory/aapl").json()

    assert body["symbol"] == "AAPL"
    assert body["query"] == "AAPL"
    assert body["matches"][0]["episode"]["id"] == "episode-0"
    assert body["matches"][0]["score"] == 0.71

    memory = container.memory
    assert isinstance(memory, StubMemory)
    assert memory.calls[0]["symbol"] == "AAPL"


def test_memory_recall_passes_query_and_kind(client: TestClient, container: Container) -> None:
    client.get("/memory/AAPL", params={"query": "breakout", "kind": "trade", "limit": 3})

    memory = container.memory
    assert isinstance(memory, StubMemory)
    call = memory.calls[0]
    assert call["query"] == "breakout"
    assert call["kinds"] == [EpisodeKind.TRADE]
    assert call["limit"] == 3


def test_memory_rejects_an_unknown_kind(client: TestClient) -> None:
    assert client.get("/memory/AAPL", params={"kind": "gossip"}).status_code == 422


# --- Trading ----------------------------------------------------------------


def test_risk_check_returns_a_decision_without_trading(client: TestClient) -> None:
    response = client.post("/risk-check", json={"symbol": "AAPL", "stop_loss": 300})
    assert response.status_code == 200

    decision = response.json()["decision"]
    assert decision["proposal"]["symbol"] == "AAPL"
    assert decision["verdict"] in {"approved", "rejected"}
    assert client.get("/orders").json()["records"] == []


def test_a_gate_rejection_is_200_with_reasons_not_an_error(client: TestClient) -> None:
    """Risk limits default to requiring a stop; omitting one must be refused."""
    response = client.post("/trade", json={"symbol": "AAPL", "quantity": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["executed"] is False
    assert body["order"] is None
    assert body["decision"]["verdict"] == "rejected"
    assert body["decision"]["violations"]


def test_an_approved_trade_fills_and_is_audited(client: TestClient) -> None:
    response = client.post("/trade", json={"symbol": "AAPL", "quantity": 5, "stop_loss": 300})

    assert response.status_code == 201
    body = response.json()
    assert body["executed"] is True
    assert body["order"]["status"] == "filled"

    records = client.get("/orders").json()["records"]
    assert len(records) == 1
    assert records[0]["order"]["symbol"] == "AAPL"
    assert records[0]["decision"]["verdict"] == "approved"


def test_an_executed_trade_is_journalled_to_memory(
    client: TestClient, container: Container
) -> None:
    journalled: list[MemoryEpisode] = []

    class RecordingMemory(StubMemory):
        async def remember(self, episode: MemoryEpisode) -> None:
            journalled.append(episode)

    memory = RecordingMemory()
    app_container = dataclasses.replace(
        container, memory=memory, journal_trade=JournalTrade(memory)
    )
    with TestClient(create_app(app_container)) as recording_client:
        recording_client.post("/trade", json={"symbol": "AAPL", "quantity": 5, "stop_loss": 300})

    assert len(journalled) == 1
    assert journalled[0].kind is EpisodeKind.TRADE
    assert journalled[0].symbol == "AAPL"


def test_trade_input_is_validated(client: TestClient) -> None:
    assert client.post("/trade", json={"symbol": "AAPL", "quantity": -5}).status_code == 422
    assert client.post("/trade", json={"symbol": "AAPL", "side": "hold"}).status_code == 422
    assert client.post("/trade", json={}).status_code == 422
