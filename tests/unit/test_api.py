"""API tests: real app + real graph, with the analysis use case stubbed."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import cast

import pytest
from fastapi.testclient import TestClient

from atp.application.orchestration import TradingOrchestrator
from atp.application.use_cases import AnalyzeTicker
from atp.composition import Container
from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.analysis import (
    SignalDirection,
    TechnicalAssessment,
    TechnicalReport,
)
from atp.domain.models.explainability import Evidence
from atp.domain.models.market import BarInterval
from atp.infrastructure.config import Settings
from atp.interfaces.api import create_app


def make_report(symbol: str) -> TechnicalReport:
    return TechnicalReport(
        symbol=symbol,
        interval=BarInterval.DAY_1,
        as_of=datetime(2026, 7, 22, tzinfo=UTC),
        latest_close=325.89,
        readings=(),
        signal_counts=dict.fromkeys(SignalDirection, 0),
        assessment=TechnicalAssessment(
            direction=SignalDirection.BULLISH,
            reasoning="Trend indicators align.",
            confidence=0.7,
            evidence=(Evidence(source="sma_trend", statement="Above SMA(50)."),),
            invalidation_conditions=("Close below SMA(50).",),
        ),
    )


class StubAnalyzeTicker:
    def __init__(self, *, error: Exception | None = None) -> None:
        self._error = error

    async def execute(
        self, symbol: str, interval: BarInterval = BarInterval.DAY_1
    ) -> TechnicalReport:
        if self._error is not None:
            raise self._error
        return make_report(symbol)


def make_client(stub: StubAnalyzeTicker) -> TestClient:
    container = Container.build(Settings(_env_file=None))
    container = dataclasses.replace(
        container, orchestrator=TradingOrchestrator(cast(AnalyzeTicker, stub))
    )
    return TestClient(create_app(container))


@pytest.fixture
def client() -> Iterator[TestClient]:
    with make_client(StubAnalyzeTicker()) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analysis_success(client: TestClient) -> None:
    response = client.post("/analysis", json={"symbol": "aapl"})
    assert response.status_code == 200

    body = response.json()
    assert body["symbol"] == "AAPL"
    assert body["interval"] == "1d"
    assert body["completed_agents"] == ["technical_analysis"]
    assessment = body["technical_report"]["assessment"]
    assert assessment["direction"] == "bullish"
    assert assessment["confidence"] == 0.7
    assert assessment["evidence"]
    assert assessment["invalidation_conditions"]


def test_analysis_failure_returns_502_with_failures() -> None:
    stub = StubAnalyzeTicker(error=InsufficientHistoryError("only 3 bars"))
    with make_client(stub) as client:
        response = client.post("/analysis", json={"symbol": "AAPL"})

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["failures"][0]["agent"] == "technical_analysis"
    assert "3 bars" in detail["failures"][0]["error"]


def test_analysis_validates_input(client: TestClient) -> None:
    assert client.post("/analysis", json={"symbol": ""}).status_code == 422
    assert client.post("/analysis", json={"symbol": "AAPL", "interval": "7y"}).status_code == 422
    assert client.post("/analysis", json={}).status_code == 422
