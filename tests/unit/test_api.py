"""API tests: real app + real graph, with the analysis use cases stubbed."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from typing import cast

import pytest
from factories import (
    make_fundamental_report,
    make_news_report,
    make_sentiment_report,
    make_technical_report,
)
from fastapi.testclient import TestClient
from pydantic import BaseModel

from atp.application.orchestration import TradingOrchestrator
from atp.application.use_cases import (
    AnalyzeFundamentals,
    AnalyzeNews,
    AnalyzeSentiment,
    AnalyzeTicker,
)
from atp.composition import Container
from atp.domain.errors import InsufficientDataError, InsufficientHistoryError
from atp.domain.models.market import BarInterval
from atp.infrastructure.config import Settings
from atp.interfaces.api import create_app


class StubUseCase:
    def __init__(
        self,
        report_factory: Callable[[str], BaseModel],
        *,
        error: Exception | None = None,
    ) -> None:
        self._report_factory = report_factory
        self._error = error

    async def execute(self, symbol: str, interval: BarInterval = BarInterval.DAY_1) -> BaseModel:
        if self._error is not None:
            raise self._error
        return self._report_factory(symbol)


def make_client(**errors: Exception) -> TestClient:
    orchestrator = TradingOrchestrator(
        cast(AnalyzeTicker, StubUseCase(make_technical_report, error=errors.get("technical"))),
        cast(
            AnalyzeFundamentals,
            StubUseCase(make_fundamental_report, error=errors.get("fundamental")),
        ),
        cast(AnalyzeNews, StubUseCase(make_news_report, error=errors.get("news"))),
        cast(
            AnalyzeSentiment,
            StubUseCase(make_sentiment_report, error=errors.get("sentiment")),
        ),
    )
    container = Container.build(Settings(_env_file=None))
    container = dataclasses.replace(container, orchestrator=orchestrator)
    return TestClient(create_app(container))


@pytest.fixture
def client() -> Iterator[TestClient]:
    with make_client() as test_client:
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
    assert sorted(body["completed_agents"]) == [
        "fundamental_analysis",
        "news_analysis",
        "sentiment_analysis",
        "technical_analysis",
    ]
    assert body["failures"] == []
    assessment = body["technical_report"]["assessment"]
    assert assessment["direction"] == "bullish"
    assert assessment["confidence"] == 0.7
    assert assessment["evidence"]
    assert assessment["invalidation_conditions"]
    assert body["fundamental_report"]["assessment"]["direction"] == "neutral"
    assert body["news_report"]["assessment"]["key_themes"] == ["product cycle"]
    assert body["sentiment_report"]["summary"]["weighted_polarity"] == 0.8
    assert body["sentiment_report"]["model_name"] == "lexicon-v1"


def test_agent_selection_is_honoured(client: TestClient) -> None:
    response = client.post("/analysis", json={"symbol": "AAPL", "agents": ["news_analysis"]})
    assert response.status_code == 200

    body = response.json()
    assert body["requested_agents"] == ["news_analysis"]
    assert body["completed_agents"] == ["news_analysis"]
    assert body["news_report"] is not None
    assert body["technical_report"] is None


def test_unknown_agent_returns_422(client: TestClient) -> None:
    response = client.post("/analysis", json={"symbol": "AAPL", "agents": ["astrology"]})
    assert response.status_code == 422
    assert "astrology" in response.json()["detail"]


def test_partial_failure_still_returns_the_other_reports() -> None:
    with make_client(technical=InsufficientHistoryError("only 3 bars")) as client:
        response = client.post("/analysis", json={"symbol": "AAPL"})

    assert response.status_code == 200
    body = response.json()
    assert body["technical_report"] is None
    assert body["news_report"] is not None
    assert body["failures"][0]["agent"] == "technical_analysis"
    assert "3 bars" in body["failures"][0]["error"]


def test_total_failure_returns_502_with_failures() -> None:
    errors = {
        "technical": InsufficientHistoryError("only 3 bars"),
        "fundamental": InsufficientDataError("no metrics"),
        "news": InsufficientDataError("no articles"),
        "sentiment": InsufficientDataError("no articles"),
    }
    with make_client(**errors) as client:
        response = client.post("/analysis", json={"symbol": "AAPL"})

    assert response.status_code == 502
    detail = response.json()["detail"]
    agents = {failure["agent"] for failure in detail["failures"]}
    assert agents == {
        "technical_analysis",
        "fundamental_analysis",
        "news_analysis",
        "sentiment_analysis",
    }


def test_analysis_validates_input(client: TestClient) -> None:
    assert client.post("/analysis", json={"symbol": ""}).status_code == 422
    assert client.post("/analysis", json={"symbol": "AAPL", "interval": "7y"}).status_code == 422
    assert client.post("/analysis", json={}).status_code == 422
