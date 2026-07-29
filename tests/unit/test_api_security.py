"""Tests for API authentication, scopes, rate limiting and hardening."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from typing import cast

import pytest
from factories import (
    make_chart_pattern_report,
    make_fundamental_report,
    make_learning_report,
    make_market_research_report,
    make_news_report,
    make_sentiment_report,
    make_technical_report,
)
from fastapi.testclient import TestClient
from pydantic import BaseModel, SecretStr

from atp.application.orchestration import TradingOrchestrator
from atp.application.use_cases import (
    AnalyzeChartPatterns,
    AnalyzeFundamentals,
    AnalyzeMarketResearch,
    AnalyzeNews,
    AnalyzeSentiment,
    AnalyzeTicker,
    LearnFromContext,
)
from atp.composition import Container
from atp.domain.models.market import BarInterval
from atp.infrastructure.config import ApiKey, Environment, Scope, SecuritySettings, Settings
from atp.interfaces.api import create_app
from atp.interfaces.api.rate_limit import RateLimiter

READ_KEY = "read-key-value"
TRADE_KEY = "trade-key-value"


class StubUseCase:
    def __init__(self, factory) -> None:  # type: ignore[no-untyped-def]
        self._factory = factory

    async def execute(
        self,
        symbol: str,
        intervals: Sequence[BarInterval] | BarInterval = BarInterval.DAY_1,
        **kwargs: object,
    ) -> BaseModel:
        return self._factory(symbol)  # type: ignore[no-any-return]


def secured_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "keys": (
            ApiKey(name="dashboard", key=SecretStr(READ_KEY), scopes=(Scope.READ,)),
            ApiKey(
                name="trader",
                key=SecretStr(TRADE_KEY),
                scopes=(Scope.READ, Scope.TRADE),
            ),
        )
    }
    return Settings(_env_file=None, api=SecuritySettings(**{**defaults, **overrides}))


def make_client(settings: Settings) -> TestClient:
    orchestrator = TradingOrchestrator(
        cast(AnalyzeTicker, StubUseCase(make_technical_report)),
        cast(AnalyzeChartPatterns, StubUseCase(make_chart_pattern_report)),
        cast(AnalyzeMarketResearch, StubUseCase(make_market_research_report)),
        cast(AnalyzeFundamentals, StubUseCase(make_fundamental_report)),
        cast(AnalyzeNews, StubUseCase(make_news_report)),
        cast(AnalyzeSentiment, StubUseCase(make_sentiment_report)),
        cast(LearnFromContext, StubUseCase(make_learning_report)),
    )
    container = dataclasses.replace(Container.build(settings), orchestrator=orchestrator)
    return TestClient(create_app(container))


@pytest.fixture
def secured() -> Iterator[TestClient]:
    with make_client(secured_settings()) as client:
        yield client


@pytest.fixture
def open_api() -> Iterator[TestClient]:
    """No keys configured — the laptop mode."""
    with make_client(Settings(_env_file=None)) as client:
        yield client


def auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


# --- Authentication ---------------------------------------------------------


def test_health_needs_no_credentials(secured: TestClient) -> None:
    """Liveness probes cannot be expected to hold a key."""
    assert secured.get("/health").status_code == 200


def test_protected_route_rejects_a_missing_key(secured: TestClient) -> None:
    response = secured.post("/analysis", json={"symbol": "AAPL"})
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_protected_route_rejects_a_wrong_key(secured: TestClient) -> None:
    response = secured.post("/analysis", json={"symbol": "AAPL"}, headers=auth("nope"))
    assert response.status_code == 401


def test_a_key_prefix_is_not_enough(secured: TestClient) -> None:
    response = secured.get("/portfolio", headers=auth(READ_KEY[:-1]))
    assert response.status_code == 401


def test_valid_key_is_accepted(secured: TestClient) -> None:
    response = secured.post("/analysis", json={"symbol": "AAPL"}, headers=auth(READ_KEY))
    assert response.status_code == 200


def test_unauthenticated_mode_serves_everything(open_api: TestClient) -> None:
    assert open_api.get("/portfolio").status_code == 200
    assert open_api.post("/analysis", json={"symbol": "AAPL"}).status_code == 200


def test_production_refuses_to_start_without_keys() -> None:
    with pytest.raises(ValueError, match="requires at least one API key"):
        Settings(_env_file=None, environment=Environment.PRODUCTION)


# --- Scopes -----------------------------------------------------------------


def test_read_key_can_read(secured: TestClient) -> None:
    assert secured.get("/portfolio", headers=auth(READ_KEY)).status_code == 200
    assert secured.get("/orders", headers=auth(READ_KEY)).status_code == 200
    assert secured.get("/memory/AAPL", headers=auth(READ_KEY)).status_code == 200


def test_read_key_cannot_trade(secured: TestClient) -> None:
    response = secured.post("/trade", json={"symbol": "AAPL"}, headers=auth(READ_KEY))
    assert response.status_code == 403
    assert "trade" in response.json()["detail"]


def test_risk_check_is_a_read_operation(secured: TestClient) -> None:
    """Seeing what the gate would say must not require permission to spend money."""
    response = secured.post(
        "/risk-check", json={"symbol": "AAPL", "stop_loss": 1}, headers=auth(READ_KEY)
    )
    assert response.status_code != 403


def test_trade_key_holds_both_scopes(secured: TestClient) -> None:
    assert secured.get("/portfolio", headers=auth(TRADE_KEY)).status_code == 200


# --- Correlation and hardening ---------------------------------------------


def test_every_response_carries_a_request_id(secured: TestClient) -> None:
    response = secured.get("/health")
    assert response.headers["X-Request-ID"]


def test_inbound_request_id_is_preserved(secured: TestClient) -> None:
    """A trace started at the gateway must stay one trace."""
    response = secured.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"


def test_security_headers_are_set(secured: TestClient) -> None:
    headers = secured.get("/health").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "no-referrer"


def test_hsts_only_in_production(secured: TestClient) -> None:
    assert "Strict-Transport-Security" not in secured.get("/health").headers


def test_domain_errors_map_to_status_and_carry_the_request_id() -> None:
    from atp.domain.errors import (
        DomainError,
        ExecutionError,
        InsufficientHistoryError,
        LLMGenerationError,
        RepositoryUnavailableError,
    )
    from atp.interfaces.api.errors import status_for

    assert status_for(InsufficientHistoryError("x")) == 422
    assert status_for(ExecutionError("x")) == 409
    assert status_for(LLMGenerationError("x")) == 502
    assert status_for(RepositoryUnavailableError("x")) == 503
    assert status_for(DomainError("x")) == 500


# --- Rate limiting ----------------------------------------------------------


def test_limiter_allows_up_to_the_limit_then_blocks() -> None:
    limiter = RateLimiter(3, clock=lambda: 100.0)

    assert [limiter.check("caller")[0] for _ in range(4)] == [True, True, True, False]


def test_limiter_reports_remaining_budget() -> None:
    limiter = RateLimiter(3, clock=lambda: 100.0)
    assert limiter.check("caller") == (True, 2)
    assert limiter.check("caller") == (True, 1)


def test_limiter_keeps_callers_in_separate_buckets() -> None:
    limiter = RateLimiter(1, clock=lambda: 100.0)
    assert limiter.check("a")[0]
    assert limiter.check("b")[0]
    assert not limiter.check("a")[0]


def test_limiter_window_resets() -> None:
    now = 100.0
    limiter = RateLimiter(1, clock=lambda: now)
    assert limiter.check("caller")[0]
    assert not limiter.check("caller")[0]

    now = 161.0
    assert limiter.check("caller")[0]


def test_zero_limit_disables_the_limiter() -> None:
    limiter = RateLimiter(0)
    assert not limiter.enabled
    assert all(limiter.check("caller")[0] for _ in range(100))


def test_requests_are_throttled_over_http() -> None:
    settings = secured_settings(rate_limit_per_minute=2)
    with make_client(settings) as client:
        statuses = [client.get("/portfolio", headers=auth(READ_KEY)).status_code for _ in range(3)]

    assert statuses[:2] == [200, 200]
    assert statuses[2] == 429


def test_health_is_never_throttled() -> None:
    settings = secured_settings(rate_limit_per_minute=1)
    with make_client(settings) as client:
        statuses = [client.get("/health").status_code for _ in range(5)]

    assert statuses == [200] * 5
