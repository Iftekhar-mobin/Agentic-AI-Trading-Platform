"""Tests for the supervisor graph, with the analysis use case stubbed."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

from atp.application.orchestration import TradingOrchestrator
from atp.application.use_cases import AnalyzeTicker
from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.analysis import (
    SignalDirection,
    TechnicalAssessment,
    TechnicalReport,
)
from atp.domain.models.explainability import Evidence
from atp.domain.models.market import BarInterval


def make_report(symbol: str = "AAPL") -> TechnicalReport:
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
        self.calls: list[str] = []

    async def execute(
        self, symbol: str, interval: BarInterval = BarInterval.DAY_1
    ) -> TechnicalReport:
        self.calls.append(symbol)
        if self._error is not None:
            raise self._error
        return make_report(symbol)


def make_orchestrator(stub: StubAnalyzeTicker) -> TradingOrchestrator:
    return TradingOrchestrator(cast(AnalyzeTicker, stub))


async def test_successful_run_produces_report() -> None:
    stub = StubAnalyzeTicker()
    state = await make_orchestrator(stub).run(" aapl ")

    assert state.symbol == "AAPL"
    assert state.technical_report is not None
    assert state.technical_report.assessment.direction is SignalDirection.BULLISH
    assert state.completed == ["technical_analysis"]
    assert state.failures == []
    assert stub.calls == ["AAPL"]


async def test_agent_runs_exactly_once() -> None:
    stub = StubAnalyzeTicker()
    await make_orchestrator(stub).run("AAPL")
    assert len(stub.calls) == 1


async def test_agent_failure_is_recorded_not_raised() -> None:
    stub = StubAnalyzeTicker(error=InsufficientHistoryError("only 3 bars"))
    state = await make_orchestrator(stub).run("AAPL")

    assert state.technical_report is None
    assert len(state.failures) == 1
    assert state.failures[0].agent == "technical_analysis"
    assert "3 bars" in state.failures[0].error
    # The failed agent is marked completed so the supervisor never retries it in-run.
    assert state.completed == ["technical_analysis"]
