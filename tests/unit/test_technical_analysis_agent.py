"""Tests for the Technical Analysis agent, with the LLM faked at the port."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TypeVar

import pytest
from pydantic import BaseModel

from atp.application.agents import TechnicalAnalysisAgent
from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.analysis import SignalDirection, TechnicalAssessment
from atp.domain.models.explainability import Evidence
from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.infrastructure.indicators import PandasIndicatorEngine

T = TypeVar("T", bound=BaseModel)


class FakeLLM:
    def __init__(self, response: BaseModel) -> None:
        self._response = response
        self.system: str | None = None
        self.prompt: str | None = None

    async def generate_structured(self, response_model: type[T], *, system: str, prompt: str) -> T:
        self.system = system
        self.prompt = prompt
        assert isinstance(self._response, response_model)
        return self._response


def make_history(n: int = 80) -> PriceHistory:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    closes = [100.0 + 0.3 * i + (i % 7) for i in range(n)]  # rising with pullbacks
    bars = tuple(
        Bar(
            timestamp=start + timedelta(days=i),
            open=close - 0.5,
            high=close + 1.0,
            low=close - 1.5,
            close=close,
            volume=1_000.0,
        )
        for i, close in enumerate(closes)
    )
    return PriceHistory(symbol="AAPL", interval=BarInterval.DAY_1, bars=bars)


def make_assessment(*, source: str = "sma_trend") -> TechnicalAssessment:
    return TechnicalAssessment(
        direction=SignalDirection.BULLISH,
        reasoning="Trend indicators align to the upside.",
        confidence=0.7,
        evidence=(Evidence(source=source, statement="Close is above the SMA(50)."),),
        invalidation_conditions=("A daily close below the SMA(50).",),
    )


async def test_report_combines_deterministic_and_llm_layers() -> None:
    llm = FakeLLM(make_assessment())
    agent = TechnicalAnalysisAgent(llm, PandasIndicatorEngine())
    history = make_history()

    report = await agent.analyze(history)

    assert report.symbol == "AAPL"
    assert report.latest_close == history.bars[-1].close
    assert report.as_of == history.bars[-1].timestamp
    assert len(report.readings) == 9
    assert report.assessment.direction is SignalDirection.BULLISH
    assert sum(report.signal_counts.values()) == len(report.readings)


async def test_llm_receives_readings_not_raw_bars() -> None:
    llm = FakeLLM(make_assessment())
    agent = TechnicalAnalysisAgent(llm, PandasIndicatorEngine())

    await agent.analyze(make_history())

    assert llm.prompt is not None
    payload = json.loads(llm.prompt)
    assert payload["symbol"] == "AAPL"
    names = {reading["name"] for reading in payload["readings"]}
    assert "rsi_14" in names
    assert "bars" not in payload  # the LLM interprets readings, it never sees raw prices
    assert llm.system is not None
    assert "never invent" in llm.system.lower()


async def test_empty_history_is_rejected(caplog: pytest.LogCaptureFixture) -> None:
    agent = TechnicalAnalysisAgent(FakeLLM(make_assessment()), PandasIndicatorEngine())
    with pytest.raises(InsufficientHistoryError):
        await agent.analyze(PriceHistory(symbol="AAPL", interval=BarInterval.DAY_1))


async def test_envelope_constraints_are_enforced_by_schema() -> None:
    with pytest.raises(ValueError, match="evidence"):
        TechnicalAssessment(
            direction=SignalDirection.BULLISH,
            reasoning="No evidence given.",
            confidence=0.9,
            evidence=(),
            invalidation_conditions=("Anything.",),
        )
    with pytest.raises(ValueError, match="confidence"):
        TechnicalAssessment(
            direction=SignalDirection.BULLISH,
            reasoning="Overconfident.",
            confidence=1.5,
            evidence=(Evidence(source="rsi_14", statement="x"),),
            invalidation_conditions=("Anything.",),
        )
