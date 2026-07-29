"""Tests for the Chart Pattern and Market Research agents (LLM faked at the port)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TypeVar

import pytest
from pydantic import BaseModel

from atp.application.agents import ChartPatternAgent, MarketResearchAgent
from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.analysis import SignalDirection
from atp.domain.models.explainability import Evidence
from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.domain.models.patterns import PatternAssessment
from atp.domain.models.research import MarketResearchAssessment

T = TypeVar("T", bound=BaseModel)

START = datetime(2026, 1, 1, tzinfo=UTC)


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


def legs(start: float, *segments: tuple[float, int]) -> list[float]:
    closes = [start]
    current = start
    for target, steps in segments:
        step = (target - current) / steps
        closes.extend(current + step * index for index in range(1, steps + 1))
        current = target
    return closes


def history(
    closes: list[float],
    interval: BarInterval = BarInterval.DAY_1,
    *,
    symbol: str = "AAPL",
) -> PriceHistory:
    step = timedelta(minutes=interval.minutes)
    bars = tuple(
        Bar(
            timestamp=START + step * index,
            open=close,
            high=close * 1.005,
            low=close * 0.995,
            close=close,
            volume=1_000.0,
        )
        for index, close in enumerate(closes)
    )
    return PriceHistory(symbol=symbol, interval=interval, bars=bars)


def zigzag(interval: BarInterval = BarInterval.DAY_1) -> PriceHistory:
    return history(legs(100, (130, 20), (115, 20), (132, 25)), interval)


def growth(start: float, rate: float, count: int = 300) -> list[float]:
    return [start * (1 + rate) ** index for index in range(count)]


def pattern_assessment(source: str = "1d:uptrend_structure") -> PatternAssessment:
    return PatternAssessment(
        direction=SignalDirection.BULLISH,
        reasoning="Structure is constructive on the higher timeframe.",
        confidence=0.55,
        evidence=(Evidence(source=source, statement="Higher highs and higher lows."),),
        invalidation_conditions=("A close below the most recent swing low.",),
        timeframe_alignment="Both timeframes point the same way.",
    )


def research_assessment(source: str = "relative_strength_3m") -> MarketResearchAssessment:
    return MarketResearchAssessment(
        direction=SignalDirection.BULLISH,
        reasoning="Leading a rising market.",
        confidence=0.6,
        evidence=(Evidence(source=source, statement="Outperforming by 6pp."),),
        invalidation_conditions=("Relative strength turning negative.",),
    )


# --- Chart pattern agent ----------------------------------------------------


async def test_pattern_report_combines_detection_and_llm() -> None:
    llm = FakeLLM(pattern_assessment())
    report = await ChartPatternAgent(llm).analyze([zigzag()])

    assert report.symbol == "AAPL"
    assert len(report.timeframes) == 1
    assert report.primary.patterns
    assert report.assessment.direction is SignalDirection.BULLISH


async def test_llm_receives_detected_patterns_not_raw_bars() -> None:
    llm = FakeLLM(pattern_assessment())
    await ChartPatternAgent(llm).analyze([zigzag()])

    assert llm.prompt is not None
    payload = json.loads(llm.prompt)
    timeframe = payload["timeframes"][0]
    assert timeframe["interval"] == "1d"
    assert "patterns" in timeframe
    assert "levels" in timeframe
    assert "bars" not in timeframe  # detection happened in code, not in the model
    assert all("confirmed" in pattern for pattern in timeframe["patterns"])


async def test_prompt_forbids_inventing_patterns() -> None:
    """The whole reason detection is deterministic is to stop confabulation."""
    llm = FakeLLM(pattern_assessment())
    await ChartPatternAgent(llm).analyze([zigzag()])

    assert llm.system is not None
    assert "only patterns that exist" in llm.system
    assert "confirmed" in llm.system


async def test_patterns_are_detected_per_timeframe() -> None:
    llm = FakeLLM(pattern_assessment())
    report = await ChartPatternAgent(llm).analyze(
        [zigzag(BarInterval.DAY_1), zigzag(BarInterval.HOUR_4)]
    )

    assert report.intervals == (BarInterval.DAY_1, BarInterval.HOUR_4)
    assert all(frame.patterns for frame in report.timeframes)
    assert report.primary.interval is BarInterval.DAY_1


async def test_short_history_is_rejected() -> None:
    llm = FakeLLM(pattern_assessment())
    with pytest.raises(InsufficientHistoryError, match="detect chart patterns"):
        await ChartPatternAgent(llm).analyze([history(legs(100, (110, 20)))])


async def test_no_timeframes_is_rejected() -> None:
    llm = FakeLLM(pattern_assessment())
    with pytest.raises(InsufficientHistoryError, match="no timeframes"):
        await ChartPatternAgent(llm).analyze([])


async def test_fabricated_pattern_citation_is_logged_not_raised() -> None:
    llm = FakeLLM(pattern_assessment(source="1d:cup_and_handle"))
    report = await ChartPatternAgent(llm).analyze([zigzag()])
    assert report.assessment.evidence[0].source == "1d:cup_and_handle"


# --- Market research agent --------------------------------------------------


async def test_research_report_carries_the_benchmark_and_readings() -> None:
    llm = FakeLLM(research_assessment())
    symbol = history(growth(100, 0.003))
    benchmark = history(growth(400, 0.001), symbol="SPY")

    report = await MarketResearchAgent(llm).analyze(symbol, benchmark)

    assert report.symbol == "AAPL"
    assert report.benchmark == "SPY"
    assert report.readings
    assert report.overlapping_bars == len(symbol)
    assert sum(report.signal_counts.values()) == len(report.readings)


async def test_research_llm_sees_relative_measurements() -> None:
    llm = FakeLLM(research_assessment())
    await MarketResearchAgent(llm).analyze(
        history(growth(100, 0.003)), history(growth(400, 0.001), symbol="SPY")
    )

    assert llm.prompt is not None
    payload = json.loads(llm.prompt)
    assert payload["benchmark"] == "SPY"
    names = {reading["name"] for reading in payload["readings"]}
    assert "relative_strength_3m" in names
    assert "beta_correlation" in names
    assert llm.system is not None
    assert "relationship" in llm.system


async def test_research_rejects_insufficient_overlap() -> None:
    llm = FakeLLM(research_assessment())
    with pytest.raises(InsufficientHistoryError, match="market research"):
        await MarketResearchAgent(llm).analyze(
            history(growth(100, 0.003, 40)),
            history(growth(400, 0.001, 40), symbol="SPY"),
        )
