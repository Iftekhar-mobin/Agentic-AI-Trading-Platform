"""Tests for the Technical Analysis agent, with the LLM faked at the port."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import TypeVar

import pytest
from pydantic import BaseModel

from atp.application.agents import TechnicalAnalysisAgent
from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.analysis import (
    SignalDirection,
    TechnicalAssessment,
    net_direction,
)
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


def make_history(
    n: int = 80, interval: BarInterval = BarInterval.DAY_1, *, rising: bool = True
) -> PriceHistory:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    step = timedelta(minutes=interval.minutes)
    closes = [
        100.0 + (0.3 * i if rising else -0.3 * i) + (i % 7) for i in range(n)
    ]  # trending with pullbacks
    bars = tuple(
        Bar(
            timestamp=start + step * i,
            open=close - 0.5,
            high=close + 1.0,
            low=close - 1.5,
            close=close,
            volume=1_000.0,
        )
        for i, close in enumerate(closes)
    )
    return PriceHistory(symbol="AAPL", interval=interval, bars=bars)


def make_assessment(*, source: str = "1d:sma_trend") -> TechnicalAssessment:
    return TechnicalAssessment(
        direction=SignalDirection.BULLISH,
        reasoning="Trend indicators align to the upside.",
        confidence=0.7,
        evidence=(Evidence(source=source, statement="Close is above the SMA(50)."),),
        invalidation_conditions=("A daily close below the SMA(50).",),
        timeframe_alignment="Both timeframes point up.",
    )


def make_agent(response: BaseModel | None = None) -> tuple[TechnicalAnalysisAgent, FakeLLM]:
    llm = FakeLLM(response or make_assessment())
    return TechnicalAnalysisAgent(llm, PandasIndicatorEngine()), llm


async def analyze(histories: Sequence[PriceHistory]) -> tuple[object, FakeLLM]:
    agent, llm = make_agent()
    return await agent.analyze(histories), llm


# --- Single timeframe -------------------------------------------------------


async def test_report_combines_deterministic_and_llm_layers() -> None:
    agent, _ = make_agent()
    history = make_history()

    report = await agent.analyze([history])

    assert report.symbol == "AAPL"
    assert report.latest_close == history.bars[-1].close
    assert report.as_of == history.bars[-1].timestamp
    assert len(report.timeframes) == 1
    assert len(report.primary.readings) == 9
    assert report.assessment.direction is SignalDirection.BULLISH
    assert sum(report.primary.signal_counts.values()) == len(report.primary.readings)


async def test_llm_receives_readings_not_raw_bars() -> None:
    agent, llm = make_agent()

    await agent.analyze([make_history()])

    assert llm.prompt is not None
    payload = json.loads(llm.prompt)
    assert payload["symbol"] == "AAPL"
    timeframe = payload["timeframes"][0]
    assert timeframe["interval"] == "1d"
    names = {reading["name"] for reading in timeframe["readings"]}
    assert "rsi_14" in names
    assert "bars" not in payload  # the LLM never sees raw prices
    assert llm.system is not None
    assert "never invent" in llm.system.lower()


async def test_empty_history_is_rejected() -> None:
    agent, _ = make_agent()
    with pytest.raises(InsufficientHistoryError):
        await agent.analyze([PriceHistory(symbol="AAPL", interval=BarInterval.DAY_1)])


async def test_no_timeframes_is_rejected() -> None:
    agent, _ = make_agent()
    with pytest.raises(InsufficientHistoryError, match="no timeframes"):
        await agent.analyze([])


# --- Multi-timeframe --------------------------------------------------------


async def test_every_timeframe_is_analyzed_and_labelled() -> None:
    agent, _ = make_agent()
    histories = [
        make_history(interval=BarInterval.DAY_1),
        make_history(interval=BarInterval.HOUR_4),
        make_history(interval=BarInterval.HOUR_1),
    ]

    report = await agent.analyze(histories)

    assert report.intervals == (BarInterval.DAY_1, BarInterval.HOUR_4, BarInterval.HOUR_1)
    assert all(frame.readings for frame in report.timeframes)
    # The primary timeframe is the highest one, and drives the convenience fields.
    assert report.primary.interval is BarInterval.DAY_1
    assert report.as_of == report.primary.as_of


async def test_prompt_carries_every_timeframe_in_order() -> None:
    agent, llm = make_agent()

    await agent.analyze(
        [make_history(interval=BarInterval.DAY_1), make_history(interval=BarInterval.HOUR_1)]
    )

    assert llm.prompt is not None
    payload = json.loads(llm.prompt)
    assert [frame["interval"] for frame in payload["timeframes"]] == ["1d", "1h"]
    assert all("deterministic_direction" in frame for frame in payload["timeframes"])
    assert llm.system is not None
    assert "highest to lowest" in llm.system


async def test_evidence_sources_are_namespaced_by_timeframe() -> None:
    """A claim has to say which timeframe it came from, or it is not checkable."""
    agent, llm = make_agent()

    await agent.analyze([make_history()])

    assert llm.system is not None
    assert '"<interval>:<reading name>"' in llm.system


async def test_conflicting_timeframes_are_reported_as_unaligned() -> None:
    agent, _ = make_agent()

    report = await agent.analyze(
        [
            make_history(interval=BarInterval.DAY_1, rising=True),
            make_history(interval=BarInterval.HOUR_1, rising=False),
        ]
    )

    directions = {frame.direction for frame in report.timeframes}
    assert len(directions) > 1
    assert report.aligned is False


async def test_agreeing_timeframes_are_reported_as_aligned() -> None:
    agent, _ = make_agent()

    report = await agent.analyze(
        [
            make_history(interval=BarInterval.DAY_1, rising=True),
            make_history(interval=BarInterval.HOUR_1, rising=True),
        ]
    )

    assert report.aligned is True


# --- Deterministic direction ------------------------------------------------


def test_net_direction_takes_the_majority() -> None:
    bullish, bearish = SignalDirection.BULLISH, SignalDirection.BEARISH
    assert net_direction([bullish, bullish, bearish]) is bullish
    assert net_direction([bearish, bearish, bullish]) is bearish


def test_net_direction_breaks_ties_to_neutral() -> None:
    """A genuine split is not a signal, and rounding it into one would be a lie."""
    assert (
        net_direction([SignalDirection.BULLISH, SignalDirection.BEARISH]) is SignalDirection.NEUTRAL
    )


def test_net_direction_ignores_neutral_readings() -> None:
    assert (
        net_direction([SignalDirection.NEUTRAL, SignalDirection.NEUTRAL]) is SignalDirection.NEUTRAL
    )
    assert (
        net_direction([SignalDirection.NEUTRAL, SignalDirection.BULLISH]) is SignalDirection.BULLISH
    )


# --- Envelope ---------------------------------------------------------------


async def test_envelope_constraints_are_enforced_by_schema() -> None:
    with pytest.raises(ValueError, match="evidence"):
        TechnicalAssessment(
            direction=SignalDirection.BULLISH,
            reasoning="No evidence given.",
            confidence=0.9,
            evidence=(),
            invalidation_conditions=("Anything.",),
            timeframe_alignment="n/a",
        )
    with pytest.raises(ValueError, match="confidence"):
        TechnicalAssessment(
            direction=SignalDirection.BULLISH,
            reasoning="Overconfident.",
            confidence=1.5,
            evidence=(Evidence(source="rsi_14", statement="x"),),
            invalidation_conditions=("Anything.",),
            timeframe_alignment="n/a",
        )


async def test_timeframe_alignment_is_required() -> None:
    """MTF reasoning is the point; an assessment that skips it is incomplete."""
    with pytest.raises(ValueError, match="timeframe_alignment"):
        TechnicalAssessment(  # type: ignore[call-arg]
            direction=SignalDirection.BULLISH,
            reasoning="Fine.",
            confidence=0.5,
            evidence=(Evidence(source="rsi_14", statement="x"),),
            invalidation_conditions=("Anything.",),
        )
