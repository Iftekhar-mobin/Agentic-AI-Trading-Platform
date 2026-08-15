"""Caller-supplied bars and setup context (``application.request_scope``).

The dangerous failure here is silent: this loader is storage-first, so bars
supplied with a request could be quietly outranked by whatever happens to be in
the database. The analysis would then be of different candles entirely, and
nothing in the output would say so — it would simply look like the agents
disagreeing. Most of these tests exist to pin that ordering down.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TypeVar

import pytest
from pydantic import BaseModel, ValidationError

from atp.application.agents import TechnicalAnalysisAgent
from atp.application.agents.evidence import external_evidence_sources
from atp.application.request_scope import bot_context, supplied_history, supplied_inputs
from atp.application.use_cases import AnalyzeTicker, LoadPriceHistory, LoadTimeframes
from atp.domain.models.analysis import SignalDirection, TechnicalAssessment
from atp.domain.models.explainability import Evidence
from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.infrastructure.indicators import PandasIndicatorEngine
from atp.interfaces.api.routes.consensus import BarInput, ConsensusRequest

T = TypeVar("T", bound=BaseModel)

NOW = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)


def make_history(n: int, *, base: float, interval: BarInterval = BarInterval.DAY_1,
                 symbol: str = "XAUUSD") -> PriceHistory:
    """``base`` separates one series from another so tests can tell them apart."""
    step = timedelta(minutes=interval.minutes)
    start = NOW - step * n
    bars = tuple(
        Bar(
            timestamp=start + step * i,
            open=base + i,
            high=base + i + 1.5,
            low=base + i - 1.5,
            close=base + i + 0.5,
            volume=1_000.0,
        )
        for i in range(n)
    )
    return PriceHistory(symbol=symbol, interval=interval, bars=bars)


class CapturingLLM:
    """Records the prompt so tests can assert what the agent was actually told."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_structured(self, response_model: type[T], *, system: str, prompt: str) -> T:
        self.prompts.append(prompt)
        assessment = TechnicalAssessment(
            direction=SignalDirection.BULLISH,
            reasoning="Trend up.",
            confidence=0.6,
            evidence=(Evidence(source="bot:trend_gate", statement="Gate agrees."),),
            invalidation_conditions=("Close below SMA(50).",),
            timeframe_alignment="One timeframe requested.",
        )
        assert isinstance(assessment, response_model)
        return assessment


class LoudRepository:
    """A repository that would happily serve plenty of (different) bars."""

    def __init__(self, history: PriceHistory) -> None:
        self._history = history
        self.calls = 0

    async def upsert_bars(self, history: PriceHistory) -> int:
        raise NotImplementedError

    async def get_bars(self, symbol: str, interval: BarInterval, **_kwargs: object) -> PriceHistory:
        self.calls += 1
        return self._history

    async def latest_timestamp(self, symbol: str, interval: BarInterval) -> datetime | None:
        return None


class LoudProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def get_bars(self, symbol: str, interval: BarInterval, **_kwargs: object) -> PriceHistory:
        self.calls += 1
        return make_history(80, base=900.0, interval=interval)


def make_use_case(repository: LoudRepository, provider: LoudProvider,
                  llm: CapturingLLM) -> AnalyzeTicker:
    agent = TechnicalAnalysisAgent(llm, PandasIndicatorEngine())
    loader = LoadPriceHistory(repository, provider, clock=lambda: NOW)
    return AnalyzeTicker(LoadTimeframes(loader), agent)


async def test_supplied_bars_beat_a_well_stocked_repository() -> None:
    """The silent-ignore case: storage has plenty, but the caller's bars win."""
    repository = LoudRepository(make_history(120, base=100.0))
    provider = LoudProvider()
    llm = CapturingLLM()
    use_case = make_use_case(repository, provider, llm)

    supplied = make_history(80, base=3_800.0)
    with supplied_inputs(bars={("XAUUSD", BarInterval.DAY_1): supplied}):
        report = await use_case.execute("XAUUSD")

    assert repository.calls == 0, "repository consulted despite supplied bars"
    assert provider.calls == 0, "vendor fetched despite supplied bars"
    # The analysed series is the supplied one, not the stored one.
    assert report.latest_close == pytest.approx(supplied.bars[-1].close)


async def test_without_supplied_bars_the_normal_path_is_unchanged() -> None:
    repository = LoudRepository(make_history(120, base=100.0))
    provider = LoudProvider()
    use_case = make_use_case(repository, provider, CapturingLLM())

    report = await use_case.execute("XAUUSD")

    assert repository.calls == 1
    assert report.latest_close == pytest.approx(219.5)  # stored series, base 100 + 119 + 0.5


async def test_context_reaches_the_agent_prompt() -> None:
    repository = LoudRepository(make_history(120, base=100.0))
    llm = CapturingLLM()
    use_case = make_use_case(repository, LoudProvider(), llm)

    context = {"strategy": "precision_sniper", "trend_gate": {"bias_direction": "bullish"}}
    with supplied_inputs(context=context):
        await use_case.execute("XAUUSD")

    payload = json.loads(llm.prompts[0])
    assert payload["external_context"] == context


async def test_no_context_means_no_external_block() -> None:
    llm = CapturingLLM()
    use_case = make_use_case(LoudRepository(make_history(120, base=100.0)), LoudProvider(), llm)

    await use_case.execute("XAUUSD")

    assert "external_context" not in json.loads(llm.prompts[0])


def test_scope_resets_so_requests_cannot_leak_into_each_other() -> None:
    supplied = make_history(10, base=1_000.0)
    with supplied_inputs(bars={("XAUUSD", BarInterval.DAY_1): supplied}, context={"a": 1}):
        assert supplied_history("xauusd", BarInterval.DAY_1) is not None
        assert bot_context() == {"a": 1}

    assert supplied_history("XAUUSD", BarInterval.DAY_1) is None
    assert bot_context() is None


def test_supplied_lookup_is_per_symbol_and_timeframe() -> None:
    with supplied_inputs(bars={("XAUUSD", BarInterval.DAY_1): make_history(10, base=1_000.0)}):
        assert supplied_history("XAUUSD", BarInterval.HOUR_4) is None
        assert supplied_history("EURUSD", BarInterval.DAY_1) is None


def test_bot_citations_are_registered_as_known_sources() -> None:
    """Otherwise every legitimate ``bot:`` citation logs as fabricated."""
    sources = external_evidence_sources(
        {"strategy": "precision_sniper", "trend_gate": {"bias_direction": "bullish"}}
    )
    assert "bot:strategy" in sources
    assert "bot:trend_gate" in sources
    assert "bot:trend_gate.bias_direction" in sources
    assert external_evidence_sources(None) == set()


def test_naive_bar_timestamps_are_refused() -> None:
    """A bot on broker server time is exactly who sends one of these."""
    with pytest.raises(ValidationError):
        BarInput(timestamp=datetime(2026, 8, 15, 12, 0), open=1, high=2, low=0.5, close=1.5)


def test_request_converts_bars_to_domain_histories() -> None:
    rows = [
        {
            "timestamp": NOW + timedelta(minutes=15 * i),
            "open": 10,
            "high": 12,
            "low": 9,
            "close": 11,
            "volume": 5,
        }
        for i in range(3)
    ]
    request = ConsensusRequest(symbol="xauusd", intervals=["15m"], bars={"15m": rows})

    histories = request.price_histories()

    assert list(histories) == [("XAUUSD", BarInterval.MIN_15)]
    assert len(histories[("XAUUSD", BarInterval.MIN_15)]) == 3


def test_inconsistent_ohlc_is_refused_at_the_boundary() -> None:
    """A bad candle must fail as a 422, not midway through the analysis."""
    with pytest.raises(ValidationError):
        ConsensusRequest(
            symbol="XAUUSD",
            intervals=["15m"],
            bars={"15m": [{"timestamp": NOW, "open": 10, "high": 9, "low": 8, "close": 11}]},
        )
