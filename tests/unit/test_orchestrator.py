"""Tests for the supervisor graph, with every analysis use case stubbed."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import cast

import pytest
from factories import (
    make_fundamental_report,
    make_learning_report,
    make_news_report,
    make_sentiment_report,
    make_technical_report,
)
from pydantic import BaseModel

from atp.application.orchestration import (
    ALL_AGENTS,
    ANALYSIS_AGENTS,
    TradingOrchestrator,
    UnknownAgentError,
    resolve_agents,
)
from atp.application.use_cases import (
    AnalyzeFundamentals,
    AnalyzeNews,
    AnalyzeSentiment,
    AnalyzeTicker,
    LearnFromContext,
)
from atp.domain.errors import InsufficientDataError, InsufficientHistoryError
from atp.domain.models.analysis import SignalDirection
from atp.domain.models.market import BarInterval


class StubUseCase:
    """Records its calls and either returns a report or raises."""

    def __init__(
        self,
        report_factory: Callable[[str], BaseModel],
        *,
        error: Exception | None = None,
    ) -> None:
        self._report_factory = report_factory
        self._error = error
        self.calls: list[str] = []
        self.started = asyncio.Event()
        self.release: asyncio.Event | None = None

    async def execute(self, symbol: str, interval: BarInterval = BarInterval.DAY_1) -> BaseModel:
        self.calls.append(symbol)
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self._error is not None:
            raise self._error
        return self._report_factory(symbol)


class StubLearning:
    """The feedback-pool stub; records the situation digest it was handed."""

    def __init__(self, *, error: Exception | None = None) -> None:
        self._error = error
        self.situations: list[str] = []

    async def execute(
        self,
        symbol: str,
        interval: BarInterval = BarInterval.DAY_1,
        *,
        situation: str,
    ) -> BaseModel:
        self.situations.append(situation)
        if self._error is not None:
            raise self._error
        return make_learning_report(symbol)


class Stubs:
    """One stub per analysis agent, wired into a real orchestrator."""

    def __init__(self, **errors: Exception) -> None:
        self.technical = StubUseCase(make_technical_report, error=errors.get("technical"))
        self.fundamental = StubUseCase(make_fundamental_report, error=errors.get("fundamental"))
        self.news = StubUseCase(make_news_report, error=errors.get("news"))
        self.sentiment = StubUseCase(make_sentiment_report, error=errors.get("sentiment"))
        self.learning = StubLearning(error=errors.get("learning"))

    def orchestrator(self) -> TradingOrchestrator:
        return TradingOrchestrator(
            cast(AnalyzeTicker, self.technical),
            cast(AnalyzeFundamentals, self.fundamental),
            cast(AnalyzeNews, self.news),
            cast(AnalyzeSentiment, self.sentiment),
            cast(LearnFromContext, self.learning),
        )


async def test_all_agents_run_and_produce_reports() -> None:
    stubs = Stubs()
    state = await stubs.orchestrator().run(" aapl ")

    assert state.symbol == "AAPL"
    assert sorted(state.completed) == sorted(ALL_AGENTS)
    assert state.failures == []
    assert state.technical_report is not None
    assert state.fundamental_report is not None
    assert state.news_report is not None
    assert state.sentiment_report is not None
    assert state.learning_report is not None
    assert state.technical_report.assessment.direction is SignalDirection.BULLISH
    assert stubs.technical.calls == ["AAPL"]


async def test_learning_runs_after_analysis_and_sees_its_output() -> None:
    """Phase two must observe phase one's artifacts, not an empty state."""
    stubs = Stubs()
    state = await stubs.orchestrator().run("AAPL")

    assert state.completed[-1] == "continuous_learning"
    assert len(stubs.learning.situations) == 1
    digest = stubs.learning.situations[0]
    assert "Technical: bullish" in digest
    assert "Fundamentals: neutral" in digest
    assert "News: bullish" in digest
    assert "Sentiment: bullish" in digest


async def test_learning_is_skipped_when_no_analysis_produced_anything() -> None:
    stubs = Stubs(
        technical=InsufficientHistoryError("no bars"),
        fundamental=InsufficientDataError("no metrics"),
        news=InsufficientDataError("no articles"),
        sentiment=InsufficientDataError("no articles"),
    )
    state = await stubs.orchestrator().run("AAPL")

    assert stubs.learning.situations == []
    assert "continuous_learning" not in state.completed
    assert state.learning_report is None


async def test_learning_failure_does_not_discard_the_analysis() -> None:
    stubs = Stubs(learning=InsufficientDataError("memory is down"))
    state = await stubs.orchestrator().run("AAPL")

    assert state.learning_report is None
    assert state.technical_report is not None
    assert [failure.agent for failure in state.failures] == ["continuous_learning"]


async def test_each_agent_runs_exactly_once() -> None:
    stubs = Stubs()
    await stubs.orchestrator().run("AAPL")

    for stub in (stubs.technical, stubs.fundamental, stubs.news, stubs.sentiment):
        assert len(stub.calls) == 1
    assert len(stubs.learning.situations) == 1


async def test_agents_are_dispatched_concurrently() -> None:
    """Every agent starts before any is allowed to finish — i.e. one superstep."""
    stubs = Stubs()
    release = asyncio.Event()
    for stub in (stubs.technical, stubs.fundamental, stubs.news, stubs.sentiment):
        stub.release = release

    run = asyncio.create_task(stubs.orchestrator().run("AAPL"))
    await asyncio.wait_for(
        asyncio.gather(
            *(
                stub.started.wait()
                for stub in (stubs.technical, stubs.fundamental, stubs.news, stubs.sentiment)
            )
        ),
        timeout=5,
    )
    release.set()
    state = await asyncio.wait_for(run, timeout=5)

    assert sorted(state.completed) == sorted(ALL_AGENTS)


async def test_agent_selection_runs_only_the_requested_agents() -> None:
    stubs = Stubs()
    state = await stubs.orchestrator().run("AAPL", agents=["news_analysis"])

    assert state.completed == ["news_analysis"]
    assert state.news_report is not None
    assert state.technical_report is None
    assert stubs.technical.calls == []
    assert stubs.news.calls == ["AAPL"]


async def test_unknown_agent_is_rejected() -> None:
    with pytest.raises(UnknownAgentError, match="astrology"):
        await Stubs().orchestrator().run("AAPL", agents=["astrology"])


async def test_one_failure_does_not_cost_the_other_reports() -> None:
    stubs = Stubs(technical=InsufficientHistoryError("only 3 bars"))
    state = await stubs.orchestrator().run("AAPL")

    assert state.technical_report is None
    assert len(state.failures) == 1
    assert state.failures[0].agent == "technical_analysis"
    assert "3 bars" in state.failures[0].error
    # The failed agent is marked completed so the supervisor never retries it in-run.
    assert sorted(state.completed) == sorted(ALL_AGENTS)
    assert state.has_report
    assert state.news_report is not None


async def test_total_failure_leaves_no_report() -> None:
    stubs = Stubs(
        technical=InsufficientHistoryError("no bars"),
        fundamental=InsufficientDataError("no metrics"),
        news=InsufficientDataError("no articles"),
        sentiment=InsufficientDataError("no articles"),
    )
    state = await stubs.orchestrator().run("AAPL")

    assert not state.has_report
    assert len(state.failures) == len(ANALYSIS_AGENTS)


def test_resolve_agents_normalizes_and_orders() -> None:
    assert resolve_agents(None) == ALL_AGENTS
    assert resolve_agents([]) == ALL_AGENTS
    # Canonical dispatch order wins over the order the caller listed them in.
    assert resolve_agents(["news_analysis", " TECHNICAL_ANALYSIS "]) == (
        "technical_analysis",
        "news_analysis",
    )
    with pytest.raises(UnknownAgentError):
        resolve_agents(["nope"])
