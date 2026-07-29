"""Tests for the learning loop: agent, situation digest, journalling, degradation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TypeVar

from factories import (
    make_episode,
    make_fundamental_report,
    make_news_report,
    make_regime,
    make_sentiment_report,
    make_technical_report,
)
from pydantic import BaseModel

from atp.application.agents import ContinuousLearningAgent
from atp.application.orchestration import TradingState, render_situation
from atp.application.use_cases import JournalTrade, LearnFromContext
from atp.domain.errors import InsufficientHistoryError, RepositoryUnavailableError
from atp.domain.models.explainability import Evidence
from atp.domain.models.market import Bar, BarInterval, PriceHistory
from atp.domain.models.memory import (
    EpisodeKind,
    EpisodeMatch,
    JournalEntry,
    MemoryEpisode,
)
from atp.domain.models.orders import ExecutionResult, Order, OrderStatus
from atp.domain.models.trading import (
    OrderSide,
    RiskDecision,
    RiskLimits,
    RiskVerdict,
    RiskViolation,
    TradeProposal,
)
from atp.infrastructure.embeddings import HashingEmbeddingModel
from atp.infrastructure.memory import JsonEpisodicMemory

T = TypeVar("T", bound=BaseModel)

NOW = datetime(2026, 7, 22, 12, tzinfo=UTC)


class FakeLLM:
    def __init__(self, response: BaseModel) -> None:
        self._response = response
        self.prompt: str | None = None
        self.system: str | None = None

    async def generate_structured(self, response_model: type[T], *, system: str, prompt: str) -> T:
        self.system = system
        self.prompt = prompt
        assert isinstance(self._response, response_model)
        return self._response


class StubHistory:
    def __init__(self, *, bars: int = 150, error: Exception | None = None) -> None:
        self._bars = bars
        self._error = error

    async def execute(
        self,
        symbol: str,
        interval: BarInterval = BarInterval.DAY_1,
        *,
        min_bars: int,
        lookback: timedelta | None = None,
    ) -> PriceHistory:
        if self._error is not None:
            raise self._error
        start = NOW - timedelta(days=self._bars)
        bars = tuple(
            Bar(
                timestamp=start + timedelta(days=index),
                open=(close := 100.0 * 1.004**index),
                high=close * 1.01,
                low=close * 0.99,
                close=close,
                volume=1_000.0,
            )
            for index in range(self._bars)
        )
        return PriceHistory(symbol=symbol, interval=interval, bars=bars)


class StubMemory:
    def __init__(
        self,
        *,
        matches: Sequence[EpisodeMatch] = (),
        recall_error: Exception | None = None,
        remember_error: Exception | None = None,
    ) -> None:
        self._matches = tuple(matches)
        self._recall_error = recall_error
        self._remember_error = remember_error
        self.remembered: list[MemoryEpisode] = []
        self.queries: list[str] = []

    async def remember(self, episode: MemoryEpisode) -> None:
        if self._remember_error is not None:
            raise self._remember_error
        self.remembered.append(episode)

    async def recall(
        self,
        query: str,
        *,
        symbol: str | None = None,
        kinds: Sequence[EpisodeKind] | None = None,
        regime_label: str | None = None,
        limit: int = 5,
    ) -> tuple[EpisodeMatch, ...]:
        self.queries.append(query)
        if self._recall_error is not None:
            raise self._recall_error
        return self._matches


def journal_entry(source: str = "episode-0") -> JournalEntry:
    return JournalEntry(
        reasoning="This setup rhymes with a precedent from a calmer tape.",
        confidence=0.4,
        evidence=(Evidence(source=source, statement="Similar bullish setup."),),
        invalidation_conditions=("Volatility shifting to volatile.",),
        lessons=("Entries against the slow-SMA slope gave back gains.",),
        regime_note="Same uptrend, calmer volatility than today.",
    )


def learn(
    llm: FakeLLM,
    *,
    history: StubHistory | None = None,
    memory: StubMemory | None = None,
) -> tuple[LearnFromContext, StubMemory]:
    store = memory or StubMemory()
    use_case = LearnFromContext(
        history or StubHistory(),  # type: ignore[arg-type]
        store,
        ContinuousLearningAgent(llm, clock=lambda: NOW),
    )
    return use_case, store


# --- Situation digest -------------------------------------------------------


def full_state() -> TradingState:
    return TradingState(
        symbol="AAPL",
        interval=BarInterval.DAY_1,
        technical_report=make_technical_report(),
        fundamental_report=make_fundamental_report(),
        news_report=make_news_report(),
        sentiment_report=make_sentiment_report(),
    )


def test_situation_digest_covers_every_report() -> None:
    digest = render_situation(full_state())

    assert digest.startswith("AAPL (1d) analysis.")
    assert "Technical: bullish at confidence 0.70" in digest
    assert "Fundamentals: neutral at confidence 0.55 (Technology)" in digest
    assert "News: bullish" in digest
    assert "Themes: product cycle" in digest
    assert "Sentiment: bullish" in digest
    assert "weighted polarity +0.80" in digest


def test_situation_digest_omits_missing_reports() -> None:
    state = TradingState(symbol="AAPL", technical_report=make_technical_report())
    digest = render_situation(state)

    assert "Technical:" in digest
    assert "Fundamentals:" not in digest
    assert "News:" not in digest
    assert "Sentiment:" not in digest


# --- Continuous learning agent ----------------------------------------------


async def test_agent_receives_regime_recalled_episodes_and_scores() -> None:
    llm = FakeLLM(journal_entry())
    agent = ContinuousLearningAgent(llm, clock=lambda: NOW)
    matches = (EpisodeMatch(episode=make_episode("episode-0"), score=0.62),)

    report = await agent.reflect(
        "aapl", situation="AAPL bullish.", regime=make_regime(), recalled=matches
    )

    assert report.symbol == "AAPL"
    assert report.regime is not None
    assert len(report.recalled) == 1
    assert report.entry.lessons

    assert llm.prompt is not None
    payload = json.loads(llm.prompt)
    assert payload["current_regime"]["label"] == "uptrend/normal"
    assert payload["recalled_episode_count"] == 1
    episode = payload["recalled_episodes"][0]
    assert episode["id"] == "episode-0"
    assert episode["similarity"] == 0.62
    assert episode["age_days"] > 0


async def test_agent_works_with_no_precedent_at_all() -> None:
    llm = FakeLLM(journal_entry(source="current_situation"))
    agent = ContinuousLearningAgent(llm, clock=lambda: NOW)

    report = await agent.reflect("AAPL", situation="AAPL bullish.", regime=None, recalled=())

    assert report.recalled == ()
    assert report.regime is None
    assert llm.prompt is not None
    payload = json.loads(llm.prompt)
    assert payload["recalled_episode_count"] == 0
    assert payload["current_regime"] is None
    assert llm.system is not None
    assert "current_situation" in llm.system


# --- LearnFromContext -------------------------------------------------------


async def test_loop_tags_regime_recalls_and_writes_back() -> None:
    match = EpisodeMatch(episode=make_episode("episode-0"), score=0.5)
    use_case, memory = learn(FakeLLM(journal_entry()), memory=StubMemory(matches=(match,)))

    report = await use_case.execute("AAPL", situation="AAPL technical bullish.")

    assert report.regime is not None
    assert report.regime.trend.value == "uptrend"
    assert memory.queries == ["AAPL technical bullish."]

    kinds = [episode.kind for episode in memory.remembered]
    assert kinds == [EpisodeKind.ANALYSIS, EpisodeKind.LESSON]
    analysis, lesson = memory.remembered
    assert analysis.summary == "AAPL technical bullish."
    assert "Entries against the slow-SMA slope" in lesson.summary
    assert analysis.regime == report.regime


async def test_written_episode_ids_are_deterministic() -> None:
    """Re-running the same moment must overwrite, not duplicate."""
    use_case, memory = learn(FakeLLM(journal_entry()))
    await use_case.execute("AAPL", situation="AAPL bullish.")
    await use_case.execute("AAPL", situation="AAPL bullish.")

    ids = [episode.id for episode in memory.remembered]
    assert ids[0] == ids[2]
    assert ids[1] == ids[3]
    assert ids[0].startswith("analysis:AAPL:")
    assert ids[1].startswith("lesson:AAPL:")


async def test_missing_history_degrades_to_no_regime() -> None:
    use_case, memory = learn(
        FakeLLM(journal_entry()),
        history=StubHistory(error=InsufficientHistoryError("only 3 bars")),
    )

    report = await use_case.execute("AAPL", situation="AAPL bullish.")

    assert report.regime is None
    assert report.entry.lessons  # the reflection still happened
    assert memory.remembered


async def test_unreachable_memory_does_not_break_the_reflection() -> None:
    use_case, _ = learn(
        FakeLLM(journal_entry(source="current_situation")),
        memory=StubMemory(
            recall_error=RepositoryUnavailableError("qdrant down"),
            remember_error=RepositoryUnavailableError("qdrant down"),
        ),
    )

    report = await use_case.execute("AAPL", situation="AAPL bullish.")

    assert report.recalled == ()
    assert report.entry.lessons


async def test_loop_round_trips_through_real_memory(tmp_path: Path) -> None:
    """Two runs: the second must recall what the first wrote."""
    memory = JsonEpisodicMemory(tmp_path / "memory.jsonl", HashingEmbeddingModel(256))
    use_case = LearnFromContext(
        StubHistory(),  # type: ignore[arg-type]
        memory,
        ContinuousLearningAgent(FakeLLM(journal_entry()), clock=lambda: NOW),
    )

    first = await use_case.execute("AAPL", situation="AAPL technical bullish on a crossover.")
    assert first.recalled == ()

    matches = await memory.recall("crossover", symbol="AAPL")
    assert {match.episode.kind for match in matches} == {
        EpisodeKind.ANALYSIS,
        EpisodeKind.LESSON,
    }


# --- JournalTrade -----------------------------------------------------------


def proposal(**overrides: object) -> TradeProposal:
    defaults: dict[str, object] = {
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "quantity": Decimal("10"),
        "entry_price": Decimal("300"),
        "stop_loss": Decimal("290"),
        "strategy_name": "ema_cross",
    }
    return TradeProposal(**{**defaults, **overrides})


def decision(
    verdict: RiskVerdict = RiskVerdict.APPROVED,
    violations: tuple[RiskViolation, ...] = (),
) -> RiskDecision:
    return RiskDecision(
        verdict=verdict,
        proposal=proposal(),
        limits=RiskLimits(),
        violations=violations,
        metrics={"risk_pct": 0.33},
    )


def filled_order() -> Order:
    return Order(
        id="order-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        stop_loss=Decimal("290"),
        status=OrderStatus.FILLED,
        submitted_at=NOW,
        fill_price=Decimal("300.15"),
        filled_at=NOW,
        strategy_name="ema_cross",
    )


async def test_filled_trade_is_journalled_with_the_regime() -> None:
    memory = StubMemory()
    result = ExecutionResult(decision=decision(), order=filled_order())

    episode = await JournalTrade(memory, clock=lambda: NOW).execute(result, regime=make_regime())

    assert episode is not None
    assert episode.id == "trade:AAPL:order-1"
    assert episode.kind is EpisodeKind.TRADE
    assert "Buy 10 AAPL filled at 300.15" in episode.summary
    assert "stop 290" in episode.summary
    assert "uptrend/normal" in episode.summary
    assert episode.metadata["outcome"] == "filled"
    assert episode.metadata["fill_price"] == 300.15
    assert memory.remembered == [episode]


async def test_risk_rejection_is_journalled_too() -> None:
    """ "We wanted this and the gate refused" is a precedent worth recalling."""
    memory = StubMemory()
    result = ExecutionResult(
        decision=decision(
            RiskVerdict.REJECTED,
            (RiskViolation(rule="max_position_pct", detail="would be 45% of equity"),),
        ),
        order=None,
    )

    episode = await JournalTrade(memory, clock=lambda: NOW).execute(result)

    assert episode is not None
    assert episode.kind is EpisodeKind.TRADE
    assert "the risk gate rejected it (max_position_pct)" in episode.summary
    assert episode.metadata["outcome"] == "rejected_by_risk"
    assert episode.occurred_at == NOW


async def test_journalling_failure_never_undoes_a_trade() -> None:
    memory = StubMemory(remember_error=RepositoryUnavailableError("qdrant down"))
    result = ExecutionResult(decision=decision(), order=filled_order())

    episode = await JournalTrade(memory, clock=lambda: NOW).execute(result)

    assert episode is None
    assert memory.remembered == []
