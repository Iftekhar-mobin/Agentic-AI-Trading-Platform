"""Tests for the consensus use case: agent votes, bot availability, risk veto."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, cast

import pytest
from factories import (
    make_chart_pattern_report,
    make_fundamental_report,
    make_market_research_report,
    make_news_report,
    make_sentiment_report,
    make_technical_report,
)

from atp.application.orchestration.consensus import ReachConsensus
from atp.application.orchestration.graph import TradingOrchestrator
from atp.application.orchestration.state import TradingState
from atp.application.use_cases.check_trade_risk import CheckTradeRisk
from atp.domain.models.analysis import SignalDirection
from atp.domain.models.market import BarInterval
from atp.domain.models.signals import BotSignal
from atp.domain.models.trading import (
    OrderSide,
    RiskDecision,
    RiskLimits,
    RiskVerdict,
    TradeProposal,
)
from atp.domain.models.voting import BotStatus, ConsensusAction, VotingPolicy

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
TTL = timedelta(minutes=15)


class StubOrchestrator:
    """Returns a state whose reports carry the directions the test wants."""

    def __init__(self, directions: dict[str, tuple[SignalDirection, float]]) -> None:
        self._directions = directions
        self.calls: list[str] = []

    async def run(self, symbol: str, intervals: Any = None, *, agents: Any = None) -> Any:
        self.calls.append(symbol)
        return _state(self._directions)


_FACTORIES: dict[str, tuple[str, Any]] = {
    "technical_analysis": ("technical_report", make_technical_report),
    "chart_pattern": ("chart_pattern_report", make_chart_pattern_report),
    "market_research": ("market_research_report", make_market_research_report),
    "fundamental_analysis": ("fundamental_report", make_fundamental_report),
    "news_analysis": ("news_report", make_news_report),
    "sentiment_analysis": ("sentiment_report", make_sentiment_report),
}


def _state(directions: dict[str, tuple[SignalDirection, float]]) -> TradingState:
    """A real TradingState carrying real reports, retuned to the wanted votes."""
    fields: dict[str, Any] = {}
    for agent, (field, factory) in _FACTORIES.items():
        entry = directions.get(agent)
        if entry is None:
            continue
        direction, confidence = entry
        report = factory("XAUUSD")
        fields[field] = report.model_copy(
            update={
                "assessment": report.assessment.model_copy(
                    update={"direction": direction, "confidence": confidence}
                )
            }
        )
    return TradingState(symbol="XAUUSD", **fields)


class StubSignals:
    def __init__(self, stored: BotSignal | None = None) -> None:
        self.stored = stored
        self.appended: list[BotSignal] = []

    async def append(self, signal: BotSignal) -> None:
        self.appended.append(signal)

    async def latest(self, symbol: str) -> BotSignal | None:
        return self.stored

    async def list_signals(self, *, limit: int | None = None) -> list[BotSignal]:
        return [self.stored] if self.stored else []


class StubRiskCheck:
    def __init__(self, verdict: RiskVerdict = RiskVerdict.APPROVED) -> None:
        self.verdict = verdict
        self.calls: list[dict[str, Any]] = []

    async def execute(self, symbol: str, **kwargs: Any) -> RiskDecision:
        self.calls.append({"symbol": symbol, **kwargs})
        return RiskDecision(
            verdict=self.verdict,
            proposal=TradeProposal(
                symbol=symbol,
                side=kwargs.get("side", OrderSide.BUY),
                quantity=Decimal("1"),
                entry_price=Decimal("4400"),
            ),
            limits=RiskLimits(),
            violations=(),
            metrics={},
        )


def build(
    directions: dict[str, tuple[SignalDirection, float]],
    *,
    signals: StubSignals | None = None,
    risk: StubRiskCheck | None = None,
    expected_bot: str | None = None,
    policy: VotingPolicy | None = None,
) -> tuple[ReachConsensus, StubRiskCheck]:
    risk = risk or StubRiskCheck()
    use_case = ReachConsensus(
        cast(TradingOrchestrator, StubOrchestrator(directions)),
        signals or StubSignals(),
        cast(CheckTradeRisk, risk),
        policy or VotingPolicy(threshold=0.35, min_voters=3),
        signal_ttl=TTL,
        expected_bot=expected_bot,
        clock=lambda: NOW,
    )
    return use_case, risk


BULLS = {
    "technical_analysis": (SignalDirection.BULLISH, 0.8),
    "chart_pattern": (SignalDirection.BULLISH, 0.8),
    "news_analysis": (SignalDirection.BULLISH, 0.7),
}


async def test_agents_alone_can_reach_a_verdict() -> None:
    use_case, _ = build(BULLS)
    result = await use_case.execute("XAUUSD", BarInterval.DAY_1)

    assert result.decision.action is ConsensusAction.BUY
    assert len(result.decision.votes) == 3
    assert result.bot_status is BotStatus.NOT_CONFIGURED
    assert "No external bot is configured" in result.bot_note


async def test_a_named_bot_that_never_published_is_reported_missing() -> None:
    """The requirement: say the bot is unavailable, then decide anyway."""
    use_case, _ = build(BULLS, expected_bot="sniper_bot")
    result = await use_case.execute("XAUUSD")

    assert result.bot_status is BotStatus.MISSING
    assert result.bot_status.degraded
    assert "sniper_bot is not available" in result.bot_note
    assert "agents' own analysis alone" in result.bot_note
    # It still decides - absence degrades the pool, it does not block it.
    assert result.decision.action is ConsensusAction.BUY


async def test_a_stale_signal_is_excluded_and_named_as_stale() -> None:
    stale = BotSignal(
        symbol="XAUUSD",
        direction=SignalDirection.BEARISH,
        confidence=0.9,
        source="sniper_bot",
        received_at=NOW - timedelta(minutes=45),
    )
    use_case, _ = build(BULLS, signals=StubSignals(stale), expected_bot="sniper_bot")
    result = await use_case.execute("XAUUSD")

    assert result.bot_status is BotStatus.STALE
    assert result.stale_signal is not None
    assert "45 minutes old" in result.bot_note
    # The stale bearish vote must not have been counted.
    assert all(v.voter != "sniper_bot" for v in result.decision.votes)
    assert result.decision.action is ConsensusAction.BUY


async def test_a_fresh_stored_signal_votes() -> None:
    fresh = BotSignal(
        symbol="XAUUSD",
        direction=SignalDirection.BEARISH,
        confidence=0.9,
        source="sniper_bot",
        received_at=NOW - timedelta(minutes=5),
    )
    use_case, _ = build(BULLS, signals=StubSignals(fresh), expected_bot="sniper_bot")
    result = await use_case.execute("XAUUSD")

    assert result.bot_status is BotStatus.VOTED
    assert "sniper_bot voted bearish" in result.bot_note
    assert any(v.voter == "sniper_bot" for v in result.decision.votes)


async def test_an_inline_signal_beats_the_stored_one() -> None:
    """A bot submitting with its request is current by definition."""
    stale = BotSignal(
        symbol="XAUUSD",
        direction=SignalDirection.BEARISH,
        confidence=0.9,
        source="old",
        received_at=NOW - timedelta(hours=5),
    )
    inline = BotSignal(
        symbol="XAUUSD",
        direction=SignalDirection.BULLISH,
        confidence=0.95,
        source="sniper_bot",
    )
    use_case, _ = build(BULLS, signals=StubSignals(stale))
    result = await use_case.execute("XAUUSD", signal=inline)

    assert result.bot_status is BotStatus.VOTED
    assert result.stale_signal is None
    assert any(v.voter == "sniper_bot" for v in result.decision.votes)


async def test_failed_agents_shrink_the_pool_rather_than_voting_neutral() -> None:
    """A missing report must not drag the score toward zero."""
    use_case, _ = build({"technical_analysis": (SignalDirection.BULLISH, 0.9)})
    result = await use_case.execute("XAUUSD")

    assert len(result.decision.votes) == 1
    assert result.decision.score == pytest.approx(1.0)
    # ...but quorum still blocks acting on one voice.
    assert result.decision.action is ConsensusAction.HOLD
    assert not result.decision.quorum_met


async def test_a_hold_verdict_never_touches_the_risk_gate() -> None:
    use_case, risk = build({"technical_analysis": (SignalDirection.BULLISH, 0.9)})
    result = await use_case.execute("XAUUSD")

    assert result.decision.action is ConsensusAction.HOLD
    assert result.risk is None
    assert risk.calls == []


async def test_the_risk_gate_can_reject_what_the_voters_wanted() -> None:
    """Consensus proposes; risk disposes. A unanimous pool is not permission."""
    use_case, risk = build(BULLS, risk=StubRiskCheck(RiskVerdict.REJECTED))
    result = await use_case.execute("XAUUSD", stop_loss=4150.0)

    assert result.decision.action is ConsensusAction.BUY
    assert result.risk is not None
    assert result.risk.verdict is RiskVerdict.REJECTED
    assert risk.calls[0]["side"] is OrderSide.BUY
    assert risk.calls[0]["stop_loss"] == Decimal("4150.0")


async def test_bears_produce_a_sell_proposal() -> None:
    bears = {
        "technical_analysis": (SignalDirection.BEARISH, 0.8),
        "chart_pattern": (SignalDirection.BEARISH, 0.8),
        "news_analysis": (SignalDirection.BEARISH, 0.7),
    }
    use_case, risk = build(bears)
    result = await use_case.execute("XAUUSD")

    assert result.decision.action is ConsensusAction.SELL
    assert risk.calls[0]["side"] is OrderSide.SELL
