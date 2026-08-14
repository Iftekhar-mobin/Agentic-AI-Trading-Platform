"""Tests for the consensus policy — the arithmetic that decides trades."""

from __future__ import annotations

from typing import Any

import pytest

from atp.domain.models.analysis import SignalDirection
from atp.domain.models.trading import OrderSide
from atp.domain.models.voting import (
    BotStatus,
    ConsensusAction,
    Vote,
    VoterKind,
    VotingPolicy,
)
from atp.domain.services.voting import tally

BULLISH = SignalDirection.BULLISH
BEARISH = SignalDirection.BEARISH
NEUTRAL = SignalDirection.NEUTRAL


def vote(
    direction: SignalDirection,
    confidence: float,
    *,
    name: str = "agent",
    kind: VoterKind = VoterKind.AGENT,
) -> Vote:
    return Vote(voter=name, kind=kind, direction=direction, confidence=confidence)


def policy(**overrides: Any) -> VotingPolicy:
    return VotingPolicy(**{"threshold": 0.35, "min_voters": 3, **overrides})


def test_unanimous_confident_bulls_buy() -> None:
    votes = [vote(BULLISH, 0.9, name=f"a{i}") for i in range(3)]
    decision = tally("XAUUSD", votes, policy())

    assert decision.action is ConsensusAction.BUY
    assert decision.score == pytest.approx(1.0)
    assert decision.agreement == pytest.approx(1.0)
    assert decision.action.side is OrderSide.BUY


def test_unanimous_bears_sell() -> None:
    votes = [vote(BEARISH, 0.8, name=f"a{i}") for i in range(3)]
    decision = tally("XAUUSD", votes, policy())

    assert decision.action is ConsensusAction.SELL
    assert decision.score == pytest.approx(-1.0)
    assert decision.action.side is OrderSide.SELL


def test_a_clear_majority_with_one_dissenter_trades() -> None:
    """Three confident bulls against one moderate bear clears 0.35 comfortably.

    Pinned because it is the intuition people get wrong: dividing by *total
    confidence* rather than voter count means a lone dissenter dilutes far less
    than a head-count suggests.
    """
    votes = [
        vote(BULLISH, 0.8, name="a"),
        vote(BULLISH, 0.65, name="b"),
        vote(BULLISH, 0.7, name="c"),
        vote(BEARISH, 0.6, name="d"),
    ]
    decision = tally("XAUUSD", votes, policy())

    assert decision.score == pytest.approx(1.55 / 2.75, abs=1e-4)  # 0.5636
    assert decision.action is ConsensusAction.BUY
    assert decision.agreement == pytest.approx(0.75)


def test_a_marginal_lean_does_not_open_a_position() -> None:
    """A genuine majority that still should not trade: conviction is too thin."""
    votes = [
        vote(BULLISH, 0.5, name="a"),
        vote(BULLISH, 0.5, name="b"),
        vote(BEARISH, 0.4, name="c"),
        vote(NEUTRAL, 1.0, name="d"),
    ]
    decision = tally("XAUUSD", votes, policy())

    assert decision.score == pytest.approx(0.6 / 2.4, abs=1e-4)  # 0.25
    assert decision.score < 0.35
    assert decision.action is ConsensusAction.HOLD
    assert "not enough" in decision.reason


def test_one_certain_voice_outweighs_several_hesitant_ones() -> None:
    votes = [
        vote(BULLISH, 0.2, name="a"),
        vote(BULLISH, 0.2, name="b"),
        vote(BEARISH, 1.0, name="c"),
    ]
    decision = tally("XAUUSD", votes, policy())

    assert decision.action is ConsensusAction.SELL
    # Head-count says bullish 2-1; weighted conviction says otherwise.
    assert decision.tally["bullish"] == 2
    assert decision.score < 0


def test_confident_neutrals_dilute_rather_than_abstain() -> None:
    """A confident 'nothing here' must be able to block a thin majority."""
    with_neutrals = tally(
        "XAUUSD",
        [
            vote(BULLISH, 0.9, name="a"),
            vote(NEUTRAL, 1.0, name="b"),
            vote(NEUTRAL, 1.0, name="c"),
        ],
        policy(),
    )
    assert with_neutrals.score == pytest.approx(0.9 / 2.9, abs=1e-4)
    assert with_neutrals.action is ConsensusAction.HOLD


def test_quorum_blocks_a_thin_pool() -> None:
    """Two certain bulls still cannot act when three voters are required."""
    votes = [vote(BULLISH, 1.0, name="a"), vote(BULLISH, 1.0, name="b")]
    decision = tally("XAUUSD", votes, policy(min_voters=3))

    assert decision.action is ConsensusAction.HOLD
    assert not decision.quorum_met
    assert "Quorum not met" in decision.reason
    # The score is still reported, so the near-miss is visible.
    assert decision.score == pytest.approx(1.0)


def test_no_votes_holds_rather_than_crashing() -> None:
    decision = tally("XAUUSD", [], policy())

    assert decision.action is ConsensusAction.HOLD
    assert decision.score == 0.0
    assert decision.action.side is None


def test_all_zero_confidence_is_blank_not_bearish() -> None:
    """Dividing by a zero weight sum must not produce a direction."""
    votes = [vote(BULLISH, 0.0, name=f"a{i}") for i in range(3)]
    decision = tally("XAUUSD", votes, policy())

    assert decision.score == 0.0
    assert decision.action is ConsensusAction.HOLD


def test_required_bot_signal_blocks_an_agent_only_pool() -> None:
    votes = [vote(BULLISH, 1.0, name=f"a{i}") for i in range(4)]
    decision = tally("XAUUSD", votes, policy(require_bot_signal=True))

    assert decision.action is ConsensusAction.HOLD
    assert "requires a fresh bot signal" in decision.reason


def test_required_bot_signal_is_satisfied_by_a_bot_vote() -> None:
    votes = [
        vote(BULLISH, 1.0, name="a"),
        vote(BULLISH, 1.0, name="b"),
        vote(BULLISH, 0.9, name="sniper_bot", kind=VoterKind.BOT),
    ]
    decision = tally("XAUUSD", votes, policy(require_bot_signal=True))

    assert decision.action is ConsensusAction.BUY


def test_the_bot_can_be_outvoted_by_the_agents() -> None:
    """The bot is one voice, not a veto — that is the point of voting."""
    votes = [
        vote(BEARISH, 0.9, name="sniper_bot", kind=VoterKind.BOT),
        vote(BULLISH, 0.95, name="a"),
        vote(BULLISH, 0.95, name="b"),
        vote(BULLISH, 0.9, name="c"),
    ]
    decision = tally("XAUUSD", votes, policy())

    assert decision.action is ConsensusAction.BUY


def test_score_stays_within_bounds_for_any_mix() -> None:
    votes = [
        vote(BULLISH, 1.0, name="a"),
        vote(BEARISH, 1.0, name="b"),
        vote(NEUTRAL, 1.0, name="c"),
        vote(BULLISH, 0.3, name="d"),
    ]
    decision = tally("XAUUSD", votes, policy())
    assert -1.0 <= decision.score <= 1.0


def test_every_outcome_explains_itself() -> None:
    """'Why did it not trade' must always be answerable from the decision alone."""
    cases = [
        [],
        [vote(BULLISH, 1.0, name="a")],
        [vote(BULLISH, 1.0, name=f"a{i}") for i in range(3)],
        [vote(BEARISH, 1.0, name=f"a{i}") for i in range(3)],
        [vote(BULLISH, 0.4, name="a"), vote(BEARISH, 0.35, name="b"), vote(NEUTRAL, 0.9, name="c")],
    ]
    for votes in cases:
        decision = tally("XAUUSD", votes, policy())
        assert decision.reason.strip()
        assert decision.reason.endswith(".")


def test_symbol_is_normalized() -> None:
    decision = tally(" xauusd ", [vote(BULLISH, 0.5)], policy())
    assert decision.symbol == "XAUUSD"


@pytest.mark.parametrize(
    ("status", "degraded"),
    [
        (BotStatus.VOTED, False),
        (BotStatus.NOT_CONFIGURED, False),
        (BotStatus.STALE, True),
        (BotStatus.MISSING, True),
    ],
)
def test_degraded_marks_only_an_expected_bot_that_failed_to_take_part(
    status: BotStatus, degraded: bool
) -> None:
    assert status.degraded is degraded
