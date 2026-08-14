"""Counting the votes.

Deliberately boring arithmetic, and deliberately the only place it happens. A
consensus rule that lived in a prompt could be argued with; this one cannot, and
the same votes always produce the same verdict.

The score is a confidence-weighted mean of signed directions::

    score = SUM(direction_i * confidence_i) / SUM(confidence_i)

Dividing by total confidence rather than the number of voters is what makes the
number comparable between runs: it always lands in [-1, 1], where +1 means
"everyone who spoke was certain and bullish", regardless of how many spoke.

Two gates stand between a score and a trade — a quorum, so a lone voice cannot
act while the rest of the pool is unavailable, and a threshold, so a marginal
lean does not become a position. Both fail closed, to hold.
"""

from __future__ import annotations

from collections.abc import Sequence

from atp.domain.models.analysis import SignalDirection
from atp.domain.models.voting import (
    ConsensusAction,
    ConsensusDecision,
    Vote,
    VoterKind,
    VotingPolicy,
)


def tally(symbol: str, votes: Sequence[Vote], policy: VotingPolicy) -> ConsensusDecision:
    """Combine votes into a decision, with the reasoning spelled out."""
    votes = tuple(votes)
    score = _score(votes)
    quorum_met = len(votes) >= policy.min_voters
    has_bot = any(vote.kind is VoterKind.BOT for vote in votes)

    action, reason = _verdict(votes, score, policy, quorum_met=quorum_met, has_bot=has_bot)
    return ConsensusDecision(
        symbol=symbol,
        action=action,
        score=round(score, 4),
        votes=votes,
        policy=policy,
        quorum_met=quorum_met,
        reason=reason,
    )


def _score(votes: Sequence[Vote]) -> float:
    """Confidence-weighted mean direction, or 0.0 when nobody committed."""
    total_confidence = sum(vote.confidence for vote in votes)
    if total_confidence <= 0:
        # Everyone abstained or answered with zero confidence. Not bearish - blank.
        return 0.0
    return sum(vote.weight for vote in votes) / total_confidence


def _verdict(
    votes: Sequence[Vote],
    score: float,
    policy: VotingPolicy,
    *,
    quorum_met: bool,
    has_bot: bool,
) -> tuple[ConsensusAction, str]:
    """The action, and a sentence explaining it.

    Every branch returns a reason. "Why did it not trade?" is the question
    actually asked of this system, and a bare HOLD does not answer it.
    """
    if not votes:
        return ConsensusAction.HOLD, "No votes were cast, so there is nothing to act on."

    if not quorum_met:
        return ConsensusAction.HOLD, (
            f"Quorum not met: {len(votes)} of {policy.min_voters} required voters took part. "
            "Holding rather than acting on a partial pool."
        )

    if policy.require_bot_signal and not has_bot:
        return ConsensusAction.HOLD, (
            "Policy requires a fresh bot signal and none was available, so the agent "
            "pool alone cannot open a position."
        )

    counts = _counts(votes)
    detail = (
        f"score {score:+.2f} against a ±{policy.threshold:.2f} threshold "
        f"({counts[SignalDirection.BULLISH]} bullish, {counts[SignalDirection.BEARISH]} bearish, "
        f"{counts[SignalDirection.NEUTRAL]} neutral across {len(votes)} voters)"
    )

    if score >= policy.threshold:
        return ConsensusAction.BUY, f"Buy: {detail}."
    if score <= -policy.threshold:
        return ConsensusAction.SELL, f"Sell: {detail}."

    return ConsensusAction.HOLD, (
        f"Hold: {detail}. The pool leans "
        f"{'bullish' if score > 0 else 'bearish' if score < 0 else 'flat'} but not enough "
        "to justify a position."
    )


def _counts(votes: Sequence[Vote]) -> dict[SignalDirection, int]:
    counts = dict.fromkeys(SignalDirection, 0)
    for vote in votes:
        counts[vote.direction] += 1
    return counts
