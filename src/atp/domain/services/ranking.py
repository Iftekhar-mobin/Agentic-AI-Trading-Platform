"""Scoring a consensus outcome so several can be compared.

Same principle as ``services.voting``: the arithmetic that orders opportunities
lives in one tested place, not in a prompt. Two screens over the same verdicts
must produce the same leaderboard, and a trader must be able to re-derive any
row's number with a calculator.

The composite is a weighted mean of four readings, then reduced by two gates::

    base    = 0.40*conviction + 0.25*agreement + 0.20*confidence + 0.15*coverage
    composite = base * quorum_penalty * risk_penalty

Why these four:

- **conviction** (|score|) is the headline. A pool at +0.8 is saying something a
  pool at +0.36 is not.
- **agreement** separates "six voters leaning" from "four certain and two
  opposed", which conviction alone cannot: both can land on the same score.
- **confidence** is the voters' own calibration. Weighted third because it is
  already inside the score - it earns a place because a unanimous but hesitant
  pool should not outrank a unanimous confident one.
- **coverage** is the honest one. When four of six agents could not analyse an
  instrument, the remaining two may be certain and unanimous and it is still
  thinner evidence than a full pool at the same numbers.

The two gates are multiplicative rather than subtractive so they cannot be
outvoted by a strong base: no amount of conviction turns a trade the risk engine
rejected into an opportunity, and a pool below quorum is a partial reading
whatever it says.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from atp.domain.models.analysis import SignalDirection
from atp.domain.models.ranking import ScoreComponents
from atp.domain.models.trading import RiskDecision, RiskVerdict
from atp.domain.models.voting import ConsensusDecision, Vote

WEIGHTS: Final[dict[str, float]] = {
    "conviction": 0.40,
    "agreement": 0.25,
    "confidence": 0.20,
    "coverage": 0.15,
}
"""Sums to 1.0, so the base is directly readable as a 0-1 reading."""

NO_QUORUM_PENALTY: Final[float] = 0.5
"""Halved, not zeroed. A sub-quorum reading is still information worth seeing
below the fold - it just must not lead a leaderboard."""

RISK_REJECTED_PENALTY: Final[float] = 0.4
"""Heavier than the quorum gate: the risk engine has already said no."""

HOLD_PENALTY: Final[float] = 0.7
"""A hold has no position to rank. Its conviction is low by construction, so
this rarely decides anything - it exists so a sub-quorum hold with a strong raw
score cannot climb past an actionable buy."""


def score(
    decision: ConsensusDecision,
    *,
    risk: RiskDecision | None = None,
    expected_voters: int = 0,
) -> ScoreComponents:
    """The composite for one consensus outcome, with its parts kept.

    ``expected_voters`` is how many voters the run asked for - the selected
    agents, plus one if a bot was expected. Pass 0 and coverage is treated as
    complete, which is the right default when the caller has no basis to say
    otherwise.
    """
    conviction = min(abs(decision.score), 1.0)
    agreement = decision.agreement
    confidence = _mean_confidence(decision.votes)
    coverage = _coverage(len(decision.votes), expected_voters)

    base = (
        WEIGHTS["conviction"] * conviction
        + WEIGHTS["agreement"] * agreement
        + WEIGHTS["confidence"] * confidence
        + WEIGHTS["coverage"] * coverage
    )

    quorum_penalty = 1.0 if decision.quorum_met else NO_QUORUM_PENALTY
    risk_penalty = _risk_penalty(decision, risk)

    return ScoreComponents(
        conviction=round(conviction, 4),
        agreement=round(agreement, 4),
        confidence=round(confidence, 4),
        coverage=round(coverage, 4),
        quorum_penalty=quorum_penalty,
        risk_penalty=risk_penalty,
        composite=round(min(base * quorum_penalty * risk_penalty, 1.0), 4),
    )


def _mean_confidence(votes: Sequence[Vote]) -> float:
    """Mean confidence of the voters who committed to a direction.

    Neutrals are excluded: a confident "nothing here" is a real vote, and the
    tally already lets it drag the score toward zero. Counting it again here
    would let an abstention *raise* an opportunity's rank.
    """
    directional = [
        vote.confidence for vote in votes if vote.direction is not SignalDirection.NEUTRAL
    ]
    if not directional:
        return 0.0
    return sum(directional) / len(directional)


def _coverage(voters: int, expected: int) -> float:
    if expected <= 0:
        return 1.0
    return min(voters / expected, 1.0)


def _risk_penalty(decision: ConsensusDecision, risk: RiskDecision | None) -> float:
    if risk is not None and risk.verdict is RiskVerdict.REJECTED:
        return RISK_REJECTED_PENALTY
    if decision.action.side is None:
        # Hold: nothing was proposed, so the gate was never consulted.
        return HOLD_PENALTY
    return 1.0
