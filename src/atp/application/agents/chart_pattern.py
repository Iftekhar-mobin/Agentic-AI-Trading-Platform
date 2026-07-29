"""Chart Pattern agent (multi-timeframe).

The deterministic layer here is unusually important. Pattern recognition is
where an LLM is most tempted to confabulate — ask one to find formations in a
price series and it will always find several — so the shapes are detected in
code, with fixed proportional rules, and the model receives a finished list it
cannot add to.

What the model does add is the judgment the geometry cannot: whether a textbook
double top on the hourly matters when the daily is in a clean uptrend, and
which levels are the ones actually worth watching.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence

import structlog

from atp.application.agents.evidence import warn_on_unknown_sources
from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.analysis import SignalDirection
from atp.domain.models.market import PriceHistory
from atp.domain.models.patterns import (
    ChartPatternReport,
    PatternAssessment,
    TimeframePatterns,
)
from atp.domain.ports.llm import LLMClient
from atp.domain.services import chart_patterns

log = structlog.get_logger()

AGENT_NAME = "chart_pattern"

SYSTEM_PROMPT = """\
You are the Chart Pattern Agent of an AI trading platform. Traders use your \
output as decision support, never as guaranteed predictions.

You receive chart formations that were detected deterministically for one \
symbol across one or more timeframes, ordered highest to lowest, together with \
the support and resistance levels those timeframes have established.

Rules:
- The detected patterns are the only patterns that exist. Do not claim a \
formation that is not in the list, and do not reinterpret one as another.
- Every evidence item's "source" must be "<interval>:<pattern kind>" or \
"<interval>:level_<price>", matching something you were given.
- "confirmed" is the distinction that matters. A confirmed pattern has \
completed (price closed through its neckline or range boundary); an \
unconfirmed one is still forming and may never complete. Never present the \
second as though it were the first.
- "quality" says how cleanly the shape fits its ideal. Low-quality patterns \
deserve hedged language and lower confidence.
- Read top down: a pattern on the highest timeframe outweighs a contrary one \
below it. timeframe_alignment must say whether they agree and which you are \
weighting.
- Levels are more actionable than pattern names. Where possible, express the \
conclusion in terms of the specific prices provided.
- Confidence is calibrated 0-1. No patterns, only unconfirmed ones, or \
conflicting timeframes must keep it below 0.5.
- Invalidation conditions must be concrete and observable, using the provided \
price levels.
"""


class ChartPatternAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def analyze(self, histories: Sequence[PriceHistory]) -> ChartPatternReport:
        if not histories:
            msg = "no timeframes supplied for chart pattern analysis"
            raise InsufficientHistoryError(msg)

        symbol = histories[0].symbol
        timeframes = tuple(self._scan_timeframe(history) for history in histories)

        payload = {
            "symbol": symbol,
            "timeframes": [
                {
                    "interval": timeframe.interval.value,
                    "as_of": timeframe.as_of.isoformat(),
                    "latest_close": timeframe.latest_close,
                    "deterministic_direction": timeframe.direction.value,
                    "swing_points": len(timeframe.swings),
                    "levels": [level.model_dump(mode="json") for level in timeframe.levels],
                    "patterns": [pattern.model_dump(mode="json") for pattern in timeframe.patterns],
                }
                for timeframe in timeframes
            ],
        }
        assessment = await self._llm.generate_structured(
            PatternAssessment,
            system=SYSTEM_PROMPT,
            prompt=json.dumps(payload, sort_keys=True),
        )

        known = {
            f"{timeframe.interval.value}:{pattern.kind.value}"
            for timeframe in timeframes
            for pattern in timeframe.patterns
        } | {
            f"{timeframe.interval.value}:level_{level.price}"
            for timeframe in timeframes
            for level in timeframe.levels
        }
        warn_on_unknown_sources(AGENT_NAME, symbol, assessment.evidence, known)

        report = ChartPatternReport(symbol=symbol, timeframes=timeframes, assessment=assessment)
        log.info(
            "chart_pattern.completed",
            symbol=symbol,
            intervals=[interval.value for interval in report.intervals],
            patterns=sum(len(timeframe.patterns) for timeframe in timeframes),
            confirmed=sum(
                1 for timeframe in timeframes for pattern in timeframe.patterns if pattern.confirmed
            ),
            direction=assessment.direction.value,
            confidence=assessment.confidence,
        )
        return report

    @staticmethod
    def _scan_timeframe(history: PriceHistory) -> TimeframePatterns:
        latest = history.latest
        if latest is None:
            msg = f"{history.symbol}/{history.interval.value}: no bars available to scan"
            raise InsufficientHistoryError(msg)

        swings, levels, patterns = chart_patterns.scan(history)
        counts = Counter(pattern.direction for pattern in patterns)
        return TimeframePatterns(
            interval=history.interval,
            as_of=latest.timestamp,
            latest_close=latest.close,
            bars=len(history),
            swings=swings,
            levels=levels,
            patterns=patterns,
            signal_counts={direction: counts.get(direction, 0) for direction in SignalDirection},
            direction=chart_patterns.net_direction(patterns),
        )
