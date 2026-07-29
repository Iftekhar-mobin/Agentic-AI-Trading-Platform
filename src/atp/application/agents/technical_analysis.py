"""Technical Analysis agent (multi-timeframe).

Division of labor:
- The IndicatorEngine computes and classifies every number, per timeframe
  (deterministic).
- The LLM interprets the readings into one assessment spanning all of them; it
  receives only precomputed values and may not invent numbers.

Why one assessment rather than one per timeframe: the useful output of
multi-timeframe analysis is the *relationship* between the timeframes. Daily
bullish with an overbought hourly is a "wait for the pullback", not two
independent verdicts, and no amount of post-hoc merging recovers that.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence

import structlog

from atp.application.agents.evidence import warn_on_unknown_sources
from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.analysis import (
    SignalDirection,
    TechnicalAssessment,
    TechnicalReport,
    TimeframeReadings,
    net_direction,
)
from atp.domain.models.market import PriceHistory
from atp.domain.ports.indicators import IndicatorEngine
from atp.domain.ports.llm import LLMClient

log = structlog.get_logger()

AGENT_NAME = "technical_analysis"

SYSTEM_PROMPT = """\
You are the Technical Analysis Agent of an AI trading platform. Traders use \
your output as decision support, never as guaranteed predictions.

You receive precomputed, deterministic indicator readings for one symbol on \
one or more timeframes, ordered from highest to lowest. Interpret them into a \
single overall technical assessment.

Rules:
- Never invent or recompute numbers; cite only the values provided.
- Every evidence item's "source" must be "<interval>:<reading name>", e.g. \
"1d:rsi_14", so each claim is traceable to the timeframe it came from.
- Read top down. The highest timeframe sets the context; a lower-timeframe \
signal that fights it is a counter-trend trade and must be called one.
- timeframe_alignment must state plainly whether the timeframes agree. When \
they conflict, say which one you are weighting and why, and what the lower \
timeframe would have to do to line up.
- Agreement across timeframes raises confidence; conflict lowers it. \
Confidence is calibrated 0-1: 0.5 means genuinely mixed evidence, and values \
above 0.8 require both multi-indicator and multi-timeframe agreement.
- Invalidation conditions must be concrete and observable, using price levels \
from the provided values and naming the timeframe (e.g. "a daily close below \
the SMA(50) at 187.40").
- This is a single-symbol technical view only; do not speculate about news, \
fundamentals, or macro conditions.
"""


class TechnicalAnalysisAgent:
    def __init__(self, llm: LLMClient, indicators: IndicatorEngine) -> None:
        self._llm = llm
        self._indicators = indicators

    async def analyze(self, histories: Sequence[PriceHistory]) -> TechnicalReport:
        if not histories:
            msg = "no timeframes supplied for technical analysis"
            raise InsufficientHistoryError(msg)

        symbol = histories[0].symbol
        timeframes = tuple(self._read_timeframe(history) for history in histories)

        payload = {
            "symbol": symbol,
            "timeframes": [
                {
                    "interval": timeframe.interval.value,
                    "as_of": timeframe.as_of.isoformat(),
                    "latest_close": timeframe.latest_close,
                    "bars": timeframe.bars,
                    "deterministic_direction": timeframe.direction.value,
                    "signal_counts": {
                        direction.value: count
                        for direction, count in timeframe.signal_counts.items()
                    },
                    "readings": [reading.model_dump(mode="json") for reading in timeframe.readings],
                }
                for timeframe in timeframes
            ],
        }
        assessment = await self._llm.generate_structured(
            TechnicalAssessment,
            system=SYSTEM_PROMPT,
            prompt=json.dumps(payload, sort_keys=True),
        )

        known = {
            f"{timeframe.interval.value}:{reading.name}"
            for timeframe in timeframes
            for reading in timeframe.readings
        }
        warn_on_unknown_sources(AGENT_NAME, symbol, assessment.evidence, known)

        report = TechnicalReport(symbol=symbol, timeframes=timeframes, assessment=assessment)
        log.info(
            "technical_analysis.completed",
            symbol=symbol,
            intervals=[interval.value for interval in report.intervals],
            direction=assessment.direction.value,
            confidence=assessment.confidence,
            timeframes_aligned=report.aligned,
        )
        return report

    def _read_timeframe(self, history: PriceHistory) -> TimeframeReadings:
        latest = history.latest
        if latest is None:
            msg = f"{history.symbol}/{history.interval.value}: no bars available to analyze"
            raise InsufficientHistoryError(msg)

        readings = self._indicators.compute_readings(history)
        counts = Counter(reading.direction for reading in readings)
        return TimeframeReadings(
            interval=history.interval,
            as_of=latest.timestamp,
            latest_close=latest.close,
            bars=len(history),
            readings=readings,
            signal_counts={direction: counts.get(direction, 0) for direction in SignalDirection},
            direction=net_direction([reading.direction for reading in readings]),
        )
