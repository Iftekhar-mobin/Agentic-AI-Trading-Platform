"""The supervisor graph.

Topology — two phases, because the second one reads the first one's output:

                    ┌─> technical_analysis   ─┐
                    ├─> fundamental_analysis ─┤
    START -> supervisor -> news_analysis     ─┼─> supervisor
                    └─> sentiment_analysis   ─┘        │
                                                       ├─> continuous_learning
                                                       │        │
                                                       └────────┴─> END

The supervisor is a deterministic router over the state: it dispatches every
selected agent whose artifact is still missing, all in one superstep, and ends
when nothing is left to do. The analysis agents are independent — none reads
another's output — so fanning them out is both correct and the difference
between one LLM round-trip of latency and four.

Continuous learning is a phase of its own rather than a fifth parallel branch:
it reflects on what the analysis pool concluded, so it cannot start until they
have finished. It is skipped entirely when no analysis produced a report —
there would be nothing to reflect on.

Keeping routing in code (not prompts) is deliberate: control-flow guarantees
like "no execution without risk approval" are graph edges, not LLM suggestions.

Agent failures are recorded in state rather than raised, so one failing agent
never costs the caller the reports the others produced. Every agent node
appends its name to ``completed`` whether it succeeded or failed, which is
what guarantees the router terminates.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from atp.application.orchestration.situation import render_situation
from atp.application.orchestration.state import AgentFailure, TradingState
from atp.application.use_cases.analyze_fundamentals import AnalyzeFundamentals
from atp.application.use_cases.analyze_news import AnalyzeNews
from atp.application.use_cases.analyze_sentiment import AnalyzeSentiment
from atp.application.use_cases.analyze_ticker import AnalyzeTicker
from atp.application.use_cases.learn_from_context import LearnFromContext
from atp.domain.errors import DomainError
from atp.domain.models.market import BarInterval

log = structlog.get_logger()

SUPERVISOR = "supervisor"
TECHNICAL_ANALYSIS = "technical_analysis"
FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
NEWS_ANALYSIS = "news_analysis"
SENTIMENT_ANALYSIS = "sentiment_analysis"
CONTINUOUS_LEARNING = "continuous_learning"

ANALYSIS_AGENTS: tuple[str, ...] = (
    TECHNICAL_ANALYSIS,
    FUNDAMENTAL_ANALYSIS,
    NEWS_ANALYSIS,
    SENTIMENT_ANALYSIS,
)
"""Phase one: canonical dispatch order, all concurrent."""

FEEDBACK_AGENTS: tuple[str, ...] = (CONTINUOUS_LEARNING,)
"""Phase two: runs only after phase one, and only if phase one produced anything."""

ALL_AGENTS: tuple[str, ...] = ANALYSIS_AGENTS + FEEDBACK_AGENTS
"""Every dispatchable agent. Also the set of names ``--agents`` accepts."""

_ARTIFACT_FIELDS: dict[str, str] = {
    TECHNICAL_ANALYSIS: "technical_report",
    FUNDAMENTAL_ANALYSIS: "fundamental_report",
    NEWS_ANALYSIS: "news_report",
    SENTIMENT_ANALYSIS: "sentiment_report",
    CONTINUOUS_LEARNING: "learning_report",
}


class UnknownAgentError(ValueError):
    """A caller asked for an agent the supervisor does not have."""


def resolve_agents(requested: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize a requested agent selection, preserving canonical dispatch order.

    ``None`` or an empty selection means every agent.
    """
    if not requested:
        return ALL_AGENTS
    names = {name.strip().lower() for name in requested}
    unknown = sorted(names - set(ALL_AGENTS))
    if unknown:
        msg = f"unknown agents {unknown}; available: {list(ALL_AGENTS)}"
        raise UnknownAgentError(msg)
    return tuple(name for name in ALL_AGENTS if name in names)


class TradingOrchestrator:
    def __init__(
        self,
        analyze_ticker: AnalyzeTicker,
        analyze_fundamentals: AnalyzeFundamentals,
        analyze_news: AnalyzeNews,
        analyze_sentiment: AnalyzeSentiment,
        learn_from_context: LearnFromContext,
    ) -> None:
        self._analyze_ticker = analyze_ticker
        self._analyze_fundamentals = analyze_fundamentals
        self._analyze_news = analyze_news
        self._analyze_sentiment = analyze_sentiment
        self._learn_from_context = learn_from_context
        self._graph = self._build()

    async def run(
        self,
        symbol: str,
        interval: BarInterval = BarInterval.DAY_1,
        *,
        agents: Sequence[str] | None = None,
    ) -> TradingState:
        initial = TradingState(
            symbol=symbol.strip().upper(),
            interval=interval,
            requested_agents=resolve_agents(agents),
        )
        log.info(
            "orchestrator.run_started",
            symbol=initial.symbol,
            interval=interval.value,
            agents=list(initial.requested_agents),
        )
        result = await self._graph.ainvoke(initial)
        state = TradingState.model_validate(result)
        log.info(
            "orchestrator.run_finished",
            symbol=state.symbol,
            completed=state.completed,
            failures=[failure.agent for failure in state.failures],
        )
        return state

    def _build(self) -> CompiledStateGraph[TradingState, Any, Any, Any]:
        graph: StateGraph[TradingState, Any, Any, Any] = StateGraph(TradingState)
        graph.add_node(SUPERVISOR, self._supervisor)
        graph.add_node(TECHNICAL_ANALYSIS, self._technical_analysis)
        graph.add_node(FUNDAMENTAL_ANALYSIS, self._fundamental_analysis)
        graph.add_node(NEWS_ANALYSIS, self._news_analysis)
        graph.add_node(SENTIMENT_ANALYSIS, self._sentiment_analysis)
        graph.add_node(CONTINUOUS_LEARNING, self._continuous_learning)

        graph.add_edge(START, SUPERVISOR)
        graph.add_conditional_edges(
            SUPERVISOR,
            self._route,
            {**{name: name for name in ALL_AGENTS}, END: END},
        )
        for name in ALL_AGENTS:
            graph.add_edge(name, SUPERVISOR)
        return graph.compile()

    async def _supervisor(self, state: TradingState) -> dict[str, Any]:
        log.debug(
            "supervisor.evaluating",
            symbol=state.symbol,
            completed=state.completed,
            failures=len(state.failures),
        )
        return {}

    def _route(self, state: TradingState) -> list[str]:
        """Dispatch phase one concurrently, then phase two; ``[END]`` when done."""
        selected = state.requested_agents or ALL_AGENTS

        analysis = self._pending(state, selected, ANALYSIS_AGENTS)
        if analysis:
            log.debug("supervisor.dispatching", symbol=state.symbol, agents=analysis)
            return analysis

        # Phase two reads phase one's output, so reflecting on an empty
        # analysis would be reflecting on nothing.
        if state.has_analysis:
            feedback = self._pending(state, selected, FEEDBACK_AGENTS)
            if feedback:
                log.debug("supervisor.dispatching", symbol=state.symbol, agents=feedback)
                return feedback
        return [END]

    @staticmethod
    def _pending(state: TradingState, selected: Sequence[str], phase: Sequence[str]) -> list[str]:
        return [
            name
            for name in phase
            if name in selected
            and name not in state.completed
            and getattr(state, _ARTIFACT_FIELDS[name], None) is None
        ]

    async def _technical_analysis(self, state: TradingState) -> dict[str, Any]:
        return await self._dispatch(
            TECHNICAL_ANALYSIS,
            state,
            lambda: self._analyze_ticker.execute(state.symbol, state.interval),
        )

    async def _fundamental_analysis(self, state: TradingState) -> dict[str, Any]:
        return await self._dispatch(
            FUNDAMENTAL_ANALYSIS,
            state,
            lambda: self._analyze_fundamentals.execute(state.symbol),
        )

    async def _news_analysis(self, state: TradingState) -> dict[str, Any]:
        return await self._dispatch(
            NEWS_ANALYSIS, state, lambda: self._analyze_news.execute(state.symbol)
        )

    async def _sentiment_analysis(self, state: TradingState) -> dict[str, Any]:
        return await self._dispatch(
            SENTIMENT_ANALYSIS, state, lambda: self._analyze_sentiment.execute(state.symbol)
        )

    async def _continuous_learning(self, state: TradingState) -> dict[str, Any]:
        situation = render_situation(state)
        return await self._dispatch(
            CONTINUOUS_LEARNING,
            state,
            lambda: self._learn_from_context.execute(
                state.symbol, state.interval, situation=situation
            ),
        )

    @staticmethod
    async def _dispatch(
        agent: str,
        state: TradingState,
        run: Callable[[], Awaitable[BaseModel]],
    ) -> dict[str, Any]:
        """Run one agent, converting a domain failure into recorded state."""
        try:
            report = await run()
        except DomainError as exc:
            log.warning("agent.failed", agent=agent, symbol=state.symbol, error=str(exc))
            return {
                "completed": [agent],
                "failures": [AgentFailure(agent=agent, error=str(exc))],
            }
        return {_ARTIFACT_FIELDS[agent]: report, "completed": [agent]}
