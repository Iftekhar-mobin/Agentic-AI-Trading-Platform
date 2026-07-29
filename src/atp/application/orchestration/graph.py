"""The supervisor graph.

Topology:

                    ┌─> technical_analysis   ─┐
                    ├─> fundamental_analysis ─┤
    START -> supervisor -> news_analysis     ─┼─> supervisor -> END
                    └─> sentiment_analysis   ─┘

The supervisor is a deterministic router over the state: it dispatches every
selected agent whose artifact is still missing, all in one superstep, and ends
when nothing is left to do. The analysis agents are independent — none reads
another's output — so fanning them out is both correct and the difference
between one LLM round-trip of latency and four.

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

from atp.application.orchestration.state import AgentFailure, TradingState
from atp.application.use_cases.analyze_fundamentals import AnalyzeFundamentals
from atp.application.use_cases.analyze_news import AnalyzeNews
from atp.application.use_cases.analyze_sentiment import AnalyzeSentiment
from atp.application.use_cases.analyze_ticker import AnalyzeTicker
from atp.domain.errors import DomainError
from atp.domain.models.market import BarInterval

log = structlog.get_logger()

SUPERVISOR = "supervisor"
TECHNICAL_ANALYSIS = "technical_analysis"
FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
NEWS_ANALYSIS = "news_analysis"
SENTIMENT_ANALYSIS = "sentiment_analysis"

ANALYSIS_AGENTS: tuple[str, ...] = (
    TECHNICAL_ANALYSIS,
    FUNDAMENTAL_ANALYSIS,
    NEWS_ANALYSIS,
    SENTIMENT_ANALYSIS,
)
"""Canonical dispatch order. Also the set of names ``--agents`` accepts."""

_ARTIFACT_FIELDS: dict[str, str] = {
    TECHNICAL_ANALYSIS: "technical_report",
    FUNDAMENTAL_ANALYSIS: "fundamental_report",
    NEWS_ANALYSIS: "news_report",
    SENTIMENT_ANALYSIS: "sentiment_report",
}


class UnknownAgentError(ValueError):
    """A caller asked for an agent the supervisor does not have."""


def resolve_agents(requested: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize a requested agent selection, preserving canonical dispatch order.

    ``None`` or an empty selection means every analysis agent.
    """
    if not requested:
        return ANALYSIS_AGENTS
    names = {name.strip().lower() for name in requested}
    unknown = sorted(names - set(ANALYSIS_AGENTS))
    if unknown:
        msg = f"unknown agents {unknown}; available: {list(ANALYSIS_AGENTS)}"
        raise UnknownAgentError(msg)
    return tuple(name for name in ANALYSIS_AGENTS if name in names)


class TradingOrchestrator:
    def __init__(
        self,
        analyze_ticker: AnalyzeTicker,
        analyze_fundamentals: AnalyzeFundamentals,
        analyze_news: AnalyzeNews,
        analyze_sentiment: AnalyzeSentiment,
    ) -> None:
        self._analyze_ticker = analyze_ticker
        self._analyze_fundamentals = analyze_fundamentals
        self._analyze_news = analyze_news
        self._analyze_sentiment = analyze_sentiment
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

        graph.add_edge(START, SUPERVISOR)
        graph.add_conditional_edges(
            SUPERVISOR,
            self._route,
            {**{name: name for name in ANALYSIS_AGENTS}, END: END},
        )
        for name in ANALYSIS_AGENTS:
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
        """Dispatch every still-pending agent at once; ``[END]`` when none remain."""
        selected = state.requested_agents or ANALYSIS_AGENTS
        pending = [
            name
            for name in selected
            if name not in state.completed and getattr(state, _ARTIFACT_FIELDS[name], None) is None
        ]
        if not pending:
            return [END]
        log.debug("supervisor.dispatching", symbol=state.symbol, agents=pending)
        return pending

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
