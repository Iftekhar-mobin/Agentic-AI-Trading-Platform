"""The supervisor graph.

Topology (grows as agents arrive in later milestones):

    START -> supervisor -> technical_analysis -> supervisor -> END

The supervisor is a deterministic router over the state: it dispatches every
agent whose artifact is still missing and ends when nothing is left to do or
an agent has failed. Keeping routing in code (not prompts) is deliberate —
control-flow guarantees like "no execution without risk approval" will be
graph edges, not LLM suggestions.

Agent failures are recorded in state rather than raised, so one failing agent
never crashes a workflow that other agents could still serve.
"""

from __future__ import annotations

from typing import Any, Literal

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from atp.application.orchestration.state import AgentFailure, TradingState
from atp.application.use_cases.analyze_ticker import AnalyzeTicker
from atp.domain.errors import DomainError
from atp.domain.models.market import BarInterval

log = structlog.get_logger()

TECHNICAL_ANALYSIS = "technical_analysis"


class TradingOrchestrator:
    def __init__(self, analyze_ticker: AnalyzeTicker) -> None:
        self._analyze_ticker = analyze_ticker
        self._graph = self._build()

    async def run(self, symbol: str, interval: BarInterval = BarInterval.DAY_1) -> TradingState:
        initial = TradingState(symbol=symbol.strip().upper(), interval=interval)
        log.info("orchestrator.run_started", symbol=initial.symbol, interval=interval.value)
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
        graph.add_node("supervisor", self._supervisor)
        graph.add_node(TECHNICAL_ANALYSIS, self._technical_analysis)
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor", self._route, {TECHNICAL_ANALYSIS: TECHNICAL_ANALYSIS, END: END}
        )
        graph.add_edge(TECHNICAL_ANALYSIS, "supervisor")
        return graph.compile()

    async def _supervisor(self, state: TradingState) -> dict[str, Any]:
        log.debug(
            "supervisor.evaluating",
            symbol=state.symbol,
            completed=state.completed,
            failures=len(state.failures),
        )
        return {}

    def _route(self, state: TradingState) -> str:
        if state.failures:
            return END
        if state.technical_report is None and TECHNICAL_ANALYSIS not in state.completed:
            return TECHNICAL_ANALYSIS
        return END

    async def _technical_analysis(
        self, state: TradingState
    ) -> dict[Literal["technical_report", "completed", "failures"], Any]:
        try:
            report = await self._analyze_ticker.execute(state.symbol, state.interval)
        except DomainError as exc:
            log.warning("agent.failed", agent=TECHNICAL_ANALYSIS, error=str(exc))
            return {
                "completed": [TECHNICAL_ANALYSIS],
                "failures": [AgentFailure(agent=TECHNICAL_ANALYSIS, error=str(exc))],
            }
        return {"technical_report": report, "completed": [TECHNICAL_ANALYSIS]}
