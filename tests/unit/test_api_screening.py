"""Tests for the screening endpoints: the menu, and the ranked shortlist."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from factories import make_technical_report
from fastapi.testclient import TestClient

from atp.application.orchestration import ScreenOpportunities
from atp.application.orchestration.consensus import ConsensusResult
from atp.application.orchestration.screening import ScreenResult, SymbolFailure
from atp.application.orchestration.state import TradingState
from atp.composition import Container
from atp.domain.models.analysis import SignalDirection
from atp.domain.models.explainability import Evidence
from atp.domain.models.ranking import (
    Candidate,
    OpportunityRanking,
    OpportunityVerdict,
    RankedOpportunity,
    ScoreComponents,
)
from atp.domain.models.universe import AssetClass
from atp.domain.models.voting import (
    BotStatus,
    ConsensusAction,
    ConsensusDecision,
    Vote,
    VoterKind,
    VotingPolicy,
)
from atp.infrastructure.config import Settings
from atp.interfaces.api import create_app

POLICY = VotingPolicy()


def _outcome(symbol: str, score: float) -> ConsensusResult:
    return ConsensusResult(
        decision=ConsensusDecision(
            symbol=symbol,
            action=ConsensusAction.BUY,
            score=score,
            votes=(
                Vote(
                    voter="technical_analysis",
                    kind=VoterKind.AGENT,
                    direction=SignalDirection.BULLISH,
                    confidence=0.8,
                ),
            ),
            policy=POLICY,
            quorum_met=True,
            reason="test",
        ),
        bot_status=BotStatus.NOT_CONFIGURED,
        bot_note="none configured",
        state=TradingState(symbol=symbol, technical_report=make_technical_report(symbol)),
    )


def _candidate(symbol: str, composite: float) -> Candidate:
    return Candidate(
        symbol=symbol,
        name=symbol,
        asset_class=AssetClass.STOCK,
        action=ConsensusAction.BUY,
        score=0.8,
        agreement=1.0,
        quorum_met=True,
        components=ScoreComponents(
            conviction=0.8,
            agreement=1.0,
            confidence=0.8,
            coverage=1.0,
            quorum_penalty=1.0,
            risk_penalty=1.0,
            composite=composite,
        ),
    )


def _verdict(symbol: str) -> OpportunityVerdict:
    return OpportunityVerdict(
        symbol=symbol,
        profit_potential=0.8,
        key_driver="trend",
        primary_risk="reversal",
        horizon="days",
        reasoning="strongest of the set",
        confidence=0.7,
        evidence=(Evidence(source=symbol, statement="aligned"),),
        invalidation_conditions=("trend breaks",),
    )


class StubScreener:
    """Returns a canned screen; records what the route asked for."""

    def __init__(self, symbols: Sequence[str] = ("NVDA", "AAPL"), *, empty: bool = False) -> None:
        self._symbols = list(symbols)
        self._empty = empty
        self.calls: list[dict[str, Any]] = []

    async def execute(self, symbols: Sequence[str], intervals: Any = None, **kwargs: Any) -> Any:
        self.calls.append({"symbols": list(symbols), "intervals": intervals, **kwargs})
        if self._empty:
            return ScreenResult(
                ranking=OpportunityRanking(ranked=(), considered=0, ranked_by_agent=False),
                failures=[SymbolFailure(symbol=symbol, error="no data") for symbol in symbols],
                requested=list(symbols),
            )
        return ScreenResult(
            ranking=OpportunityRanking(
                ranked=(
                    RankedOpportunity(
                        rank=1,
                        candidate=_candidate(self._symbols[0], 0.9),
                        verdict=_verdict(self._symbols[0]),
                    ),
                ),
                narrative="one clear leader",
                considered=len(self._symbols),
            ),
            outcomes={symbol: _outcome(symbol, 0.8) for symbol in self._symbols},
            requested=self._symbols,
            duration_ms=1234.5,
        )


@pytest.fixture
def screener() -> StubScreener:
    return StubScreener()


@pytest.fixture
def client(tmp_path: Path, screener: StubScreener) -> Iterator[TestClient]:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    container = dataclasses.replace(
        Container.build(settings), screen_opportunities=cast(ScreenOpportunities, screener)
    )
    with TestClient(create_app(container)) as test_client:
        yield test_client


def test_the_universe_is_grouped_and_carries_vendor_tickers(client: TestClient) -> None:
    """An operator reading XAUUSD needs to see which series produced the numbers."""
    body = client.get("/universe").json()

    assert body["count"] > 0
    classes = {category["asset_class"] for category in body["categories"]}
    assert {"forex", "stock", "index", "commodity", "crypto"} <= classes

    assets = {
        asset["symbol"]: asset for category in body["categories"] for asset in category["assets"]
    }
    assert assets["XAUUSD"]["vendor_symbol"] == "GC=F"
    assert assets["EURUSD"]["vendor_symbol"] == "EURUSD=X"
    assert assets["AAPL"]["vendor_symbol"] == "AAPL"


def test_the_universe_advertises_the_screen_cap(client: TestClient) -> None:
    body = client.get("/universe").json()
    assert body["max_symbols"] == Settings(_env_file=None).screening.max_symbols
    assert body["default_selection"]


def test_a_screen_returns_the_shortlist_and_every_outcome(
    client: TestClient, screener: StubScreener
) -> None:
    response = client.post(
        "/screen",
        json={"symbols": ["NVDA", "AAPL"], "intervals": ["1d"], "top_n": 1},
    )
    assert response.status_code == 200
    body = response.json()

    ranked = body["ranking"]["ranked"]
    assert [row["candidate"]["symbol"] for row in ranked] == ["NVDA"]
    assert ranked[0]["verdict"]["key_driver"] == "trend"
    # ...and the ones that missed the cut are still returned with their analysis,
    # so "why is mine not listed?" does not cost a second screen.
    assert [outcome["symbol"] for outcome in body["outcomes"]] == ["NVDA", "AAPL"]
    assert body["outcomes"][1]["reports"]["technical_report"]["symbol"] == "AAPL"
    assert screener.calls[0]["top_n"] == 1


def test_request_options_reach_the_use_case(client: TestClient, screener: StubScreener) -> None:
    client.post(
        "/screen",
        json={
            "symbols": ["NVDA"],
            "intervals": ["1d", "4h"],
            "agents": ["technical_analysis"],
            "expected_bot": "sniper_bot",
        },
    )
    call = screener.calls[0]
    assert [interval.value for interval in call["intervals"]] == ["1d", "4h"]
    assert call["agents"] == ["technical_analysis"]
    assert call["expected_bot"] == "sniper_bot"


def test_a_screen_where_nothing_could_be_analysed_is_a_502(tmp_path: Path) -> None:
    """An empty leaderboard would read as 'no opportunities', not 'no data'."""
    settings = Settings(_env_file=None, data_dir=tmp_path)
    container = dataclasses.replace(
        Container.build(settings),
        screen_opportunities=cast(ScreenOpportunities, StubScreener(empty=True)),
    )
    with TestClient(create_app(container)) as client:
        response = client.post("/screen", json={"symbols": ["NOPE"]})

    assert response.status_code == 502
    assert response.json()["detail"]["failures"][0]["symbol"] == "NOPE"


def test_an_empty_basket_is_rejected(client: TestClient) -> None:
    assert client.post("/screen", json={"symbols": []}).status_code == 422
