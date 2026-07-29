"""Streamlit dashboard for the Agentic AI Trading Platform.

An internal operator console, not a trading terminal. The thing it is built to
show is *why* — every agent conclusion is rendered with its evidence,
confidence and invalidation conditions, because a recommendation you cannot
interrogate is one you should not act on.

Run it with the API already serving:

    uv run atp serve
    uv run streamlit run ui/streamlit_app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from client import DEFAULT_BASE_URL, ApiError, AtpClient

AGENTS = [
    "technical_analysis",
    "chart_pattern",
    "market_research",
    "fundamental_analysis",
    "news_analysis",
    "sentiment_analysis",
    "continuous_learning",
]

# Highest to lowest — the order multi-timeframe analysis is read in.
TIMEFRAMES = ["1wk", "1d", "4h", "1h", "15m", "5m", "1m"]
DEFAULT_TIMEFRAMES = ["1d", "4h", "1h"]

DIRECTION_ICON = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}

st.set_page_config(page_title="ATP Console", page_icon="📈", layout="wide")


def client() -> AtpClient:
    return AtpClient(
        st.session_state.get("base_url", DEFAULT_BASE_URL),
        st.session_state.get("api_key", ""),
    )


def show_error(exc: ApiError) -> None:
    st.error(str(exc))
    if exc.status_code == 401:
        st.info("Set an API key in the sidebar (`ATP_API__KEYS` on the server).")
    elif exc.status_code == 403:
        st.info("This key lacks the required scope.")


def confidence_bar(label: str, confidence: float) -> None:
    st.progress(min(max(confidence, 0.0), 1.0), text=f"{label} confidence {confidence:.0%}")


def render_explanation(explanation: dict[str, Any]) -> None:
    """The envelope, rendered in full. This is the point of the whole platform."""
    st.markdown(f"**Reasoning.** {explanation['reasoning']}")

    st.markdown("**Evidence**")
    for item in explanation.get("evidence", []):
        st.markdown(f"- `{item['source']}` — {item['statement']}")

    st.markdown("**This conclusion is wrong if**")
    for condition in explanation.get("invalidation_conditions", []):
        st.markdown(f"- {condition}")


def render_timeframe_directions(timeframes: list[dict[str, Any]]) -> None:
    """One tile per timeframe, highest first — the MTF view at a glance."""
    columns = st.columns(len(timeframes))
    for column, frame in zip(columns, timeframes, strict=True):
        icon = DIRECTION_ICON.get(frame["direction"], "⚪")
        column.metric(frame["interval"], f"{icon} {frame['direction']}")


def render_assessment(title: str, report: dict[str, Any], key: str = "assessment") -> None:
    assessment = report[key]
    direction = assessment.get("direction", "neutral")
    with st.expander(f"{DIRECTION_ICON.get(direction, '⚪')} {title} — {direction}", expanded=True):
        confidence_bar(title, assessment["confidence"])
        render_explanation(assessment)


def page_analysis() -> None:
    st.header("Analysis")

    with st.form("analysis"):
        columns = st.columns([2, 3])
        symbol = columns[0].text_input("Symbol", value="AAPL").strip().upper()
        intervals = columns[1].multiselect(
            "Timeframes (multi-timeframe analysis)",
            TIMEFRAMES,
            default=DEFAULT_TIMEFRAMES,
            help=(
                "Pick several and the technical and chart-pattern agents reason about "
                "alignment across them, weighting the highest as context. "
                "4H is aggregated from 1H."
            ),
        )
        agents = st.multiselect("Agents", AGENTS, default=AGENTS)
        submitted = st.form_submit_button("Run analysis", type="primary")

    if not submitted:
        st.caption("Each run costs one LLM call per selected agent.")
        return
    if not symbol:
        st.warning("Enter a symbol.")
        return
    if not intervals:
        st.warning("Pick at least one timeframe.")
        return

    api = client()
    try:
        selected = ", ".join(intervals)
        with st.spinner(f"Running {len(agents)} agents on {symbol} across {selected}..."):
            result = api.analyze(symbol, intervals, agents)
    except ApiError as exc:
        show_error(exc)
        return
    finally:
        api.close()

    for failure in result.get("failures", []):
        st.warning(f"**{failure['agent']}** did not complete: {failure['error']}")

    st.caption("Timeframes analysed: " + ", ".join(result.get("intervals", [])))

    if technical := result.get("technical_report"):
        render_assessment("Technical", technical)
        st.info(f"**Timeframe alignment.** {technical['assessment']['timeframe_alignment']}")
        render_timeframe_directions(technical["timeframes"])
        for frame in technical["timeframes"]:
            with st.expander(f"Indicator readings - {frame['interval']}"):
                st.dataframe(
                    [
                        {
                            "indicator": reading["name"],
                            "direction": reading["direction"],
                            "summary": reading["summary"],
                        }
                        for reading in frame["readings"]
                    ],
                    width="stretch",
                    hide_index=True,
                )

    if patterns := result.get("chart_pattern_report"):
        render_assessment("Chart patterns", patterns)
        st.info(f"**Timeframe alignment.** {patterns['assessment']['timeframe_alignment']}")
        render_timeframe_directions(patterns["timeframes"])
        for frame in patterns["timeframes"]:
            label = (
                f"{frame['interval']} - {len(frame['patterns'])} patterns, "
                f"{len(frame['levels'])} levels"
            )
            with st.expander(label):
                if frame["patterns"]:
                    st.dataframe(
                        [
                            {
                                "pattern": pattern["kind"],
                                "direction": pattern["direction"],
                                "confirmed": pattern["confirmed"],
                                "quality": pattern["quality"],
                                "summary": pattern["summary"],
                            }
                            for pattern in frame["patterns"]
                        ],
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.caption("No formations detected on this timeframe.")
                if frame["levels"]:
                    st.dataframe(
                        [
                            {
                                "level": level["kind"],
                                "price": level["price"],
                                "touches": level["touches"],
                                "distance %": level["distance_pct"],
                            }
                            for level in frame["levels"]
                        ],
                        width="stretch",
                        hide_index=True,
                    )

    if research := result.get("market_research_report"):
        render_assessment(f"Market research vs {research['benchmark']}", research)
        with st.expander("Relative measurements"):
            st.dataframe(
                [
                    {
                        "measure": reading["name"],
                        "direction": reading["direction"],
                        "summary": reading["summary"],
                    }
                    for reading in research["readings"]
                ],
                width="stretch",
                hide_index=True,
            )

    if fundamental := result.get("fundamental_report"):
        render_assessment("Fundamentals", fundamental)
        with st.expander("Metric readings"):
            st.dataframe(
                [
                    {
                        "metric": reading["name"],
                        "category": reading["category"],
                        "direction": reading["direction"],
                        "summary": reading["summary"],
                    }
                    for reading in fundamental["readings"]
                ],
                width="stretch",
                hide_index=True,
            )

    if news := result.get("news_report"):
        render_assessment("News", news)
        st.markdown("**Themes:** " + ", ".join(news["assessment"]["key_themes"]))
        with st.expander(f"{len(news['articles'])} articles"):
            for article in news["articles"]:
                st.markdown(
                    f"- [{article['title']}]({article['url'] or '#'}) — "
                    f"{article['publisher'] or 'unknown'}, {article['published_at'][:16]}"
                )

    if sentiment := result.get("sentiment_report"):
        render_assessment("Sentiment", sentiment)
        summary = sentiment["summary"]
        columns = st.columns(4)
        columns[0].metric("Weighted polarity", f"{summary['weighted_polarity']:+.2f}")
        columns[1].metric("Articles", summary["article_count"])
        columns[2].metric("Aggregate", summary["direction"])
        columns[3].metric("Classifier", sentiment["model_name"])

    if learning := result.get("learning_report"):
        render_assessment("Learning", learning, key="entry")
        if regime := learning.get("regime"):
            st.info(f"Market regime: **{regime['trend']} / {regime['volatility']}**")
        st.markdown("**Lessons**")
        for lesson in learning["entry"]["lessons"]:
            st.markdown(f"- {lesson}")
        st.caption(learning["entry"]["regime_note"])
        if recalled := learning.get("recalled"):
            with st.expander(f"{len(recalled)} recalled episodes"):
                for match in recalled:
                    episode = match["episode"]
                    st.markdown(
                        f"- **{episode['kind']}** · {episode['occurred_at'][:10]} · "
                        f"similarity {match['score']:.2f} — {episode['summary'][:200]}"
                    )


def page_portfolio() -> None:
    st.header("Portfolio")
    api = client()
    try:
        portfolio = api.portfolio()
        orders = api.orders(limit=50)
    except ApiError as exc:
        show_error(exc)
        return
    finally:
        api.close()

    columns = st.columns(5)
    columns[0].metric("Equity", f"${float(portfolio['equity']):,.2f}")
    columns[1].metric("Cash", f"${float(portfolio['cash']):,.2f}")
    columns[2].metric("Exposure", f"{float(portfolio['exposure_pct']):.1f}%")
    columns[3].metric("Drawdown", f"{float(portfolio['drawdown_pct']):.1f}%")
    columns[4].metric("Realized P&L today", f"${float(portfolio['realized_pnl_today']):,.2f}")

    st.subheader("Positions")
    positions = portfolio["positions"]
    if positions:
        st.dataframe(
            [
                {
                    "symbol": position["symbol"],
                    "qty": float(position["quantity"]),
                    "entry": float(position["avg_entry_price"]),
                    "price": float(position["current_price"]),
                    "value": float(position["market_value"]),
                    "P&L": float(position["unrealized_pnl"]),
                    "P&L %": float(position["unrealized_pnl_pct"]),
                }
                for position in positions
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No open positions.")

    st.subheader("Execution audit trail")
    records = orders["records"]
    if records:
        st.dataframe(
            [
                {
                    "submitted": record["order"]["submitted_at"][:19],
                    "symbol": record["order"]["symbol"],
                    "side": record["order"]["side"],
                    "qty": float(record["order"]["quantity"]),
                    "status": record["order"]["status"],
                    "fill": record["order"]["fill_price"],
                    "verdict": record["decision"]["verdict"],
                }
                for record in reversed(records)
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No orders yet.")


def page_trade() -> None:
    st.header("Trade")
    st.caption(
        "Every proposal passes the risk gate. A rejection is the system working — "
        "the reasons are shown below."
    )

    with st.form("trade"):
        columns = st.columns(3)
        symbol = columns[0].text_input("Symbol", value="AAPL").strip().upper()
        side = columns[1].selectbox("Side", ["buy", "sell"])
        quantity = columns[2].number_input(
            "Quantity (0 = size from risk limits)", min_value=0.0, step=1.0, value=0.0
        )
        columns = st.columns(2)
        stop_loss = columns[0].number_input("Stop loss", min_value=0.0, step=1.0, value=0.0)
        take_profit = columns[1].number_input("Take profit", min_value=0.0, step=1.0, value=0.0)

        actions = st.columns(2)
        checked = actions[0].form_submit_button("Risk check", type="secondary")
        executed = actions[1].form_submit_button("Execute", type="primary")

    if not (checked or executed):
        return

    payload: dict[str, Any] = {"symbol": symbol, "side": side}
    if quantity > 0:
        payload["quantity"] = quantity
    if stop_loss > 0:
        payload["stop_loss"] = stop_loss
    if take_profit > 0:
        payload["take_profit"] = take_profit

    api = client()
    try:
        result = api.trade(**payload) if executed else api.risk_check(**payload)
    except ApiError as exc:
        show_error(exc)
        return
    finally:
        api.close()

    decision = result["decision"]
    proposal = decision["proposal"]
    st.write(
        f"Proposal: **{proposal['side']} {float(proposal['quantity']):g} {proposal['symbol']}** "
        f"at {float(proposal['entry_price']):,.2f}"
    )
    if decision["verdict"] == "approved":
        st.success("Risk gate: APPROVED")
    else:
        st.error("Risk gate: REJECTED")
        for violation in decision["violations"]:
            st.markdown(f"- **{violation['rule']}** — {violation['detail']}")

    with st.expander("Risk metrics"):
        st.json(decision["metrics"])

    if order := result.get("order"):
        st.json(order)


def page_memory() -> None:
    st.header("Memory")
    st.caption("Regime-tagged episodes from the platform's own journal.")

    columns = st.columns([2, 4, 1])
    symbol = columns[0].text_input("Symbol", value="AAPL").strip().upper()
    query = columns[1].text_input("Query", placeholder="defaults to the symbol")
    limit = columns[2].number_input("Limit", min_value=1, max_value=50, value=10)

    if not st.button("Recall", type="primary"):
        return

    api = client()
    try:
        result = api.memory(symbol, query or None, int(limit))
    except ApiError as exc:
        show_error(exc)
        return
    finally:
        api.close()

    matches = result["matches"]
    if not matches:
        st.info("Nothing recalled yet — run some analyses or trades first.")
        return

    for match in matches:
        episode = match["episode"]
        regime = episode.get("regime")
        label = f"{regime['trend']}/{regime['volatility']}" if regime else "unknown regime"
        with st.expander(
            f"{episode['kind']} · {episode['occurred_at'][:10]} · {label} · "
            f"similarity {match['score']:.2f}"
        ):
            st.write(episode["summary"])
            if episode.get("metadata"):
                st.json(episode["metadata"])


PAGES = {
    "Analysis": page_analysis,
    "Portfolio": page_portfolio,
    "Trade": page_trade,
    "Memory": page_memory,
}


def main() -> None:
    st.sidebar.title("📈 ATP Console")
    st.session_state.setdefault("base_url", DEFAULT_BASE_URL)
    st.session_state.setdefault("api_key", "")

    st.sidebar.text_input("API base URL", key="base_url")
    st.sidebar.text_input("API key", key="api_key", type="password")

    api = client()
    try:
        health = api.health()
        st.sidebar.success(f"API {health['version']} · {health['status']}")
    except ApiError as exc:
        st.sidebar.error(f"API unreachable\n\n{exc}")
    finally:
        api.close()

    page = st.sidebar.radio("Page", list(PAGES))
    st.sidebar.caption("Decision support only. Recommendations are explainable, not guaranteed.")
    PAGES[page]()


main()
