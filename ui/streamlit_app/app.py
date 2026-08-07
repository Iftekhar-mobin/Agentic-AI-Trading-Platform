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

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from client import DEFAULT_BASE_URL, MAX_BODY_CHARS, ApiError, AtpClient


def _fill_width() -> dict[str, Any]:
    """Kwargs telling a table to fill its container, across Streamlit versions.

    1.51 replaced ``use_container_width=True`` with ``width="stretch"``, and
    passing the new form to an older build raises ``TypeError`` at render time.
    The dashboard is routinely launched with whatever Streamlit is on the path
    rather than the project's pinned one, so it supports both.
    """
    try:
        major, minor = (int(part) for part in st.__version__.split(".")[:2])
    except ValueError:  # a dev build like "1.60.0.dev0" - assume current
        return {"width": "stretch"}
    return {"width": "stretch"} if (major, minor) >= (1, 51) else {"use_container_width": True}


FILL_WIDTH = _fill_width()

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

PROVIDER_HELP = {
    "openrouter": (
        "No models returned. OpenRouter's catalog is public, so this usually means "
        "the server could not reach openrouter.ai."
    ),
    "ollama": (
        "Ollama is not reachable. Install it from ollama.com, run `ollama serve`, "
        "then reload this page."
    ),
    "anthropic": "No models configured.",
}

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


STATUS_ICON = {"ok": "✅", "failed": "❌"}


def _duration_bar(duration_ms: float, longest: float, width: int = 18) -> str:
    """A proportional bar, so the expensive step is obvious at a glance."""
    if longest <= 0:
        return ""
    filled = max(1, round(width * duration_ms / longest))
    return "█" * filled


def record_processing(api: AtpClient, result: dict[str, Any] | None = None) -> None:
    """Stash what a call did, so the panel survives Streamlit's reruns.

    Streamlit re-executes the whole script on every interaction; without this
    the trace would vanish the moment you expanded a section to read it.
    """
    st.session_state["processing"] = {
        "exchanges": list(api.exchanges),
        "result": result,
    }


def render_processing() -> None:
    """The Processing panel: what was asked, what ran, and what came back.

    Deliberately shows the mechanism rather than a tidy summary. When a run is
    slow, partial, or surprising, this is the page that says why — which agent
    burned the time, which one failed and with what error, which model produced
    the reasoning, and the exact JSON that crossed the wire.
    """
    trace = st.session_state.get("processing")
    if not trace:
        st.caption("Run an analysis to see how the work was done.")
        return

    exchanges: list[dict[str, Any]] = trace.get("exchanges") or []
    result: dict[str, Any] = trace.get("result") or {}
    steps: list[dict[str, Any]] = result.get("steps") or []

    # --- What the run cost -------------------------------------------------
    model = result.get("active_model") or {}
    failed = [step for step in steps if step["status"] == "failed"]
    columns = st.columns(4)
    columns[0].metric("Total time", f"{result.get('duration_ms', 0) / 1000:.1f}s")
    columns[1].metric("Agents run", len(steps))
    columns[2].metric("Failed", len(failed), delta=None if not failed else f"-{len(failed)}")
    columns[3].metric("Model", model.get("model", "—").split("/")[-1] or "—")
    if model:
        st.caption(f"Reasoning produced by **{model.get('provider')} / {model.get('model')}**")

    # --- Which agent did what ---------------------------------------------
    if steps:
        st.subheader("Agent timeline")
        st.caption(
            "The analysis agents are dispatched together in one superstep, so their "
            "durations overlap and will not sum to the total. `continuous_learning` "
            "runs afterwards, because it reflects on what the others concluded."
        )
        longest = max(step["duration_ms"] for step in steps)
        st.dataframe(
            [
                {
                    "": STATUS_ICON.get(step["status"], "•"),
                    "agent": step["agent"],
                    "phase": step["phase"],
                    "seconds": round(step["duration_ms"] / 1000, 2),
                    "time": _duration_bar(step["duration_ms"], longest),
                    "outcome": step["detail"],
                }
                for step in steps
            ],
            **FILL_WIDTH,
            hide_index=True,
        )
        for step in failed:
            st.warning(
                f"**{step['agent']}** failed after "
                f"{step['duration_ms'] / 1000:.1f}s — {step['detail']}"
            )

    # --- What crossed the wire --------------------------------------------
    st.subheader("API calls")
    if not exchanges:
        st.caption("No requests recorded.")
    for index, exchange in enumerate(exchanges, start=1):
        status = exchange.get("status", 0)
        icon = "✅" if 200 <= status < 300 else "❌"
        label = (
            f"{icon} {index}. {exchange['method']} {exchange['url']} → "
            f"{status or 'no response'} · {exchange.get('duration_ms', 0) / 1000:.2f}s"
        )
        with st.expander(label):
            meta = st.columns(3)
            meta[0].metric("Status", status or "—")
            meta[1].metric("Duration", f"{exchange.get('duration_ms', 0) / 1000:.2f}s")
            meta[2].metric("Response size", f"{exchange.get('bytes', 0):,} B")
            if request_id := exchange.get("request_id"):
                st.caption(
                    f"`X-Request-ID: {request_id}` — every server log line for this "
                    "call carries the same id."
                )
            if error := exchange.get("error"):
                st.error(error)

            st.markdown("**Request sent**")
            st.json(exchange.get("request") or {}, expanded=True)

            if "response" in exchange:
                st.markdown("**Response received**")
                body = json.dumps(exchange["response"], indent=2, default=str)
                if len(body) > MAX_BODY_CHARS:
                    st.caption(f"Truncated to {MAX_BODY_CHARS:,} of {len(body):,} characters.")
                    st.code(body[:MAX_BODY_CHARS] + "\n...", language="json")
                else:
                    st.json(exchange["response"], expanded=False)


BOT_STATUS_RENDER = {
    "voted": ("✅", st.success),
    "stale": ("🕒", st.warning),
    "missing": ("⚠️", st.warning),
    "not_configured": ("🤖", st.info),
}

ACTION_RENDER = {
    "buy": ("🟢 BUY", st.success),
    "sell": ("🔴 SELL", st.error),
    "hold": ("⚪ HOLD", st.info),
}


def page_consensus() -> None:
    st.header("Consensus")
    st.caption(
        "Your trading bot and the agent pool both vote; a deterministic policy "
        "counts them. Consensus decides what is wanted — the risk gate still "
        "decides what is allowed, and nothing executes without you."
    )

    with st.form("consensus"):
        columns = st.columns([2, 3, 2])
        symbol = columns[0].text_input("Symbol", value="XAUUSD").strip().upper()
        intervals = columns[1].multiselect("Timeframes", TIMEFRAMES, default=["1d", "4h"])
        stop_loss = columns[2].number_input(
            "Stop loss",
            min_value=0.0,
            step=1.0,
            value=0.0,
            help="Needed to size the proposal. Without it the risk gate rejects for safety.",
        )

        st.markdown("**Bot integration**")
        bot_columns = st.columns([1, 2, 3])
        use_bot = bot_columns[0].checkbox(
            "Expect a bot",
            value=True,
            help=(
                "Name the bot that should be voting. If it has published nothing "
                "recent, the agents decide alone and say so explicitly."
            ),
        )
        bot_name = bot_columns[1].text_input("Bot name", value="sniper_bot").strip()
        bot_columns[2].caption(
            "Your bot publishes to `POST /signals`. Leave this on and the panel "
            "below will tell you whether it actually took part."
        )

        agents = st.multiselect("Agents", AGENTS[:-1], default=AGENTS[:-1])
        submitted = st.form_submit_button("Reach consensus", type="primary")

    if not submitted:
        _render_recent_signals()
        return
    if not symbol or not intervals:
        st.warning("Enter a symbol and at least one timeframe.")
        return

    payload: dict[str, Any] = {
        "symbol": symbol,
        "intervals": intervals,
        "agents": agents,
    }
    if use_bot and bot_name:
        payload["expected_bot"] = bot_name
    if stop_loss > 0:
        payload["stop_loss"] = stop_loss

    api = client()
    try:
        with st.spinner(f"Collecting {len(agents)} agent votes on {symbol}..."):
            result = api.consensus(**payload)
    except ApiError as exc:
        record_processing(api)
        show_error(exc)
        return
    finally:
        api.close()

    record_processing(api, result)
    render_consensus(result)


def render_consensus(result: dict[str, Any]) -> None:
    decision = result["decision"]

    # Bot availability first: it changes how the verdict below should be read.
    icon, renderer = BOT_STATUS_RENDER.get(result["bot_status"], ("•", st.info))
    renderer(f"{icon} {result['bot_note']}")

    label, render_action = ACTION_RENDER.get(decision["action"], ("HOLD", st.info))
    render_action(f"**{label}** — {decision['reason']}")

    columns = st.columns(4)
    columns[0].metric("Score", f"{decision['score']:+.2f}", help="Confidence-weighted direction")
    columns[1].metric("Threshold", f"±{decision['policy']['threshold']:.2f}")
    columns[2].metric("Voters", len(decision["votes"]))
    columns[3].metric("Agreement", f"{result['agreement']:.0%}")

    tally = result["tally"]
    st.caption(
        f"Tally — 🟢 {tally.get('bullish', 0)} bullish · "
        f"🔴 {tally.get('bearish', 0)} bearish · ⚪ {tally.get('neutral', 0)} neutral"
        + ("" if decision["quorum_met"] else " · quorum NOT met")
    )

    st.subheader("How each voter voted")
    st.dataframe(
        [
            {
                "": DIRECTION_ICON.get(vote["direction"], "⚪"),
                "voter": vote["voter"],
                "type": vote["kind"],
                "direction": vote["direction"],
                "confidence": f"{vote['confidence']:.0%}",
                "weight": round(
                    {"bullish": 1, "bearish": -1, "neutral": 0}[vote["direction"]]
                    * vote["confidence"],
                    3,
                ),
                "rationale": vote["rationale"][:160],
            }
            for vote in decision["votes"]
        ],
        **FILL_WIDTH,
        hide_index=True,
    )

    _render_risk(result)


def _render_risk(result: dict[str, Any]) -> None:
    risk = result.get("risk")
    if not risk:
        st.caption("No position proposed, so there was nothing for the risk gate to assess.")
        return

    st.subheader("Risk gate")
    proposal = risk["proposal"]
    st.write(
        f"Proposal: **{proposal['side']} {float(proposal['quantity']):g} {proposal['symbol']}** "
        f"at {float(proposal['entry_price']):,.2f}"
    )
    if risk["verdict"] == "approved":
        st.success("APPROVED — consensus and risk agree.")
    else:
        st.error("REJECTED — the gate overrides the vote.")
        for violation in risk["violations"]:
            st.markdown(f"- **{violation['rule']}** — {violation['detail']}")

    if result.get("executable"):
        st.info(
            "Ready to execute. Go to the **Trade** page to place it — execution is "
            "deliberately a separate, explicit step."
        )
    with st.expander("Risk metrics"):
        st.json(risk["metrics"])


def _render_recent_signals() -> None:
    """What the bots have been saying, so 'is it connected?' is answerable."""
    api = client()
    try:
        recent = api.signals(limit=10)
    except ApiError:
        api.close()
        return
    api.close()

    signals = recent.get("signals") or []
    st.subheader("Recent bot signals")
    if not signals:
        st.info(
            "No bot has published a signal yet. Point your bot at "
            "`POST /signals` — until then the agents decide alone."
        )
        return
    st.dataframe(
        [
            {
                "": DIRECTION_ICON.get(signal["direction"], "⚪"),
                "received": signal["received_at"][:19].replace("T", " "),
                "source": signal["source"],
                "symbol": signal["symbol"],
                "direction": signal["direction"],
                "confidence": f"{signal['confidence']:.0%}",
                "rationale": (signal.get("rationale") or "")[:120],
            }
            for signal in reversed(signals)
        ],
        **FILL_WIDTH,
        hide_index=True,
    )


def page_processing() -> None:
    st.header("Processing")
    st.caption(
        "How the last run was actually carried out — the request, the agents "
        "dispatched, their timings and outcomes, and the raw response."
    )
    render_processing()


def page_models() -> None:
    st.header("Models")
    st.caption(
        "Which language model the agents use. Anthropic is the highest quality; "
        "OpenRouter has a free tier; Ollama runs locally at no cost and sends "
        "nothing off the machine."
    )

    api = client()
    try:
        current = api.models()
    except ApiError as exc:
        show_error(exc)
        api.close()
        return

    active = current["active"]
    switching = current["switching_enabled"]
    columns = st.columns(2)
    columns[0].metric("Active provider", active["provider"])
    columns[1].metric("Active model", active["model"])
    if not switching:
        st.warning(
            "Runtime model switching is disabled on the server "
            "(ATP_LLM__ALLOW_RUNTIME_MODEL_SWITCHING=false)."
        )

    providers = current["providers"]
    provider = st.selectbox("Provider", providers, index=providers.index(active["provider"]))
    free_only = st.checkbox("Free models only", value=provider == "openrouter")

    try:
        listing = api.models(provider, free_only)
    except ApiError as exc:
        show_error(exc)
        api.close()
        return

    models = listing["models"]
    if not models:
        st.info(PROVIDER_HELP.get(provider, "No models returned by this provider."))
    else:
        st.dataframe(
            [
                {
                    "model": model["id"],
                    "name": model["name"],
                    "free": model["free"],
                    "installed": model["installed"],
                    "context": model["context_length"],
                }
                for model in models
            ],
            **FILL_WIDTH,
            hide_index=True,
        )

        choice = st.selectbox("Select a model", [model["id"] for model in models])
        selected = next(model for model in models if model["id"] == choice)
        needs_download = selected["installed"] is False

        actions = st.columns(2)
        if actions[0].button(
            "Use this model", type="primary", disabled=not switching or needs_download
        ):
            try:
                result = api.select_model(provider, choice)
                st.success(f"Agents now use {result['provider']} / {result['model']}")
                st.rerun()
            except ApiError as exc:
                show_error(exc)
        if needs_download:
            actions[0].caption("Download it first.")

        if provider == "ollama" and actions[1].button("Download", disabled=not needs_download):
            try:
                with st.spinner(f"Pulling {choice} - this can take several minutes..."):
                    pulled = api.pull_model(provider, choice)
                if pulled["installed"]:
                    st.success(f"{pulled['model']} is ready to use.")
                    st.rerun()
                else:
                    st.warning(f"Pull finished with status: {pulled.get('detail')}")
            except ApiError as exc:
                show_error(exc)

    if provider == "ollama":
        with st.expander("Download another model"):
            st.caption(
                "Any name from ollama.com/library, e.g. `llama3.2:3b` or `qwen2.5:14b`. "
                "Smaller models are faster but hold to the output schema less reliably."
            )
            name = st.text_input("Model name", key="ollama_pull_name").strip()
            if st.button("Download", key="ollama_pull_button", disabled=not name):
                try:
                    with st.spinner(f"Pulling {name} - this can take several minutes..."):
                        pulled = api.pull_model("ollama", name)
                    st.success(f"{pulled['model']}: {pulled.get('detail') or 'done'}")
                    st.rerun()
                except ApiError as exc:
                    show_error(exc)

    api.close()


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
        if st.session_state.get("processing"):
            with st.expander("⚙️ Processing — how the last run was done"):
                render_processing()
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
        record_processing(api)
        show_error(exc)
        return
    finally:
        api.close()

    record_processing(api, result)

    for failure in result.get("failures", []):
        st.warning(f"**{failure['agent']}** did not complete: {failure['error']}")

    st.caption("Timeframes analysed: " + ", ".join(result.get("intervals", [])))

    with st.expander("⚙️ Processing — how this run was done", expanded=False):
        render_processing()

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
                    **FILL_WIDTH,
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
                        **FILL_WIDTH,
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
                        **FILL_WIDTH,
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
                **FILL_WIDTH,
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
                **FILL_WIDTH,
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
            **FILL_WIDTH,
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
            **FILL_WIDTH,
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
    "Consensus": page_consensus,
    "Processing": page_processing,
    "Portfolio": page_portfolio,
    "Trade": page_trade,
    "Memory": page_memory,
    "Models": page_models,
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
