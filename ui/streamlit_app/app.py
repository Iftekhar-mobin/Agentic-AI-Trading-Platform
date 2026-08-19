"""Streamlit dashboard for the Agentic AI Trading Platform.

An internal operator console, not a trading terminal. The thing it is built to
show is *why* — every agent conclusion is rendered with its evidence,
confidence and invalidation conditions, because a recommendation you cannot
interrogate is one you should not act on.

One command is enough — the sidebar can start the API itself when nothing is
answering on the configured address:

    uv run streamlit run ui/streamlit_app/app.py

Running `uv run atp serve` separately still works, and is what you want when you
need the server's log in front of you.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from client import DEFAULT_BASE_URL, MAX_BODY_CHARS, ApiError, AtpClient
from server import ApiServer, parse_target


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

# Vendor names as a person would say them, not as the API spells them. A
# comment rather than an attribute docstring: Streamlit's "magic" renders every
# bare expression at module level, and a docstring is one.
PROVIDER_LABEL = {
    "anthropic": "Anthropic",
    "openrouter": "OpenRouter",
    "ollama": "Ollama (local)",
}

st.set_page_config(page_title="ATP Console", page_icon="📈", layout="wide")


def client() -> AtpClient:
    return AtpClient(
        st.session_state.get("base_url", DEFAULT_BASE_URL),
        st.session_state.get("api_key", ""),
    )


def model_line(provider: str | None, model: str | None) -> str:
    """Vendor and exact model id, in one line, spelled the same everywhere.

    Every place the console attributes reasoning goes through here. The model
    id is never shortened: `nemotron-3-super-120b-a12b:free` and its paid twin
    differ by a suffix, and a console that hides the suffix is the reason
    someone spends an afternoon wondering why quality dropped.
    """
    label = PROVIDER_LABEL.get(provider or "", provider or "unknown vendor")
    return f"🧠 **{label}** · `{model or 'no model'}`"


def render_model_badge(provider: str | None, model: str | None, *, suffix: str = "") -> None:
    st.caption(model_line(provider, model) + suffix)


def llm_attribution() -> dict[str, dict[str, Any]]:
    """Agent name -> the LLM call it made, for the run currently on screen.

    Built from the trace the API returns rather than from the active model, so
    a report rendered after someone switched models is still credited to the
    backend that actually wrote it.
    """
    calls = st.session_state.get("llm_calls") or []
    return {call["agent"]: call for call in calls if call.get("agent")}


def render_agent_model(agent: str | None) -> None:
    """Say which model produced the report being read, right where it is read."""
    if not agent:
        return
    call = llm_attribution().get(agent)
    if call is None:
        # No trace for this agent: an older server, or a report served from a
        # path that made no LLM call. Claiming the active model here would be a
        # guess, so say nothing rather than something possibly untrue.
        return
    seconds = f" · {call.get('duration_ms', 0) / 1000:.1f}s"
    failed = " · ❌ the call failed" if call.get("status") == "failed" else ""
    render_model_badge(call.get("provider"), call.get("model"), suffix=seconds + failed)


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


def section(label: str, *, nested: bool = False, expanded: bool = False) -> Any:
    """A collapsible box — or a plain labelled panel when already inside one.

    Streamlit refuses to nest expanders, and the screener's per-symbol row *is*
    an expander. Rather than keep a second copy of every report renderer for
    that page, all of them open their boxes through here: at the top level it
    collapses, one level down it is simply an open panel.
    """
    if not nested:
        return st.expander(label, expanded=expanded)
    st.markdown(f"**{label}**")
    return st.container(border=True)


def render_assessment(
    title: str,
    report: dict[str, Any],
    key: str = "assessment",
    *,
    nested: bool = False,
    agent: str | None = None,
) -> None:
    assessment = report[key]
    direction = assessment.get("direction", "neutral")
    label = f"{DIRECTION_ICON.get(direction, '⚪')} {title} — {direction}"
    with section(label, nested=nested, expanded=True):
        # Whose judgement this is, before the judgement itself: the same
        # reasoning from a 120B free model and from Opus deserves different
        # weight, and the reader can only apply that if they are told which.
        render_agent_model(agent)
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
    # Kept beside it so every report renderer can name the model that wrote it
    # without being handed the whole response.
    st.session_state["llm_calls"] = (result or {}).get("llm_calls") or []


def render_llm_calls(calls: list[dict[str, Any]]) -> None:
    """Every language-model round-trip the run made, with vendor and model.

    The Model metric above says what is configured; this says what answered.
    They are normally the same, and the times they are not - a model switched
    while a screen was running, a provider that failed over - are precisely the
    runs someone comes to this page to understand.
    """
    st.subheader("LLM calls")
    if not calls:
        st.caption(
            "No language-model calls recorded for this run. Either nothing "
            "reached an agent, or the server predates per-call reporting."
        )
        return

    vendors = sorted({model_line(call.get("provider"), call.get("model")) for call in calls})
    for vendor in vendors:
        st.markdown(vendor)
    if len(vendors) > 1:
        st.warning(
            "More than one model served this run - the reports below were not "
            "all written by the same backend."
        )

    st.dataframe(
        [
            {
                "": STATUS_ICON.get(call.get("status", "ok"), "•"),
                "agent": call.get("agent") or "—",
                "vendor": PROVIDER_LABEL.get(call.get("provider", ""), call.get("provider", "—")),
                "model": call.get("model", "—"),
                "schema": call.get("response_model", ""),
                "seconds": round(call.get("duration_ms", 0) / 1000, 2),
                "detail": call.get("detail", ""),
            }
            for call in calls
        ],
        **FILL_WIDTH,
        hide_index=True,
    )
    total = sum(call.get("duration_ms", 0) for call in calls)
    st.caption(
        f"{len(calls)} calls, {total / 1000:.1f}s of model time in total. The "
        "analysis agents run concurrently, so this exceeds the wall clock."
    )


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
    calls: list[dict[str, Any]] = result.get("llm_calls") or []
    failed = [step for step in steps if step["status"] == "failed"]
    columns = st.columns(4)
    columns[0].metric("Total time", f"{result.get('duration_ms', 0) / 1000:.1f}s")
    columns[1].metric("Agents run", len(steps))
    columns[2].metric("Failed", len(failed), delta=None if not failed else f"-{len(failed)}")
    columns[3].metric("LLM calls", len(calls))
    if model:
        st.info(model_line(model.get("provider"), model.get("model")) + " produced this reasoning")

    render_llm_calls(calls)

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


def render_consensus(result: dict[str, Any], *, nested: bool = False) -> None:
    """The verdict, the voters, and what the risk gate made of it.

    Reused verbatim for each row of a screen: a shortlisted symbol is read
    exactly the way a single-symbol consensus is."""
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

    _render_risk(result, nested=nested)


def _render_risk(result: dict[str, Any], *, nested: bool = False) -> None:
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
    with section("Risk metrics", nested=nested):
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
    st.session_state["active_model"] = active

    # The banner, not just two metrics: this is the page where the answer is
    # changed, so it is the page that has to be unambiguous about what the
    # answer currently is.
    st.success(model_line(active["provider"], active["model"]) + " — in use by every agent")
    if switched := st.session_state.pop("model_switched", None):
        st.toast(f"Now using {switched}", icon="✅")
    columns = st.columns(2)
    columns[0].metric("Active provider", PROVIDER_LABEL.get(active["provider"], active["provider"]))
    columns[1].metric("Active model", active["model"])
    if not switching:
        st.warning(
            "Runtime model switching is disabled on the server "
            "(ATP_LLM__ALLOW_RUNTIME_MODEL_SWITCHING=false)."
        )

    providers = current["providers"]
    # Defaults to the active provider, but remembers where you browsed to:
    # comparing Ollama's catalog should not reset every time the page reruns.
    browsing = st.session_state.get("models_provider", active["provider"])
    if browsing not in providers:
        browsing = active["provider"]
    provider = st.selectbox("Provider", providers, index=providers.index(browsing))
    st.session_state["models_provider"] = provider
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

        def is_active(model: dict[str, Any]) -> bool:
            return provider == active["provider"] and model["id"] == active["model"]

        st.dataframe(
            [
                {
                    # The catalog can run to hundreds of rows; the one in use
                    # should be findable without reading the id of each.
                    "": "✅" if is_active(model) else "",
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

        ids = [model["id"] for model in models]
        # Preselect what is running, so the button under it never offers to
        # switch to something the reader did not choose.
        default = active["model"] if active["model"] in ids else ids[0]
        choice = st.selectbox("Select a model", ids, index=ids.index(default))
        if provider == active["provider"] and choice == active["model"]:
            st.caption("✅ This is the model the agents are using now.")
        else:
            st.caption(
                f"Selecting this replaces {model_line(active['provider'], active['model'])} "
                "for every agent, from the next call onwards."
            )
        selected = next(model for model in models if model["id"] == choice)
        needs_download = selected["installed"] is False

        actions = st.columns(2)
        if actions[0].button(
            "Use this model", type="primary", disabled=not switching or needs_download
        ):
            try:
                result = api.select_model(provider, choice)
            except ApiError as exc:
                show_error(exc)
            else:
                # Survives the rerun that follows: the banner and the sidebar
                # both re-read the server, and this says what just changed.
                st.session_state["active_model"] = result
                st.session_state["model_switched"] = (
                    f"{PROVIDER_LABEL.get(result['provider'], result['provider'])} / "
                    f"{result['model']}"
                )
                st.rerun()
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

    render_reports(result)


def render_reports(result: dict[str, Any], *, nested: bool = False) -> None:
    """Every agent report the run produced, in reading order.

    Shared by the Analysis page and the screener's per-symbol detail: the same
    conclusions deserve the same rendering wherever they are read, and a second
    copy of this would drift within a milestone. ``nested`` says whether there
    is already an expander above us — see ``section``.
    """
    if technical := result.get("technical_report"):
        render_assessment("Technical", technical, nested=nested, agent="technical_analysis")
        st.info(f"**Timeframe alignment.** {technical['assessment']['timeframe_alignment']}")
        render_timeframe_directions(technical["timeframes"])
        for frame in technical["timeframes"]:
            with section(f"Indicator readings - {frame['interval']}", nested=nested):
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
        render_assessment("Chart patterns", patterns, nested=nested, agent="chart_pattern")
        st.info(f"**Timeframe alignment.** {patterns['assessment']['timeframe_alignment']}")
        render_timeframe_directions(patterns["timeframes"])
        for frame in patterns["timeframes"]:
            label = (
                f"{frame['interval']} - {len(frame['patterns'])} patterns, "
                f"{len(frame['levels'])} levels"
            )
            with section(label, nested=nested):
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
        render_assessment(
            f"Market research vs {research['benchmark']}",
            research,
            nested=nested,
            agent="market_research",
        )
        with section("Relative measurements", nested=nested):
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
        render_assessment("Fundamentals", fundamental, nested=nested, agent="fundamental_analysis")
        with section("Metric readings", nested=nested):
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
        render_assessment("News", news, nested=nested, agent="news_analysis")
        st.markdown("**Themes:** " + ", ".join(news["assessment"]["key_themes"]))
        with section(f"{len(news['articles'])} articles", nested=nested):
            for article in news["articles"]:
                st.markdown(
                    f"- [{article['title']}]({article['url'] or '#'}) — "
                    f"{article['publisher'] or 'unknown'}, {article['published_at'][:16]}"
                )

    if sentiment := result.get("sentiment_report"):
        render_assessment("Sentiment", sentiment, nested=nested, agent="sentiment_analysis")
        summary = sentiment["summary"]
        columns = st.columns(4)
        columns[0].metric("Weighted polarity", f"{summary['weighted_polarity']:+.2f}")
        columns[1].metric("Articles", summary["article_count"])
        columns[2].metric("Aggregate", summary["direction"])
        columns[3].metric("Classifier", sentiment["model_name"])

    if learning := result.get("learning_report"):
        render_assessment("Learning", learning, "entry", nested=nested, agent="continuous_learning")
        if regime := learning.get("regime"):
            st.info(f"Market regime: **{regime['trend']} / {regime['volatility']}**")
        st.markdown("**Lessons**")
        for lesson in learning["entry"]["lessons"]:
            st.markdown(f"- {lesson}")
        st.caption(learning["entry"]["regime_note"])
        if recalled := learning.get("recalled"):
            with section(f"{len(recalled)} recalled episodes", nested=nested):
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


CLASS_ICON = {
    "forex": "💱",
    "stock": "🏢",
    "index": "📊",
    "commodity": "🛢️",
    "crypto": "🪙",
}

ACTION_BADGE = {"buy": "🟢 BUY", "sell": "🔴 SELL", "hold": "⚪ HOLD"}


def _universe() -> dict[str, Any] | None:
    """The instrument menu, fetched once per session.

    Streamlit reruns the whole script on every click, and the menu does not
    change between them; refetching would add a request to the trace for each
    ticked box and make the Processing panel useless.
    """
    if cached := st.session_state.get("universe"):
        return cached if isinstance(cached, dict) else None

    api = client()
    try:
        universe = api.universe()
    except ApiError as exc:
        show_error(exc)
        return None
    finally:
        api.close()

    st.session_state["universe"] = universe
    return universe


def _basket(universe: dict[str, Any]) -> list[str]:
    """The selection widgets, and the symbols they currently hold.

    Outside a form on purpose: the cost estimate below has to move as boxes are
    ticked, and a form would only tell you what a screen would cost after you
    had already asked for it.
    """
    picked: list[str] = []
    defaults: list[str] = universe["default_selection"]
    columns = st.columns(2)

    for index, category in enumerate(universe["categories"]):
        assets = category["assets"]
        labels = {asset["symbol"]: f"{asset['symbol']} · {asset['name']}" for asset in assets}
        icon = CLASS_ICON.get(category["asset_class"], "•")
        with columns[index % 2]:
            picked.extend(
                st.multiselect(
                    f"{icon} {category['label']} ({len(assets)})",
                    options=list(labels),
                    default=[symbol for symbol in defaults if symbol in labels],
                    format_func=lambda symbol, labels=labels: labels[symbol],  # type: ignore[misc]
                    key=f"basket_{category['asset_class']}",
                )
            )

    custom = st.text_input(
        "Anything else (comma separated)",
        placeholder="SHEL, ^HSI, DOGEUSD",
        help="Not limited to the menu. Unlisted tickers are screened and ranked "
        "the same way; they just have no curated name.",
    )
    for entry in custom.split(","):
        symbol = entry.strip().upper()
        if symbol and symbol not in picked:
            picked.append(symbol)
    return picked


def page_opportunities() -> None:
    st.header("Opportunities")
    st.caption(
        "Pick instruments from any market, run the full agent pool on each, and let "
        "the ranking agent say which few are worth your attention. The shortlist is "
        "comparative — it says which evidence here is strongest, never what will pay."
    )

    universe = _universe()
    if universe is None:
        return

    st.subheader("1. Choose a basket")
    symbols = _basket(universe)

    st.subheader("2. How to analyse them")
    columns = st.columns([3, 2, 2])
    intervals = columns[0].multiselect("Timeframes", TIMEFRAMES, default=["1d", "4h"])
    top_n = int(columns[1].number_input("Shortlist size", min_value=1, max_value=25, value=5))
    bot_name = columns[2].text_input(
        "Expect a bot (optional)",
        value="",
        help="Name a bot and its silence is reported per symbol instead of passing unnoticed.",
    )
    agents = st.multiselect("Voting agents", AGENTS[:-1], default=AGENTS[:-1])

    maximum = universe["max_symbols"]
    calls = len(symbols) * len(agents) + 1
    over = len(symbols) > maximum
    st.caption(
        f"**{len(symbols)} symbols x {len(agents)} agents = ~{calls} LLM calls.** "
        f"Server cap is {maximum} symbols per screen; a few run at a time, so expect "
        "minutes rather than seconds."
    )
    if over:
        st.warning(
            f"{len(symbols)} selected but only the first {maximum} would be screened. "
            "Trim the basket so nothing is silently dropped."
        )

    blocked = not symbols or not intervals or not agents or over
    if not st.button("Screen and rank", type="primary", disabled=blocked):
        if not symbols:
            st.info("Tick at least one instrument above.")
        _render_screen(st.session_state.get("screen"))
        return

    api = client()
    payload: dict[str, Any] = {
        "symbols": symbols,
        "intervals": intervals,
        "agents": agents,
        "top_n": top_n,
    }
    if bot_name.strip():
        payload["expected_bot"] = bot_name.strip()

    try:
        with st.spinner(f"Screening {len(symbols)} instruments — this is the slow one..."):
            result = api.screen(**payload)
    except ApiError as exc:
        record_processing(api)
        show_error(exc)
        return
    finally:
        api.close()

    record_processing(api)
    st.session_state["screen"] = result
    _render_screen(result)


def _render_screen(result: dict[str, Any] | None) -> None:
    if not result:
        return

    ranking = result["ranking"]
    outcomes = {outcome["symbol"]: outcome for outcome in result["outcomes"]}
    ranked = ranking["ranked"]

    st.subheader("3. The shortlist")
    columns = st.columns(4)
    columns[0].metric("Screened", len(outcomes))
    columns[1].metric("Shortlisted", len(ranked))
    columns[2].metric("Failed", len(result.get("failures") or []))
    columns[3].metric("Took", f"{result.get('duration_ms', 0) / 1000:.0f}s")

    for failure in result.get("failures") or []:
        st.warning(f"**{failure['symbol']}** could not be screened — {failure['error']}")

    if not ranking.get("ranked_by_agent", True):
        st.warning(
            "The ranking agent did not answer, so this order is the deterministic "
            "composite alone — no comparative judgement has been applied."
        )
    if narrative := ranking.get("narrative"):
        st.info(f"**The basket as a whole.** {narrative}")

    if not ranked:
        st.caption("Nothing to rank.")
        return

    st.dataframe(
        [_leaderboard_row(row) for row in ranked],
        **FILL_WIDTH,
        hide_index=True,
    )
    st.caption(
        "Composite = 0.40·conviction + 0.25·agreement + 0.20·confidence + 0.15·coverage, "
        "then reduced if quorum was missed or the risk gate refused. Potential is the "
        "ranking agent's comparative read, relative to this basket only."
    )

    st.subheader("4. Why — click any row")
    for row in ranked:
        _render_opportunity(row, outcomes.get(row["candidate"]["symbol"]))

    shortlisted = {row["candidate"]["symbol"] for row in ranked}
    rest = [outcome for symbol, outcome in outcomes.items() if symbol not in shortlisted]
    if rest:
        st.subheader(f"Also screened ({len(rest)})")
        st.caption("Everything that did not make the shortlist, with its full analysis.")
        for outcome in rest:
            with st.expander(
                f"{ACTION_BADGE.get(outcome['decision']['action'], '⚪')} {outcome['symbol']} — "
                f"score {outcome['decision']['score']:+.2f}"
            ):
                render_consensus(outcome, nested=True)
                render_reports(outcome["reports"], nested=True)


def _leaderboard_row(row: dict[str, Any]) -> dict[str, Any]:
    candidate = row["candidate"]
    verdict = row.get("verdict") or {}
    components = candidate["components"]
    return {
        "#": row["rank"],
        "": CLASS_ICON.get(candidate["asset_class"], "•"),
        "symbol": candidate["symbol"],
        "name": candidate["name"],
        "action": ACTION_BADGE.get(candidate["action"], candidate["action"]),
        "potential": f"{verdict['profit_potential']:.0%}" if verdict else "—",
        "composite": round(components["composite"], 3),
        "score": f"{candidate['score']:+.2f}",
        "agreement": f"{candidate['agreement']:.0%}",
        "voters": len(candidate["votes"]),
        "risk": candidate.get("risk_verdict") or "not proposed",
        "driver": verdict.get("key_driver", "—"),
    }


def _render_opportunity(row: dict[str, Any], outcome: dict[str, Any] | None) -> None:
    """One shortlisted instrument: the ranker's case, then the whole analysis."""
    candidate = row["candidate"]
    verdict = row.get("verdict")
    icon = CLASS_ICON.get(candidate["asset_class"], "•")
    potential = f" · potential {verdict['profit_potential']:.0%}" if verdict else ""
    label = (
        f"#{row['rank']}  {icon} {candidate['symbol']} — {candidate['name']} · "
        f"{ACTION_BADGE.get(candidate['action'], candidate['action'])}{potential}"
    )

    with st.expander(label, expanded=row["rank"] == 1):
        if verdict:
            confidence_bar("Ranking", verdict["confidence"])
            st.markdown(f"**Why it ranks here.** {verdict['reasoning']}")
            columns = st.columns(3)
            columns[0].markdown(f"**Key driver**\n\n{verdict['key_driver']}")
            columns[1].markdown(f"**Primary risk**\n\n{verdict['primary_risk']}")
            columns[2].markdown(f"**Horizon**\n\n{verdict['horizon']}")
            st.markdown("**Evidence**")
            for item in verdict.get("evidence", []):
                st.markdown(f"- `{item['source']}` — {item['statement']}")
            st.markdown("**This ranking is wrong if**")
            for condition in verdict.get("invalidation_conditions", []):
                st.markdown(f"- {condition}")
        else:
            st.caption(
                "The ranking agent did not write this one up; it is placed by the "
                "deterministic composite alone."
            )

        components = candidate["components"]
        st.markdown("**How the composite was reached**")
        columns = st.columns(6)
        for column, name in zip(
            columns,
            ("conviction", "agreement", "confidence", "coverage"),
            strict=False,
        ):
            column.metric(name.title(), f"{components[name]:.2f}")
        penalty = components["quorum_penalty"] * components["risk_penalty"]
        columns[4].metric("Penalties", f"x{penalty:.2f}")
        columns[5].metric("Composite", f"{components['composite']:.3f}")

        if candidate.get("highlights"):
            for highlight in candidate["highlights"]:
                st.caption(f"• {highlight}")

        if outcome is None:
            st.caption("The underlying analysis is unavailable for this symbol.")
            return

        st.divider()
        render_consensus(outcome, nested=True)
        st.divider()
        render_reports(outcome["reports"], nested=True)


PAGES: list[tuple[str, str, Callable[[], None]]] = [
    # (title, icon, renderer) — the order is the order of the top bar, and the
    # first one is where the console opens.
    ("Opportunities", ":material/leaderboard:", page_opportunities),
    ("Analysis", ":material/analytics:", page_analysis),
    ("Consensus", ":material/how_to_vote:", page_consensus),
    ("Processing", ":material/manage_search:", page_processing),
    ("Portfolio", ":material/account_balance_wallet:", page_portfolio),
    ("Trade", ":material/swap_horiz:", page_trade),
    ("Memory", ":material/psychology:", page_memory),
    ("Models", ":material/smart_toy:", page_models),
]


def navigate() -> None:
    """Top-bar navigation, one page per selection.

    ``st.tabs`` would look the part and be the wrong mechanism: it renders every
    tab's body on every run, so merely opening the console would fire a
    portfolio fetch, an order fetch and a memory query for pages nobody looked
    at. ``st.navigation`` runs only the page you are on and puts it in the URL,
    so a page can be linked to and the back button behaves.

    Session state is shared across pages, which is what lets a screen survive
    stepping over to Processing to see how it was run.
    """
    if not hasattr(st, "navigation"):  # Streamlit older than the pinned 1.60
        titles = {title: render for title, _, render in PAGES}
        st.sidebar.divider()
        titles[st.sidebar.radio("Page", list(titles))]()
        return

    st.navigation(
        [
            st.Page(render, title=title, icon=icon, url_path=_url_path(title), default=index == 0)
            for index, (title, icon, render) in enumerate(PAGES)
        ],
        position="top",
    ).run()


def _url_path(title: str) -> str:
    return title.lower().replace(" ", "-")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_LOG = PROJECT_ROOT / "data" / "api-server.log"
# Seconds to wait for the API to answer after starting it. Cold imports —
# pandas, and torch when FinBERT is selected — dominate this, not the server.
# A comment rather than an attribute docstring on purpose: Streamlit's "magic"
# renders every bare expression at module level, and a docstring is one.
STARTUP_TIMEOUT = 40.0


@st.cache_resource
def api_server() -> ApiServer:
    """One supervised server, shared by every browser session.

    ``cache_resource`` rather than session state: two open tabs must see the
    same process, or the second one would happily start a rival server and both
    would then fight over the port.
    """
    return ApiServer(PROJECT_ROOT, SERVER_LOG)


def _api_is_up() -> bool:
    api = client()
    try:
        api.health()
    except ApiError:
        return False
    else:
        return True
    finally:
        api.close()


def server_controls(*, reachable: bool) -> None:
    """Start and stop the API without leaving the console.

    Only offered for a local address: a base URL pointing at another machine is
    someone else's server, and a button that pretends otherwise would start a
    process that answers nothing the console is looking at.
    """
    server = api_server()
    target = parse_target(st.session_state.get("base_url", DEFAULT_BASE_URL))

    if not target.is_local:
        st.sidebar.caption(f"`{target.host}` is remote — start it there.")
        return

    if reachable:
        if server.managed:
            st.sidebar.caption(f"Started here · pid {server.pid}")
            if st.sidebar.button("Stop the API", **FILL_WIDTH):
                server.stop()
                st.rerun()
        else:
            # Someone else's process. We can talk to it; we do not own it.
            st.sidebar.caption("Started outside the console.")
        return

    if st.sidebar.button("Start the API", type="primary", **FILL_WIDTH):
        server.start(target)
        deadline = time.monotonic() + STARTUP_TIMEOUT
        with st.spinner(f"Starting the API on {target.host}:{target.port}..."):
            while time.monotonic() < deadline:
                if server.exit_code is not None:
                    break  # It died — fall through and show the log.
                if _api_is_up():
                    st.rerun()
                time.sleep(0.5)
        st.rerun()

    if server.exit_code is not None:
        st.sidebar.error(f"The server exited with code {server.exit_code}.")
    if tail := server.log_tail():
        with st.sidebar.expander("Server log"):
            st.code(tail, language="text")


def render_active_model(active: dict[str, Any] | None) -> None:
    """The model every page is about to use, kept on screen at all times.

    Read from `/health` rather than `/models`, which lists a provider's whole
    catalog over the network. This is the answer to "which model am I running?"
    and it should not cost a round-trip to openrouter.ai to get it.

    It also closes the loop on the Models page: after switching, the sidebar
    shows the new model on every page and after every rerun, so a selection
    made once stays visible rather than living only in the toast that
    announced it.
    """
    if not active:
        st.sidebar.caption("🧠 No model reported by this server.")
        return
    provider, model = active.get("provider"), active.get("model")
    previous = st.session_state.get("active_model")
    st.session_state["active_model"] = active

    with st.sidebar.container(border=True):
        st.markdown("**Agents are using**")
        st.markdown(model_line(provider, model))
        if previous and previous != active:
            label = PROVIDER_LABEL.get(previous.get("provider", ""), previous.get("provider"))
            st.caption(f"Switched from `{previous.get('model')}` ({label}).")


def sidebar() -> None:
    """Connection, server control, and nothing else — navigation is up top."""
    st.sidebar.title("📈 ATP Console")
    st.session_state.setdefault("base_url", DEFAULT_BASE_URL)
    st.session_state.setdefault("api_key", "")

    st.sidebar.text_input("API base URL", key="base_url")
    st.sidebar.text_input("API key", key="api_key", type="password")

    api = client()
    reachable = True
    try:
        health = api.health()
        st.sidebar.success(f"API {health['version']} · {health['status']}")
        render_active_model(health.get("active_model"))
    except ApiError as exc:
        reachable = False
        st.sidebar.error(f"API unreachable\n\n{exc}")
    finally:
        api.close()

    server_controls(reachable=reachable)
    st.sidebar.caption("Decision support only. Recommendations are explainable, not guaranteed.")


def main() -> None:
    sidebar()
    navigate()


main()
