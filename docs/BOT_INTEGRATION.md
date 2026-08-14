# Connecting a trading bot

How an external bot — `python_signal_bot`, an MT5 EA, a TradingView alert — takes
part in the platform's decisions, and what happens when it does not.

---

## 1. The idea

The bot and the agent pool are both voters. Neither decides alone.

```
  sniper_bot ──── vote (direction × confidence) ───┐
                                                   ├──> voting policy ──> consensus
  6 specialist agents ──── one vote each ──────────┘         │
                                                             ▼
                                                        risk engine  ← veto, not a vote
                                                             │
                                                             ▼
                                                   proposal shown to you
```

Three properties worth being explicit about:

- **The policy is deterministic Python**, unit-tested, in
  `domain/services/voting.py`. Agents contribute votes; no model ever computes
  the tally. A consensus a model could talk itself into is not a control.
- **The risk engine is a veto, not a participant.** Voting answers "do we want
  this?"; risk answers "are we allowed?". A unanimous, maximally confident pool
  still cannot open a position that breaches exposure limits.
- **Nothing executes.** `/consensus` returns a proposal. Placing it is a
  separate call, on a separate scope, after a human decides.

## 2. The scoring rule

```
score = Σ(direction_i × confidence_i) / Σ(confidence_i)      direction: +1 / 0 / −1
```

Trade when `|score| ≥ threshold` **and** at least `min_voters` took part.
Anything else holds.

Dividing by *total confidence* rather than by voter count keeps the score in
`[-1, 1]` and comparable between runs. It also means the intuition from a
head-count is wrong, in a way worth internalising before tuning the threshold:

| Votes | Score | At threshold 0.35 |
|---|---|---|
| 3 bullish @0.8/0.65/0.7, 1 bearish @0.6 | **+0.56** | BUY |
| 2 bullish @0.5, 1 bearish @0.4, 1 neutral @1.0 | **+0.25** | HOLD |
| 2 bullish @0.2, 1 bearish @1.0 | **−0.43** | SELL (outvoted on conviction) |

A confident *neutral* is not an abstention — it dilutes. That is deliberate: five
shrugs should be able to stop one bull.

Configure with `ATP_VOTING__THRESHOLD`, `ATP_VOTING__MIN_VOTERS`,
`ATP_VOTING__REQUIRE_BOT_SIGNAL`, `ATP_VOTING__SIGNAL_TTL_SECONDS`.

## 3. When the bot is not there

Name the bot you expect — `ATP_VOTING__EXPECTED_BOT=sniper_bot`, or
`expected_bot` per request — and its absence becomes a reported status instead of
an unnoticed gap. **Silence and disagreement are different events**, and a
trader who thinks a mechanical system endorsed a trade when it was actually
offline is acting on evidence that does not exist.

| `bot_status` | Meaning | What you see |
|---|---|---|
| `voted` | Fresh signal counted | *sniper_bot voted bullish at 82% confidence.* |
| `stale` | Older than the TTL, excluded | *sniper_bot's last signal for XAUUSD is 45 minutes old (limit 15), so it was not counted. Deciding on the agents' own analysis alone.* |
| `missing` | Expected, never published | *sniper_bot is not available - it has published no signal for XAUUSD. Deciding on the agents' own analysis alone.* |
| `not_configured` | No bot expected | *No external bot is configured, so this verdict is the agents' alone.* |

In every case the agents still decide. Absence shrinks the pool; it does not
block the workflow. Note the quorum still applies — with `min_voters=3`, losing
the bot *and* two failed agents means a hold.

## 4. Two ways to integrate

### `POST /signals` — fire and forget (scope: `signal`)

For a bot on a fast cycle that should not block on an LLM workflow. The signal is
stored and votes in any consensus run within its TTL.

```bash
curl -X POST http://127.0.0.1:8000/signals \
  -H "authorization: Bearer $ATP_SIGNAL_KEY" \
  -H "content-type: application/json" \
  -d '{"symbol":"XAUUSD","direction":"BUY","confidence":0.82,
       "source":"sniper_bot","rationale":"EMA cross + H4 range breakout",
       "strategy":"ema_cross"}'
```

### `POST /consensus` — ask and get an answer (scope: `read`)

For a bot that wants the verdict back so it can tell its trader. Submit the vote
inline and the whole exchange is one call.

```bash
curl -X POST http://127.0.0.1:8000/consensus \
  -H "content-type: application/json" \
  -d '{"symbol":"XAUUSD","intervals":["1d","4h"],
       "expected_bot":"sniper_bot","stop_loss":4150,
       "signal":{"symbol":"XAUUSD","direction":"BUY","confidence":0.82,
                 "source":"sniper_bot","rationale":"EMA cross"}}'
```

**Direction vocabulary is flexible.** `BUY`/`SELL`/`NEUTRAL`,
`bullish`/`bearish`/`neutral`, `long`/`short`, `up`/`down`, `hold`/`flat` are all
accepted and normalized — so `python_signal_bot` can forward
`TradeDecision.direction` unchanged.

> **Latency.** A consensus run is one LLM call per agent — 15–60s on free
> OpenRouter models. Call it only when your bot has something actionable
> (`threshold_met`), not every cycle. Otherwise you will exhaust the free quota
> and add a minute to a loop that runs in seconds.

## 5. Drop-in client for `python_signal_bot`

Save as `src/atp_consensus.py`. It maps `TradeDecision` onto the API and formats
a message for `TelegramSignalNotifier.send_text`.

```python
"""Ask the ATP agent pool to vote on this bot's decision, and report back."""

from __future__ import annotations

import os
from typing import Any

import requests

ATP_URL = os.getenv("ATP_URL", "http://127.0.0.1:8000")
ATP_KEY = os.getenv("ATP_API_KEY", "")
BOT_NAME = os.getenv("ATP_BOT_NAME", "sniper_bot")
TIMEOUT = float(os.getenv("ATP_TIMEOUT", "300"))

_ICON = {"buy": "🟢", "sell": "🔴", "hold": "⚪"}


def ask_consensus(
    symbol: str,
    decision: Any,          # src.models.TradeDecision
    *,
    timeframes: list[str] | None = None,
    stop_loss: float | None = None,
) -> dict[str, Any] | None:
    """Return the combined verdict, or None if ATP is unreachable.

    Never raises: the agent pool is an enhancement to this bot, and losing it
    must not take the bot down with it.
    """
    payload: dict[str, Any] = {
        "symbol": symbol,
        "intervals": timeframes or ["1d", "4h"],
        "expected_bot": BOT_NAME,
        "signal": {
            "symbol": symbol,
            "direction": decision.direction,      # BUY / SELL / NEUTRAL, sent as-is
            "confidence": max(0.0, min(1.0, float(decision.confidence))),
            "source": BOT_NAME,
            "rationale": (decision.summary or "")[:2000],
        },
    }
    if stop_loss:
        payload["stop_loss"] = float(stop_loss)

    headers = {"Authorization": f"Bearer {ATP_KEY}"} if ATP_KEY else {}
    try:
        response = requests.post(
            f"{ATP_URL}/consensus", json=payload, headers=headers, timeout=TIMEOUT
        )
        response.raise_for_status()
        return dict(response.json())
    except requests.RequestException as exc:
        print(f"[atp] consensus unavailable: {exc}")
        return None


def format_for_trader(result: dict[str, Any] | None, symbol: str) -> str:
    """A Telegram-ready summary. Says plainly when the agents decided alone."""
    if result is None:
        return f"⚠️ {symbol}: ATP agents unreachable — this bot's own signal stands alone."

    decision = result["decision"]
    action = decision["action"]
    lines = [
        f"{_ICON.get(action, '•')} <b>{symbol} — {action.upper()}</b>",
        f"Score {decision['score']:+.2f} (needs ±{decision['policy']['threshold']:.2f}), "
        f"{len(decision['votes'])} voters, {result['agreement']:.0%} agreement",
        "",
        result["bot_note"],
        "",
        "<b>Votes</b>",
    ]
    for vote in decision["votes"]:
        mark = {"bullish": "🟢", "bearish": "🔴"}.get(vote["direction"], "⚪")
        lines.append(f"{mark} {vote['voter']} ({vote['kind']}) {vote['confidence']:.0%}")

    if risk := result.get("risk"):
        if risk["verdict"] == "approved":
            proposal = risk["proposal"]
            lines += [
                "",
                f"✅ Risk gate approved: {proposal['side']} "
                f"{float(proposal['quantity']):g} @ {float(proposal['entry_price']):,.2f}",
            ]
        else:
            reasons = ", ".join(v["rule"] for v in risk["violations"])
            lines += ["", f"⛔ Risk gate rejected: {reasons}"]

    lines += ["", f"<i>{decision['reason']}</i>"]
    return "\n".join(lines)
```

Wire it in where the bot already decides — only when it has something to say:

```python
from src.atp_consensus import ask_consensus, format_for_trader

if snapshot.decision.threshold_met:
    result = ask_consensus(
        snapshot.symbol,
        snapshot.decision,
        stop_loss=snapshot.risk_plan.stop_loss,
    )
    notifier.send_text(format_for_trader(result, snapshot.symbol))
```

`ask_consensus` returning `None` is handled, not fatal — if ATP is down the bot
carries on exactly as it does today and the trader is told so.

## 6. Credentials

Give the bot a `signal`-scoped key, not a `trade` one. It is the credential most
likely to sit on a remote box, and it has no business executing:

```ini
ATP_API__KEYS=[
  {"name":"dashboard","key":"...","scopes":["read"]},
  {"name":"sniper_bot","key":"...","scopes":["signal","read"]},
  {"name":"trader","key":"...","scopes":["read","trade"]}
]
```

## 7. From the dashboard

The **Consensus** page does all of the above interactively: symbol, timeframes,
an **Expect a bot** toggle with its name, and a stop loss. It shows the bot
status banner first — because it changes how the verdict below should be read —
then the verdict, the per-voter table with signed weights, and the risk gate's
answer. Recent bot signals are listed underneath so "is it actually connected?"
is answerable at a glance.

## 8. Endpoints

| Endpoint | Scope | Purpose |
|---|---|---|
| `POST /signals` | `signal` | Publish a bot's opinion |
| `GET /signals` | `read` | Audit trail of what bots have claimed |
| `POST /consensus` | `read` | Run the agents, count the votes, return the verdict |
| `POST /trade` | `trade` | Execute — separate step, separate scope |
