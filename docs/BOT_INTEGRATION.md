# Connecting a trading bot

How an external bot — `python_signal_bot`, an MT5 EA, a TradingView alert —
takes part in the platform's decisions, and what happens when it does not.

---

## 1. Two shapes, and they are not the same

There are two genuinely different ways to connect a bot, and choosing the wrong
one produces a number that looks meaningful and is not.

**As a voting member.** The bot's opinion joins the agent pool and a
deterministic policy counts them all together. One combined verdict comes out.

```
  bot ──────── vote (direction × confidence) ───┐
                                                ├──> voting policy ──> consensus
  specialist agents ──── one vote each ─────────┘         │
                                                          ▼
                                                     risk engine  ← veto, not a vote
```

**As an independent reviewer.** The bot keeps its own decision, asks the agents
to reach *theirs* without being told what the bot concluded, and compares the
two afterwards. The bot's vote is deliberately absent from the tally.

```
  bot decides ─────────────────────────────────┐
                                               ├──> agree? ──> place / skip
  agents vote among themselves ──> verdict ────┘
```

The second is what you want for an approval gate, and it has one rule that is
easy to get wrong: **the bot must not publish a signal at all.** See §4.

Three properties hold either way:

- **The policy is deterministic Python**, unit-tested, in
  `domain/services/voting.py`. Agents contribute votes; no model ever computes
  the tally. A consensus a model could talk itself into is not a control.
- **The risk engine is a veto, not a participant.** Voting answers "do we want
  this?"; risk answers "are we allowed?". A unanimous, maximally confident pool
  still cannot open a position that breaches exposure limits.
- **Nothing executes.** `/consensus` returns a proposal. Placing it is a
  separate call, on a separate scope, after something else decides.

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

### The rule behaves differently in a small pool

Confidence-weighting is well behaved across six or seven voters and sharp-edged
across two. With a two-agent pool, technical bullish @0.80 against chart_pattern
**bearish** @0.30 scores `(0.80 − 0.30) / 1.10 = +0.45`, clears the default
`0.35` threshold, and returns **buy** — while one of your two voters actively
disagreed. Nothing is broken; the arithmetic is doing what it says. But a caller
running a small pool should check `agreement` and the `tally` rather than trust
the threshold alone, and treat anything short of unanimity as a hold.

Two more things follow from a small pool:

- **`min_voters` defaults to 3.** Run two agents against that default and quorum
  is never met, so *every* verdict is a hold — including two unanimous bullish
  agents scoring `+1.00`. Nothing warns you, the agents still run, and the LLM
  calls are still spent. Set `ATP_VOTING__MIN_VOTERS=2`.
- **A failed agent shrinks the pool.** With `min_voters=2` and two selected
  agents, one failure means a hold. That is the correct outcome, but it makes
  agent reliability a trading concern rather than a cosmetic one.

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
block the workflow.

Callers building an independent review should leave `expected_bot` unset. The
bot is not missing — it is deliberately not voting, and `missing` would describe
a fault where there is none.

## 4. Three ways to integrate

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

### `POST /consensus` with an inline vote — one round trip (scope: `read`)

For a bot that wants the combined verdict back so it can tell its trader.

```bash
curl -X POST http://127.0.0.1:8000/consensus \
  -H "content-type: application/json" \
  -d '{"symbol":"XAUUSD","intervals":["1d","4h"],
       "expected_bot":"sniper_bot","stop_loss":4150,
       "signal":{"symbol":"XAUUSD","direction":"BUY","confidence":0.82,
                 "source":"sniper_bot","rationale":"EMA cross"}}'
```

### `POST /consensus` with no vote — an independent second opinion

Omit `signal` and the agents decide alone, which is what makes their verdict
usable as a check on the bot's own.

> **Omitting `signal` is not sufficient on its own.** When no vote is supplied
> inline, `ReachConsensus._resolve_signal` falls back to the most recent
> **stored** signal for that symbol and counts it if it is inside the TTL. A bot
> that also calls `POST /signals` therefore has its vote pulled back into the
> tally through the back door, and the "independent" verdict quietly becomes
> partly its own. A bot building a review gate must not publish signals at all —
> keep its audit trail on its own side.

```bash
curl -X POST http://127.0.0.1:8000/consensus \
  -H "content-type: application/json" \
  -d '{"symbol":"XAUUSD","intervals":["15m"],
       "agents":["technical_analysis","chart_pattern"],
       "stop_loss":3878.0,
       "bars":{"15m":[{"timestamp":"2026-08-15T14:15:00+00:00",
                       "open":3900.0,"high":3905.0,"low":3895.0,
                       "close":3902.0,"volume":120}]},
       "context":{"strategy":"precision_sniper","entry_timeframe":"M15",
                  "trend_gate":{"regime_direction":"bullish",
                                "bias_direction":"bullish"},
                  "readings":{"ema_50":3891.4,"atr_14":7.31}}}'
```

The `bars` array is truncated to one candle for readability. Sent as-is it would
be *dropped* for having fewer than 60 bars, and the agents would fall back to
fetching their own — see §5.

**Direction vocabulary is flexible.** `BUY`/`SELL`/`NEUTRAL`,
`bullish`/`bearish`/`neutral`, `long`/`short`, `up`/`down`, `hold`/`flat` are all
accepted and normalized — so `python_signal_bot` can forward
`TradeDecision.direction` unchanged.

> **Latency.** A consensus run is one LLM call per selected agent — 15–60s each
> on free OpenRouter models, run concurrently. Call it only when your bot has
> something actionable (`threshold_met`), not every cycle. Selecting fewer
> agents is the most effective lever you have on both latency and quota.

## 5. Supplying your own candles

`bars` carries the candles to analyse, per timeframe, instead of this platform
fetching its own. A bot that has already pulled them from its broker should send
them, for three reasons that compound:

- **They are the series the decision was made on.** A re-fetch happens minutes
  later and may include a different forming bar. When the question is "do you
  agree with what I just decided", both sides have to be looking at the same
  thing.
- **They are the right instrument.** Yahoo serves no spot metal, so `XAUUSD`
  otherwise resolves to `GC=F`, front-month COMEX gold — a real instrument with
  a real basis against spot. Supplied bars bypass vendor symbol translation
  entirely.
- **They are the right feed.** Your broker's prices, spreads and session
  boundaries, not a vendor's.

Rules the boundary enforces, so a bad candle fails as a `422` naming the bar
rather than surfacing later as a `500`:

| Requirement | Why |
|---|---|
| `timestamp` carries a timezone offset | A naive timestamp is ambiguous, and a bot on broker server time is exactly the caller most likely to send one |
| OHLC internally consistent | `high ≥ max(open, close)`, `low ≤ min(open, close)` |
| Prices positive, volume non-negative | Same domain rule as any stored bar |
| Strictly ascending, unique timestamps | Duplicates and gaps break the series |

**60 usable bars per timeframe minimum** (`MIN_HISTORY_BARS` and
`MIN_PATTERN_BARS`). A timeframe carrying fewer is dropped with a log line, not
an error — so under-sending silently narrows the analysis instead of failing it.
Send several hundred.

Supplied bars take priority over both the database and the vendor. That ordering
is deliberate and load-bearing: `LoadPriceHistory` is storage-first, so were it
reversed, a symbol already stored with enough bars would quietly outrank yours
and the agents would analyse different data with nothing in the output saying
so. Look for `history.using_supplied_bars` in the logs to confirm.

Timeframes with no supplied bars fall back to the normal path, so a partial
`bars` map is fine.

## 6. Supplying setup context

`context` is free-form JSON describing what your system sees — indicator values,
a higher-timeframe gate, levels, execution conditions. It reaches
`technical_analysis` and `chart_pattern` as `external_context`, under prompt
rules that mark it supplementary rather than authoritative, and anything an
agent draws from it is cited as `bot:<field>`.

**Leave your intended direction out of it.** An agent told the answer cannot
independently check it, and a matching verdict then means nothing. Send the
readings, withhold the verdict. Some direction inevitably leaks — a gate reading
"D1 bullish, H4 bullish" narrows the field — but the agents still have to commit
to a side of their own accord, which is the property the gate depends on.

The other agents ignore `context` entirely; only the two bar-driven ones read it.

## 7. Building an approval gate

The pattern `python_signal_bot` uses. Each point below exists because the
obvious implementation gets it wrong.

- **Ask only when it matters.** One review per executable signal per bar, not
  per scan cycle. A signal recomputed every 30 seconds on a forming candle would
  otherwise spend an LLM call each time.
- **Do not block.** A verdict takes minutes; a scan loop runs in seconds and
  also manages trailing stops. Submit the request on a worker thread and poll it
  across later cycles.
- **A verdict approves a direction, not a price.** By the time it lands, price
  has moved. Recompute entry, stop, target and size against the current market
  and re-run the filters before placing, rather than firing a stale plan wearing
  an approval.
- **Re-check your own signal.** If the bot no longer wants that direction, an
  approval for it is worthless.
- **Decide what silence means, and say it out loud.** Fail-closed is the safe
  default, but it turns an outage — or an exhausted quota — into a bot that
  places nothing while looking exactly like a quiet market. Count consecutive
  failures and alert on them.
- **Check `agreement`, not just `action`.** See §2.

## 8. Reference implementation

`python_signal_bot` ships this, and the modules are more current than any
snippet pasted here:

| File | Role |
|---|---|
| `src/atp_review.py` | Builds the context payload and bars, posts `/consensus` off the trading thread, normalizes the verdict |
| `src/review_gate.py` | Pending-review queue, deadlines, approve/reject/timeout policy, Telegram reporting |
| `src/headless_bot.py` | `--review` and the three-phase cycle that scans, resolves verdicts, then opens new reviews |
| `scripts/atp_review_stub_server.py` | Runs this API for real with only the LLM stubbed |
| `scripts/e2e_atp_review.py` | Drives the gate against it over HTTP and asserts the agents read the supplied bars |

That last pair is the fastest way to confirm an integration end to end without
an LLM provider configured.

## 9. Credentials

Give the bot a `signal`-scoped key only if it actually publishes signals — a
review-gate bot does not, and needs `read` alone. It is the credential most
likely to sit on a remote box, and it has no business executing:

```ini
ATP_API__KEYS=[
  {"name":"dashboard","key":"...","scopes":["read"]},
  {"name":"sniper_bot","key":"...","scopes":["read"]},
  {"name":"trader","key":"...","scopes":["read","trade"]}
]
```

## 10. From the dashboard

The **Consensus** page does all of the above interactively: symbol, timeframes,
an **Expect a bot** toggle with its name, and a stop loss. It shows the bot
status banner first — because it changes how the verdict below should be read —
then the verdict, the per-voter table with signed weights, and the risk gate's
answer. Recent bot signals are listed underneath so "is it actually connected?"
is answerable at a glance.

## 11. Endpoints

| Endpoint | Scope | Purpose |
|---|---|---|
| `POST /signals` | `signal` | Publish a bot's opinion |
| `GET /signals` | `read` | Audit trail of what bots have claimed |
| `POST /consensus` | `read` | Run the agents, count the votes, return the verdict |
| `POST /trade` | `trade` | Execute — separate step, separate scope |
