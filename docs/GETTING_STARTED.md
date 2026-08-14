# Getting Started

A run-and-operate guide: how to stand the platform up, how to ask it a question,
and what actually happens between the two. For the design rationale see
[ARCHITECTURE.md](ARCHITECTURE.md); for the command and endpoint reference see
the [README](../README.md).

---

## 1. What you need

| Requirement | Why | Optional? |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | Installs Python 3.12 and every dependency | No |
| An LLM backend | The agents' reasoning layer | No — pick one in §2 |
| Docker Desktop | TimescaleDB, Redis, Qdrant | **Yes** — see below |
| Network access | Yahoo Finance prices, news, model API | No |

**Docker is optional for a first run.** Price history is storage-first with a
live-provider fallback: when the database is unreachable the platform logs
`history.repository_unavailable` and fetches from Yahoo directly. Episodic memory
defaults to a local JSON file. So the whole workflow runs with nothing but `uv`
and a key — you lose bar caching and vector recall quality, not function.

---

## 2. Setup

```bash
uv sync                 # Python 3.12 + dependencies
cp .env.example .env    # your local configuration
```

Now choose a reasoning backend and edit `.env`.

### Option A — OpenRouter (free, recommended for evaluation)

OpenRouter aggregates hundreds of models behind one API and offers a genuinely
free tier: any model whose id ends in `:free`. This is the fastest way to see the
whole system work without spending anything.

1. Create an account at [openrouter.ai](https://openrouter.ai) and a key at
   [openrouter.ai/keys](https://openrouter.ai/keys).
2. In `.env`:

```ini
ATP_LLM__PROVIDER=openrouter
ATP_LLM__OPENROUTER_API_KEY=sk-or-v1-...
ATP_LLM__OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
```

> **The free roster changes without notice.** Models are added and withdrawn
> weekly, so the id above may be gone by the time you read this. The Models page
> in the dashboard lists what is live right now, or:
> `curl -s https://openrouter.ai/api/v1/models | grep -o '"id":"[^"]*:free"'`
>
> **Prefer models above ~100B parameters.** Every agent output must carry
> reasoning, calibrated confidence, cited evidence *and* invalidation
> conditions. Small models drop required fields and surface as failed agents.
> See §7.

### Option B — Ollama (free, local, private)

No key, no per-token cost, and no prompt ever leaves the machine. Install from
[ollama.com](https://ollama.com), then:

```bash
ollama serve
ollama pull llama3.1:8b
```

```ini
ATP_LLM__PROVIDER=ollama
ATP_LLM__OLLAMA_MODEL=llama3.1:8b
```

Ollama constrains decoding with the JSON schema itself, so schema adherence is
much better than free hosted models of comparable size. It is slow on CPU — a
minute per agent is normal, not a hang.

### Option C — Anthropic (paid, best quality)

```ini
ATP_LLM__PROVIDER=anthropic
ATP_LLM__ANTHROPIC_API_KEY=sk-ant-...
ATP_LLM__MODEL=claude-opus-4-8
```

### Verify

```bash
uv run atp status
```

Prints the resolved configuration — provider, model, trading mode, risk limits.
If this is right, everything downstream is wired correctly.

---

## 3. Running it

Three interfaces over the same use cases. Nothing is available in one that is
absent from another, because they all call the same application layer.

### CLI — fastest for a single question

```bash
uv run atp analyze XAUUSD                      # every agent, daily
uv run atp analyze AAPL -t 1d,4h,1h            # multi-timeframe
uv run atp analyze AAPL -a technical_analysis  # one agent only
uv run atp bars GC=F -n 5                      # inspect stored bars
uv run atp backtest AAPL -s ema_cross          # backtest a preset
uv run atp memory AAPL                         # recall past episodes
```

### HTTP API

```bash
uv run atp serve      # http://127.0.0.1:8000, interactive docs at /docs
```

```bash
curl -X POST http://127.0.0.1:8000/analysis \
  -H "content-type: application/json" \
  -d '{"symbol":"XAUUSD","intervals":["1d","4h"],"agents":["technical_analysis"]}'
```

### Dashboard — the intended way to *read* a result

```bash
uv run streamlit run ui/streamlit_app/app.py    # -> :8501
```

One terminal is enough: if the API is not running, the sidebar offers a **Start
the API** button that launches it for you (and a **Stop** button when it did).
`uv run atp serve` in a second terminal still works and is what you want when
you need the server's log in front of you.

Pages, along the top: **Opportunities** (screen a basket of instruments and rank
the best few), **Analysis** (run agents on one symbol, read the evidence),
**Consensus** (agents plus your bot vote), **Processing** (see §4.1),
**Portfolio** (positions marked to live prices, audit trail), **Trade** (risk
gate), **Memory** (recall), **Models** (list/switch/download models without a
restart).

The dashboard is a pure API client — it holds no business logic and talks to the
server over HTTP, so it works against a remote deployment by changing the base
URL in the sidebar.

> Launch it with `uv run` where you can. It also runs against a system-wide
> Streamlit (1.48+ is handled), but only `uv run` guarantees the pinned version.

---

## 4. How a request actually works

Take `POST /analysis {"symbol": "XAUUSD", "intervals": ["1d","4h"]}`.

```
                    ┌─> technical_analysis   ─┐
                    ├─> chart_pattern        ─┤
  START ─> supervisor ─> market_research      ─┼─> supervisor ─> continuous_learning ─> END
                    ├─> fundamental_analysis ─┤
                    ├─> news_analysis        ─┤
                    └─> sentiment_analysis   ─┘
```

1. **Routing is code, not a prompt.** The supervisor is a deterministic function
   over the state: it dispatches every requested agent whose artifact is still
   missing and ends when none are left. Guarantees like *no execution without
   risk approval* are graph edges — an LLM is never asked to honour them.

2. **The six analysis agents run concurrently.** None reads another's output, so
   fanning them out costs one round-trip of latency instead of six.

3. **Each agent computes first, then interprets.** This is the platform's central
   rule: **LLMs never compute numbers.** Deterministic, unit-tested Python
   produces every figure — indicators, pattern geometry, relative strength,
   sentiment aggregates, regime tags. The model receives those figures and
   explains what they mean. So a wrong number is a bug with a failing test, not
   a hallucination.

4. **Symbols are translated at the vendor boundary** (see §5).

5. **`continuous_learning` runs second**, because it reflects on what phase one
   concluded. It tags the market regime, recalls similar past episodes from the
   journal, and compares now to precedent. Skipped entirely if no analysis
   produced a report.

6. **Failures are recorded, not raised.** An agent that fails appends to
   `failures` and the rest still return their work. A missing news feed costs you
   the news section, not the analysis.

7. **Every conclusion arrives in an explanation envelope** — the reason the
   platform exists:

   | Field | Meaning |
   |---|---|
   | `direction` | bullish / bearish / neutral |
   | `confidence` | 0–1, calibrated |
   | `reasoning` | the argument |
   | `evidence` | source-tagged facts, e.g. `1d:macd_12_26_9` |
   | `invalidation_conditions` | what would make this wrong |

   Read `invalidation_conditions` first. A conclusion you cannot interrogate is
   one you should not act on, and a model that cannot say what would falsify its
   own view has not reasoned about it.

### 4.1 Watching it happen — the Processing panel

Everything above is observable rather than asserted. The **Processing** page —
and the `⚙️ Processing` expander under each Analysis result — shows how the last
run was actually carried out:

- **Cost of the run** — total wall-clock time, agents dispatched, how many
  failed, and which provider/model produced the reasoning.
- **Agent timeline** — every agent with its phase (`analysis` or `feedback`),
  status, duration with a proportional bar, and what it concluded
  (`bullish, 65% confidence, 8 pieces of evidence`) or why it failed. This is
  where you see that one slow agent, not the platform, owns your latency.
- **API calls** — each HTTP exchange with the exact request JSON sent, the
  response received, status, duration, payload size, and the `X-Request-ID`
  that ties it to every server log line for that call.

Because the analysis agents run concurrently, their durations overlap and will
not sum to the total — the timeline labels the phases so this reads correctly.

The trace comes from the API itself (`steps`, `active_model` and `duration_ms`
on the analysis response), so it is available to any client, not just this
dashboard.

---

## 5. Symbols

Ordinary equity tickers work as typed (`AAPL`, `MSFT`). Yahoo does not serve spot
metals or FX under the names traders use, so those are translated at the vendor
boundary — the domain keeps the symbol you asked for, and reports still say
`XAUUSD`.

| You type | Fetched as | Note |
|---|---|---|
| `XAUUSD` | `GC=F` | Gold — COMEX front-month future |
| `XAGUSD` | `SI=F` | Silver |
| `XPTUSD` / `XPDUSD` | `PL=F` / `PA=F` | Platinum / palladium |
| `EURUSD`, `GBPJPY` | `EURUSD=X` | Any ISO currency pair |
| `BTCUSD`, `BTCUSDT` | `BTC-USD` | Major crypto |
| `GC=F`, `^GSPC`, `BRK.B` | unchanged | Already vendor-shaped |

> **Metals resolve to futures, not spot.** The front-month contract carries roll
> gaps and a basis of a few dollars against the spot fix. For trend, structure
> and levels — what the agents reason about — that is immaterial. For anything
> settling against spot, it is not.

Anything unrecognized passes through untouched. See
`src/atp/infrastructure/vendor_symbols.py`.

**Commodities and FX have no equity fundamentals.** Running
`fundamental_analysis` on `XAUUSD` correctly fails with *"0 fundamental metrics
available, 3 required"* — gold has no P/E or profit margin. That is honest
degradation, not a defect. Skip the agent with
`-a technical_analysis,chart_pattern,market_research,news_analysis,sentiment_analysis`.

---

## 6. Trading

Paper by default. `live` is refused unless `ATP_ENVIRONMENT=production`.

```bash
uv run atp risk-check AAPL --side buy --stop 180   # verdict only
uv run atp trade AAPL --side buy --stop 180        # through the gate
```

Every proposal passes a deterministic risk engine before reaching a broker. **A
rejection is the system working.** With no `--qty`, position size is derived from
your risk limits and the stop distance. Omitting `--stop` is rejected outright
(`missing_stop_loss`) — the platform will not size a position whose loss it
cannot bound. Limits live under `ATP_RISK__*`; the engine is the only thing that
enforces them and nothing downstream can loosen them at runtime.

Filled or refused, every order lands in the append-only audit trail and the
episodic journal.

---

## 7. Free-model realities

Running on `:free` models is genuinely useful and has three sharp edges.

**Schema adherence.** The output contract is demanding. When a model omits a
required field, the OpenRouter adapter spends exactly one corrective round-trip
quoting the precise validation errors — which fixes the common
single-dropped-field case. A model that misses twice becomes a recorded agent
failure. Look for `llm.schema_repair` in the logs; frequent entries mean the
model is too small.

**Rate limits.** Free models share a small quota. A 7-agent run is 7+ calls, and
HTTP 429 surfaces as *"OpenRouter rate-limited model ..."*. Wait, switch models
on the Models page, or run fewer agents.

**Latency.** 4–45s per agent depending on the model and queue depth, against
~2s for Anthropic. A full run can take several minutes.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `OpenRouter needs an API key` | Key unset | Set `ATP_LLM__OPENROUTER_API_KEY` in `.env` |
| `No endpoints found for <model>` | Model withdrawn from free tier | Pick a current one on the Models page |
| `OpenRouter rate-limited` | Free quota exhausted | Wait, switch model, or run fewer agents |
| `returned JSON that does not match` | Model too weak, twice | Use a larger model (§2) |
| `0 fundamental metrics available` | Commodity/FX symbol | Expected — skip that agent (§5) |
| Empty/short history | Unknown ticker | Check §5; verify on Yahoo Finance |
| `history.repository_unavailable` | No database | Harmless — falls back to live fetch |
| Dashboard: *API unreachable* | Server not running | `uv run atp serve` |
| Dashboard 401/403 | Key missing or wrong scope | Set it in the sidebar |
| A run was slow or partial | — | Open **Processing** (§4.1) — it names the agent |

---

## 9. Before you commit

The quality gate, identical to CI:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

Integration tests skip automatically when the database is not running.

---

## 10. A word on what this is

Decision support, not a signal service. Every recommendation is explainable by
construction — numbers from tested code, interpretation from a model, evidence
and invalidation conditions attached to both. Nothing here is financial advice,
the paper broker is a simulation, and a confident-sounding agent is still an
agent.
