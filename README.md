# Agentic AI Trading Platform

A production-grade multi-agent AI trading assistant: a Supervisor agent orchestrates
specialist agents that research markets, generate and backtest strategies, manage
risk, execute trades, and learn from outcomes — with every recommendation fully
explainable.

**Architecture and roadmap:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (manages Python 3.12 and dependencies automatically)
- Docker Desktop (for TimescaleDB + Redis; required from Milestone 1)

## Quickstart

```bash
uv sync                    # create venv, install Python 3.12 + dependencies
cp .env.example .env       # local configuration
docker compose up -d       # start TimescaleDB + Redis + Qdrant
uv run atp status          # smoke command: prints configuration wiring
uv run atp sync AAPL       # pull daily bars into TimescaleDB
uv run atp bars AAPL -n 5  # show the 5 most recent stored bars
uv run atp analyze AAPL    # run all analysis agents in parallel (needs an Anthropic key)
uv run atp analyze AAPL -a technical_analysis,news_analysis   # run a subset
uv run atp backtest AAPL -s ema_cross   # backtest a strategy preset
uv run atp serve           # start the HTTP API on http://127.0.0.1:8000
uv run streamlit run ui/streamlit_app/app.py   # dashboard (needs the API running)
uv run pytest              # run the test suite
```

API example:

```bash
curl -X POST http://127.0.0.1:8000/analysis \
  -H "content-type: application/json" \
  -d '{"symbol": "AAPL", "interval": "1d"}'
```

Integration tests skip automatically when the database is not running.

## Development

| Command | Purpose |
|---------|---------|
| `uv run pytest --cov=atp` | tests with coverage |
| `uv run ruff check .` | lint |
| `uv run ruff format .` | format |
| `uv run mypy` | strict type checking |
| `uv run pre-commit install` | enable git hooks (once) |
| `docker compose up -d` | start TimescaleDB + Redis + Qdrant |

CI (GitHub Actions) runs format check, lint, mypy, and tests on every push/PR.

## Analysis agents

`atp analyze` (and `POST /analysis`) runs the supervisor graph over five specialists:

| Agent | Deterministic layer | LLM layer |
|-------|---------------------|-----------|
| `technical_analysis` | indicator engine + rule classification | interprets the readings |
| `fundamental_analysis` | threshold rules over a fundamentals snapshot | interprets, sector-adjusts |
| `news_analysis` | — (article retrieval only) | themes, catalysts, materiality |
| `sentiment_analysis` | FinBERT/lexicon scores + recency-weighted aggregate | interprets the aggregate |
| `continuous_learning` | regime tagging + semantic recall from the journal | compares now to precedent |

The first four run concurrently; `continuous_learning` runs after them, because it
reflects on what they concluded. An agent that fails is reported in `failures`; the
others still return their work.

Sentiment defaults to a transparent offline lexicon classifier. For real FinBERT:

```bash
uv sync --extra finbert
ATP_SENTIMENT__MODEL=finbert uv run atp analyze AAPL -a sentiment_analysis
```

## Memory

Every analysis and every trade — fills and risk rejections alike — is journalled as
a regime-tagged episode, then recalled semantically on later runs:

```bash
uv run atp memory AAPL                      # recall episodes for a symbol
uv run atp memory AAPL -q "breakout entry"  # recall by free-text query
```

Storage defaults to a local file so the loop works with nothing running. Point it at
Qdrant (`docker compose up -d`) for production:

```bash
ATP_MEMORY__BACKEND=qdrant uv run atp analyze AAPL
```

## Security and operations

- **Authentication** — bearer API keys with `read` / `trade` scopes, configured
  as JSON in `ATP_API__KEYS`. With no keys the API runs open, which is handy on
  a laptop and refused outright in production: `ATP_ENVIRONMENT=production`
  will not start without at least one key.
- **Rate limiting** — per credential, `ATP_API__RATE_LIMIT_PER_MINUTE`
  (`/health` is never throttled). In-process for now; Redis when there is more
  than one replica.
- **Correlation** — every response carries `X-Request-ID`, and every log line
  produced during that request carries the same id. Send your own to keep a
  trace whole across services.
- **Tracing** — OpenTelemetry over OTLP/HTTP, off unless
  `ATP_OBSERVABILITY__OTLP_ENDPOINT` is set. When on, logs also carry
  `trace_id` and `span_id`.
- **Errors** — domain failures map to meaningful statuses (422 bad input, 409
  execution conflict, 502 model failure, 503 dependency down) with the request
  id in the body.

## Safety defaults

- `trading_mode` defaults to **paper**; `live` is rejected unless
  `ATP_ENVIRONMENT=production`.
- Secrets load from environment / `.env` only (never committed) and are masked
  in logs and reprs via `SecretStr`.
- Every order — filled or refused by the risk gate — lands in the append-only
  audit trail and the episodic journal, whichever interface placed it.
