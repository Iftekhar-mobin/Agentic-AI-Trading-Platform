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
docker compose up -d       # start TimescaleDB + Redis
uv run atp status          # smoke command: prints configuration wiring
uv run atp sync AAPL       # pull daily bars into TimescaleDB
uv run atp bars AAPL -n 5  # show the 5 most recent stored bars
uv run atp analyze AAPL    # run the analysis workflow (needs an Anthropic key)
uv run atp backtest AAPL -s ema_cross   # backtest a strategy preset
uv run atp serve           # start the HTTP API on http://127.0.0.1:8000
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
| `docker compose up -d` | start TimescaleDB + Redis |

CI (GitHub Actions) runs format check, lint, mypy, and tests on every push/PR.

## Safety defaults

- `trading_mode` defaults to **paper**; `live` is rejected unless
  `ATP_ENVIRONMENT=production`.
- Secrets load from environment / `.env` only (never committed) and are masked
  in logs and reprs via `SecretStr`.
