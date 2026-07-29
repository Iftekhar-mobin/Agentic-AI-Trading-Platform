# Agentic AI Trading Platform — Architecture

> Living document. Version 0.1 — initial system design (2026-07-23).

## 1. Vision

An autonomous AI trading organization: a Supervisor agent orchestrates specialist
agents that research markets, generate and validate strategies, manage risk, execute
trades, and learn from outcomes. The system **assists traders with explainable
recommendations** — it never produces black-box signals.

## 2. Architectural Style

**Modular monolith with Clean (Hexagonal) Architecture**, deployed as containers,
designed so any module can later be extracted into a service without rewrites.

Dependency rule: source code dependencies point **inward only**.

```
┌─────────────────────────────────────────────────────────┐
│  interfaces/      FastAPI REST + WebSocket, Streamlit,  │
│                   CLI                                   │
├─────────────────────────────────────────────────────────┤
│  application/     Use cases, agent orchestration        │
│                   (LangGraph), workflow definitions     │
├─────────────────────────────────────────────────────────┤
│  domain/          Entities, value objects, domain       │
│                   services, ports (abstract interfaces) │
├─────────────────────────────────────────────────────────┤
│  infrastructure/  Adapters: market-data providers,      │
│                   brokers, LLM clients, TimescaleDB,    │
│                   Redis, Qdrant, indicator engine       │
└─────────────────────────────────────────────────────────┘
```

- `domain` has **zero** third-party runtime dependencies beyond Pydantic.
- `application` depends on `domain` only; it talks to the outside world through
  **ports** (Python `Protocol`s / ABCs defined in `domain/ports`).
- `infrastructure` implements those ports; swapping Yahoo → Polygon or Alpaca → IBKR
  is a new adapter, not a rewrite.
- Wiring happens in a single **composition root** via dependency injection.

## 3. Agent Model

Orchestration: **LangGraph** (chosen over CrewAI — see §7). The Supervisor is a
LangGraph `StateGraph` whose nodes are specialist agents.

Every agent obeys one contract:

- **Typed I/O**: Pydantic request/response models — no free-form dict passing.
- **Explainability envelope**: every output carries `reasoning`, `confidence`,
  `evidence[]`, and `invalidation_conditions[]`.
- **Statelessness**: agents read/write shared `TradingState`; persistence lives
  behind repositories, never inside agents.
- **Tool separation**: deterministic computation (indicators, backtests, risk math)
  is plain tested Python called *by* agents. LLMs interpret and decide; they never
  compute numbers.

```
Supervisor (StateGraph router)
  ├── analysis pool      : technical, fundamental, news, sentiment
  │                        (dispatched concurrently in one superstep)
  ├── strategy pool      : strategy_generation → backtesting → optimization
  ├── decision pool      : risk_management (veto power) → portfolio
  ├── execution          : trade_execution (paper first, gated live)
  └── feedback           : continuous_learning (writes to episodic memory)
```

Risk Management is a **hard gate**: no order reaches the execution agent without a
risk approval token in state. This is enforced in graph topology, not prompts.

## 4. Data Architecture

| Store        | Role                                                        |
|--------------|-------------------------------------------------------------|
| TimescaleDB  | OHLCV hypertables, indicator caches, backtest results       |
| PostgreSQL   | Trades, orders, portfolios, strategy definitions, audit log |
| Redis        | Quote cache, rate-limit buckets, task queues, pub/sub       |
| Qdrant       | Episodic memory: embedded trade rationales, news, regimes   |

Market data flows through a **provider-agnostic port** (`MarketDataProvider`), with
yfinance as the dev adapter and Polygon/Alpaca as production adapters. All fetched
data is normalized to canonical domain models before anything else touches it.

## 5. Repository Layout

```
agentic-trading-platform/
├── pyproject.toml
├── docker-compose.yml
├── .github/workflows/
├── docs/
├── src/atp/                        # "Agentic Trading Platform"
│   ├── domain/
│   │   ├── models/                 # Instrument, Bar, Signal, Order, Position,
│   │   │                           # Portfolio, Strategy, RiskLimits, ...
│   │   ├── ports/                  # MarketDataProvider, Broker, LLMClient,
│   │   │                           # NewsProvider, repositories (Protocols)
│   │   └── services/               # pure domain logic (position sizing math, ...)
│   ├── application/
│   │   ├── agents/                 # one package per agent
│   │   ├── orchestration/          # LangGraph graphs, TradingState, routing
│   │   └── use_cases/              # AnalyzeTicker, RunBacktest, PlaceTrade, ...
│   ├── infrastructure/
│   │   ├── market_data/            # yfinance_adapter, polygon_adapter, ...
│   │   ├── brokers/                # alpaca_adapter (paper), ibkr_adapter, ...
│   │   ├── llm/                    # anthropic_client, openai_client
│   │   ├── indicators/             # deterministic TA engine (pandas/polars)
│   │   ├── backtesting/            # backtesting.py integration, walk-forward
│   │   ├── persistence/            # SQLAlchemy repos, Timescale, Redis, Qdrant
│   │   └── config/                 # pydantic-settings, secrets loading
│   ├── interfaces/
│   │   ├── api/                    # FastAPI routers, schemas, deps
│   │   └── cli/
│   └── composition.py              # DI container / composition root
├── ui/streamlit_app/               # internal dashboard (React later)
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

## 6. Milestone Roadmap

| # | Milestone | Delivers |
|---|-----------|----------|
| 0 | Foundation | Repo scaffold, tooling (ruff, mypy, pytest, pre-commit), config, logging, Docker Compose (Postgres+Timescale, Redis), CI |
| 1 | Domain + market data | Core domain models, `MarketDataProvider` port, yfinance adapter, Timescale persistence, first tests |
| 2 | Indicator engine + TA agent | Deterministic indicator library, signal classification, first LLM-backed agent with explainability envelope |
| 3 | Orchestration skeleton | LangGraph supervisor, `TradingState`, end-to-end "analyze ticker" flow via CLI + FastAPI endpoint |
| 4 | Strategy + backtesting | Strategy definition schema (declarative rules), backtesting.py integration, metrics suite, look-ahead-bias guards |
| 5 | Optimization | Optuna integration, walk-forward validation, Monte Carlo, overfitting controls |
| 6 | Risk + portfolio | Risk limit engine (hard gate), position sizing, portfolio tracking + rebalancing |
| 7 | Execution | Alpaca paper trading, order state machine, bracket/trailing orders, execution audit log |
| 8 | Intelligence expansion | Fundamentals, news, sentiment agents; FinBERT + LLM pipeline |
| 9 | Learning + memory | Qdrant episodic memory, trade journaling, regime tagging, retrieval into future decisions |
| 10 | Product surface | Streamlit dashboard → React, authn/z, observability (OpenTelemetry), hardening |

Each milestone ends with working, tested, demoable software.

## 7. Key Decisions & Trade-offs

- **LangGraph over CrewAI** — trading workflows need deterministic routing, cycles
  (generate → backtest → optimize → regenerate), checkpointing, and human-in-the-loop
  gates. LangGraph gives explicit graph control; CrewAI's role-play autonomy is a
  liability where a risk gate must be structurally unavoidable.
- **Modular monolith first, not microservices** — one deployable with strict internal
  boundaries. Microservices now would multiply ops cost with zero benefit at this
  stage; clean ports make later extraction cheap.
- **LLMs never compute** — all numbers (indicators, risk, backtest stats) come from
  deterministic, unit-tested code. LLMs interpret, contextualize, and explain.
  This makes outputs reproducible and testable.
- **backtesting.py to start** — matches the brief and is simple; its event loop is
  single-asset-oriented, so the integration hides behind a `BacktestEngine` port and
  vectorbt/custom engine can replace it for portfolio-level simulation later.
- **yfinance for dev only** — unreliable/unofficial; fine behind the provider port
  for development, replaced by Polygon/Alpaca keys in production settings.
- **FinBERT optional, lexicon default** — the sentiment classifier sits behind a
  `SentimentModel` port. FinBERT (`transformers` + `torch`) is the production
  path but an optional extra: it adds gigabytes to an image most deployments do
  not need, and the suite must run offline. The default adapter is a real
  Loughran-McDonald-style lexicon classifier — blunt, but deterministic and
  explainable — so the sentiment pipeline is meaningfully testable without it.
- **Analysis agents fan out** — technical, fundamental, news and sentiment are
  independent (none reads another's output), so the supervisor dispatches them
  in one superstep. That is four LLM round-trips of latency collapsed into one.
  A shared, locked TTL cache in `LoadNews` stops the news and sentiment agents
  double-fetching the same articles from a rate-limited vendor.
- **Paper trading is the default** — live execution requires an explicit config flag,
  a separate credential set, and passes through the same risk gate.

## 8. Cross-Cutting Requirements

- **Explainability**: the envelope (why / why now / evidence / risk / confidence /
  invalidation) is part of the response schema — code-enforced, not prompt-hoped.
- **Auditability**: every agent decision, LLM call, and order transition is logged
  with correlation IDs (structlog, JSON output).
- **Testing**: unit tests per module (deterministic parts get golden-value tests
  against known data); integration tests with testcontainers; LLM calls faked at the
  port boundary so the suite runs offline.
- **Security**: secrets via env/secret manager only, broker credentials never
  logged, API authn from milestone 10, risk limits stored server-side.
- **Compliance posture**: the system produces *recommendations and user-approved
  executions*, with full audit trail — required if this becomes commercial.
