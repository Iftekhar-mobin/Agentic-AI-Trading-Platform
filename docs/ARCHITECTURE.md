# Agentic AI Trading Platform — Architecture

> Living document. Version 0.2 — milestones 0-10 delivered (2026-07-29).

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
  ├── analysis pool      : technical, chart_pattern, market_research,
  │                        fundamental, news, sentiment
  │                        (dispatched concurrently in one superstep)
  ├── strategy pool      : strategy_generation → backtesting → optimization
  ├── decision pool      : risk_management (veto power) → portfolio
  ├── execution          : trade_execution (paper first, gated live)
  └── feedback           : continuous_learning (recalls precedent, writes
                           back to episodic memory; runs after the analysis
                           pool because it reflects on that pool's output)
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
│   │   ├── fundamentals/           # company metrics snapshots
│   │   ├── news/                   # article retrieval
│   │   ├── sentiment/              # finbert (optional), lexicon (default)
│   │   ├── embeddings/             # hashing (default), hosted models later
│   │   ├── memory/                 # qdrant_memory, json_memory
│   │   ├── brokers/                # alpaca_adapter (paper), ibkr_adapter, ...
│   │   ├── llm/                    # anthropic_client, openai_client
│   │   ├── indicators/             # deterministic TA engine (pandas/polars)
│   │   ├── backtesting/            # backtesting.py integration, walk-forward
│   │   ├── optimization/           # optuna
│   │   ├── observability/          # structlog config, OpenTelemetry tracing
│   │   ├── persistence/            # SQLAlchemy repos, Timescale, Redis, Qdrant
│   │   └── config/                 # pydantic-settings, secrets loading
│   ├── interfaces/
│   │   ├── api/                    # routers, security, middleware, errors
│   │   └── cli/
│   └── composition.py              # DI container / composition root
├── ui/streamlit_app/               # internal dashboard (React later)
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

## 6. Milestone Roadmap

| # | Milestone | Delivers | Status |
|---|-----------|----------|--------|
| 0 | Foundation | Repo scaffold, tooling (ruff, mypy, pytest, pre-commit), config, logging, Docker Compose (Postgres+Timescale, Redis), CI | ✅ |
| 1 | Domain + market data | Core domain models, `MarketDataProvider` port, yfinance adapter, Timescale persistence, first tests | ✅ |
| 2 | Indicator engine + TA agent | Deterministic indicator library, signal classification, first LLM-backed agent with explainability envelope | ✅ |
| 3 | Orchestration skeleton | LangGraph supervisor, `TradingState`, end-to-end "analyze ticker" flow via CLI + FastAPI endpoint | ✅ |
| 4 | Strategy + backtesting | Strategy definition schema (declarative rules), backtesting.py integration, metrics suite, look-ahead-bias guards | ✅ |
| 5 | Optimization | Optuna integration, walk-forward validation, Monte Carlo, overfitting controls | ✅ |
| 6 | Risk + portfolio | Risk limit engine (hard gate), position sizing, portfolio tracking + rebalancing | ✅ |
| 7 | Execution | Alpaca paper trading, order state machine, bracket/trailing orders, execution audit log | ✅ |
| 8 | Intelligence expansion | Fundamentals, news, sentiment agents; FinBERT + LLM pipeline | ✅ |
| 9 | Learning + memory | Qdrant episodic memory, trade journaling, regime tagging, retrieval into future decisions | ✅ |
| 10 | Product surface | Streamlit dashboard → React, authn/z, observability (OpenTelemetry), hardening | ✅ |

Each milestone ends with working, tested, demoable software. All ten are
delivered; the roadmap now continues into hardening the pieces marked as
starter implementations below (in-process rate limiting, hashing embeddings,
the paper broker).

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
- **Embeddings behind a port, hashing by default** — episodic memory needs
  vectors, and Anthropic has no embeddings API, so the alternative to a port was
  a hard dependency on a second vendor. The default adapter is a deterministic
  hashed bag-of-ngrams: it captures lexical overlap only, but it makes storage,
  filtering and ranking testable end to end with no network. A hosted embedding
  model is a new adapter and one line in the composition root.
- **Memory degrades, never blocks** — an unreachable vector store, a missing
  regime, a failed write: each is logged and the reflection still returns. An
  analysis workflow that fails because a database is down is a worse trade than
  one that returns without precedent.
- **API keys, not user accounts (yet)** — the platform has one operator and one
  paper portfolio, so scoped bearer keys buy the property that actually matters
  now: a dashboard credential that can read everything cannot place an order.
  Real user accounts arrive with real multi-tenancy, not before.
- **Unauthenticated mode cannot be deployed** — running open is what makes
  `atp serve` usable on a laptop, so instead of removing it, `Settings` refuses
  to start in production without keys and every anonymous request is logged.
  A safe default beats a documented warning.
- **Streamlit over the HTTP API, not the container** — the dashboard is just
  another client. If something is awkward to render, the API is missing
  something, and the React front end will hit the same endpoints unchanged.
- **Multi-timeframe is one assessment, not N** — the useful output of MTF
  analysis is the *relationship* between timeframes ("daily is bullish, the
  hourly is overbought — wait for the pullback"). Running the agent once per
  timeframe and merging afterwards throws exactly that away, so the agent
  receives every timeframe at once and returns a single verdict plus an
  explicit `timeframe_alignment`. A deterministic per-timeframe direction is
  computed alongside it, so agreement is checkable in code.
- **4H is synthesized, not sourced** — no mainstream vendor serves it. Rather
  than drop a standard swing-trading timeframe or teach every adapter to fake
  one, bars are aggregated in a domain service and applied by a provider
  decorator, on fixed UTC boundaries so a candle's contents never depend on
  when the fetch started. The cost is that buckets do not align to exchange
  sessions, which is documented rather than hidden.
- **Pattern detection is code, emphatically** — this is where an LLM is most
  prone to confabulating, because given latitude something is always a head and
  shoulders. Formations are detected with fixed proportional rules, carry the
  price levels that define them, and are marked `confirmed` or forming; the
  model is told the list is exhaustive and may not add to it.
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
