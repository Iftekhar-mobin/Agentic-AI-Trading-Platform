"""backtesting.py adapter for the BacktestEngine port.

The declarative StrategyDefinition is interpreted into a dynamically built
``backtesting.Strategy`` subclass. Indicator math reuses our own tested engine
functions, so backtests and live analysis compute identical numbers.

Look-ahead bias guards, in layers:
- PriceHistory guarantees strictly ascending bars (domain invariant).
- Indicators are rolling with NaN warmup; conditions on NaN are always False,
  so nothing can trade before every indicator is defined.
- ``self.I`` exposes indicator arrays truncated to the current bar only.
- Orders fill at the *next* bar's open (library default), never the bar that
  produced the signal.

The library's single-asset event loop is why this stays behind a port: a
portfolio-level engine can replace it without touching any caller.
"""

from __future__ import annotations

import math
import os

import pandas as pd
import structlog

# backtesting.py renders a tqdm progress bar per run, which floods structured
# logs during optimization (hundreds of runs). Must be set before import.
os.environ.setdefault("TQDM_DISABLE", "1")

from backtesting import Backtest, Strategy

from atp.domain.errors import InsufficientHistoryError
from atp.domain.models.backtest import BacktestResult, TradeRecord
from atp.domain.models.market import PriceHistory
from atp.domain.models.strategy import (
    Comparison,
    Condition,
    Operand,
    OperandKind,
    StrategyDefinition,
)
from atp.infrastructure.backtesting.metrics import PERIODS_PER_YEAR, compute_metrics
from atp.infrastructure.indicators import engine

log = structlog.get_logger()


class BacktestingPyEngine:
    def run(
        self,
        strategy: StrategyDefinition,
        history: PriceHistory,
        *,
        initial_cash: float = 10_000.0,
        commission: float = 0.001,
    ) -> BacktestResult:
        if len(history) < strategy.min_history_bars:
            msg = (
                f"{history.symbol}: {len(history)} bars available, strategy "
                f"'{strategy.name}' needs at least {strategy.min_history_bars}"
            )
            raise InsufficientHistoryError(msg)

        frame = _to_backtesting_frame(history)
        backtest = Backtest(
            frame,
            _build_strategy_class(strategy),
            cash=initial_cash,
            commission=commission,
            finalize_trades=True,  # close open positions so they count in stats
        )
        stats = backtest.run()

        equity = stats["_equity_curve"]["Equity"]
        trades_frame = stats["_trades"]
        trade_returns_pct = [float(value) * 100.0 for value in trades_frame["ReturnPct"]]
        trade_log = tuple(
            TradeRecord(
                entry_time=row.EntryTime.to_pydatetime(),
                exit_time=row.ExitTime.to_pydatetime(),
                return_pct=float(row.ReturnPct) * 100.0,
            )
            for row in trades_frame.itertuples()
        )

        result = BacktestResult(
            strategy=strategy,
            symbol=history.symbol,
            interval=history.interval,
            start=history.bars[0].timestamp,
            end=history.bars[-1].timestamp,
            bars=len(history),
            initial_cash=initial_cash,
            final_equity=float(equity.iloc[-1]),
            metrics=compute_metrics(equity, trade_returns_pct, PERIODS_PER_YEAR[history.interval]),
            trade_log=trade_log,
        )
        log.info(
            "backtest.completed",
            strategy=strategy.name,
            symbol=history.symbol,
            bars=result.bars,
            trades=result.metrics.trades,
            total_return_pct=round(result.metrics.total_return_pct, 2),
        )
        return result


def _to_backtesting_frame(history: PriceHistory) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [bar.open for bar in history.bars],
            "High": [bar.high for bar in history.bars],
            "Low": [bar.low for bar in history.bars],
            "Close": [bar.close for bar in history.bars],
            "Volume": [bar.volume for bar in history.bars],
        },
        index=pd.DatetimeIndex([bar.timestamp for bar in history.bars]),
    )


def _operand_series(operand: Operand, frame: pd.DataFrame) -> pd.Series:
    close = frame["Close"]
    period = operand.period
    match operand.kind:
        case OperandKind.OPEN | OperandKind.HIGH | OperandKind.LOW | OperandKind.CLOSE:
            return frame[operand.kind.value.capitalize()]
        case OperandKind.VOLUME:
            return frame["Volume"]
        case OperandKind.SMA:
            assert period is not None
            return engine.sma(close, period)
        case OperandKind.EMA:
            assert period is not None
            return engine.ema(close, period)
        case OperandKind.RSI:
            assert period is not None
            return engine.rsi(close, period)
        case OperandKind.ATR:
            assert period is not None
            return engine.atr(frame["High"], frame["Low"], close, period)
        case OperandKind.MACD_LINE:
            return engine.macd(close)["macd"]
        case OperandKind.MACD_SIGNAL:
            return engine.macd(close)["signal"]
        case (
            OperandKind.BOLLINGER_UPPER
            | OperandKind.BOLLINGER_MIDDLE
            | (OperandKind.BOLLINGER_LOWER)
        ):
            assert period is not None
            band = operand.kind.value.removeprefix("bollinger_")
            return engine.bollinger(close, period)[band]
        case _:
            msg = f"operand kind '{operand.kind}' has no series form"
            raise ValueError(msg)


def _conditions(definition: StrategyDefinition) -> tuple[Condition, ...]:
    return definition.entry + definition.exit


def _attr(label: str) -> str:
    return f"ind_{label}"


_BRACKET_ATTR = "ind_atr_bracket"


def _build_strategy_class(definition: StrategyDefinition) -> type[Strategy]:
    class RuleStrategy(Strategy):  # type: ignore[misc]  # Strategy is untyped
        def init(self) -> None:
            frame: pd.DataFrame = self.data.df
            # Indicators MUST be stored as instance attributes: backtesting.py
            # locates them by scanning the strategy's attributes and re-binds
            # them to progressively truncated views on every bar. An indicator
            # hidden in a container would stay full-length — i.e. look-ahead.
            for condition in _conditions(definition):
                for operand in (condition.left, condition.right):
                    if operand.kind is OperandKind.CONSTANT:
                        continue
                    if not hasattr(self, _attr(operand.label)):
                        series = _operand_series(operand, frame)
                        setattr(
                            self,
                            _attr(operand.label),
                            self.I(lambda s=series: s, name=operand.label, plot=False),
                        )
            self._uses_bracket = (
                definition.stop_loss_atr is not None or definition.take_profit_atr is not None
            )
            if self._uses_bracket:
                atr_series = engine.atr(
                    frame["High"], frame["Low"], frame["Close"], definition.atr_period
                )
                setattr(
                    self,
                    _BRACKET_ATTR,
                    self.I(lambda s=atr_series: s, name="atr_bracket", plot=False),
                )

        def next(self) -> None:
            if self.position:
                if any(self._holds(condition) for condition in definition.exit):
                    self.position.close()
                return
            if all(self._holds(condition) for condition in definition.entry):
                self._enter()

        def _enter(self) -> None:
            stop_loss = take_profit = None
            if self._uses_bracket:
                atr_now = float(getattr(self, _BRACKET_ATTR)[-1])
                if math.isnan(atr_now):
                    return  # brackets undefined during warmup: don't trade blind
                # Brackets are anchored to the signal bar's close; the entry
                # itself fills at the next bar's open (library default).
                price = float(self.data.Close[-1])
                if definition.stop_loss_atr is not None:
                    stop_loss = price - definition.stop_loss_atr * atr_now
                if definition.take_profit_atr is not None:
                    take_profit = price + definition.take_profit_atr * atr_now
            self.buy(sl=stop_loss, tp=take_profit)

        def _value(self, operand: Operand, offset: int = -1) -> float:
            if operand.kind is OperandKind.CONSTANT:
                assert operand.value is not None
                return operand.value
            series = getattr(self, _attr(operand.label))
            if len(series) < -offset:
                return math.nan
            return float(series[offset])

        def _holds(self, condition: Condition) -> bool:
            left_now = self._value(condition.left)
            right_now = self._value(condition.right)
            if math.isnan(left_now) or math.isnan(right_now):
                return False
            match condition.comparison:
                case Comparison.GT:
                    return left_now > right_now
                case Comparison.LT:
                    return left_now < right_now
                case Comparison.CROSSES_ABOVE | Comparison.CROSSES_BELOW:
                    left_prev = self._value(condition.left, offset=-2)
                    right_prev = self._value(condition.right, offset=-2)
                    if math.isnan(left_prev) or math.isnan(right_prev):
                        return False
                    if condition.comparison is Comparison.CROSSES_ABOVE:
                        return left_prev <= right_prev and left_now > right_now
                    return left_prev >= right_prev and left_now < right_now

    RuleStrategy.__name__ = f"RuleStrategy_{definition.name}"
    return RuleStrategy
