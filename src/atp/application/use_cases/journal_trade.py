"""Use case: journal an executed trade into episodic memory.

Deliberately separate from ``ExecuteTrade``. That use case owns one guarantee —
no order reaches the broker without risk approval — and the value of it being
short and obvious is worth more than the convenience of folding journaling in.
The append-only order audit trail is still the system of record; this is the
searchable, regime-tagged layer on top of it.

Rejections are journalled too. "We wanted this trade and the gate refused" is
exactly the kind of precedent worth recalling later.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import structlog

from atp.domain.errors import DomainError
from atp.domain.models.memory import EpisodeKind, MemoryEpisode, RegimeTag
from atp.domain.models.orders import ExecutionResult
from atp.domain.ports.memory import EpisodicMemory

log = structlog.get_logger()


class JournalTrade:
    def __init__(
        self,
        memory: EpisodicMemory,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._memory = memory
        self._clock = clock

    async def execute(
        self,
        result: ExecutionResult,
        *,
        regime: RegimeTag | None = None,
    ) -> MemoryEpisode | None:
        """Write the outcome to memory. Returns the episode, or None if it could not be stored."""
        episode = self._to_episode(result, regime, self._clock())
        try:
            await self._memory.remember(episode)
        except DomainError as exc:
            # Journaling must never undo a completed trade.
            log.warning("journal.remember_failed", episode_id=episode.id, error=str(exc))
            return None
        log.info("journal.trade_recorded", episode_id=episode.id, symbol=episode.symbol)
        return episode

    @staticmethod
    def _to_episode(
        result: ExecutionResult, regime: RegimeTag | None, now: datetime
    ) -> MemoryEpisode:
        decision = result.decision
        proposal = decision.proposal
        order = result.order

        if order is None:
            # A rejected proposal never became an order, so it has no timestamp
            # of its own; the gate ran just now.
            occurred_at = now
            identifier = f"trade:{proposal.symbol}:{occurred_at.isoformat()}"
            violations = ", ".join(violation.rule for violation in decision.violations)
            summary = (
                f"Proposed {proposal.side.value} {proposal.quantity} {proposal.symbol} "
                f"at {proposal.entry_price}; the risk gate rejected it ({violations})."
            )
            metadata: dict[str, str | float | int | bool | None] = {
                "outcome": "rejected_by_risk",
                "side": proposal.side.value,
                "quantity": float(proposal.quantity),
                "entry_price": float(proposal.entry_price),
                "violations": violations,
            }
        else:
            occurred_at = order.filled_at or order.submitted_at
            identifier = f"trade:{proposal.symbol}:{order.id}"
            fill = f" filled at {order.fill_price}" if order.fill_price is not None else ""
            stop = f", stop {proposal.stop_loss}" if proposal.stop_loss is not None else ""
            take = f", target {proposal.take_profit}" if proposal.take_profit is not None else ""
            summary = (
                f"{order.side.value.capitalize()} {order.quantity} {order.symbol}"
                f"{fill}{stop}{take}. Order {order.status.value}"
                f"{f' ({order.reason})' if order.reason else ''}."
            )
            metadata = {
                "outcome": order.status.value,
                "order_id": order.id,
                "side": order.side.value,
                "quantity": float(order.quantity),
                "fill_price": float(order.fill_price) if order.fill_price else None,
                "stop_loss": float(proposal.stop_loss) if proposal.stop_loss else None,
                "take_profit": float(proposal.take_profit) if proposal.take_profit else None,
                "strategy": proposal.strategy_name,
            }

        if regime is not None:
            summary += f" Market regime at the time: {regime.label}."
        return MemoryEpisode(
            id=identifier,
            symbol=proposal.symbol,
            kind=EpisodeKind.TRADE,
            occurred_at=occurred_at,
            summary=summary,
            regime=regime,
            metadata=metadata,
        )
