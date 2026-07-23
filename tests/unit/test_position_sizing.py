"""Golden-value tests for position sizing."""

from __future__ import annotations

from decimal import Decimal

import pytest

from atp.domain.services import (
    kelly_fraction,
    max_affordable,
    size_by_fraction,
    size_by_risk,
)


class TestSizeByRisk:
    def test_textbook_case(self) -> None:
        # 1% of 100k = 1000 risk budget; $5 per-share risk -> 200 shares.
        shares = size_by_risk(Decimal("100000"), 1.0, Decimal("100"), Decimal("95"))
        assert shares == Decimal("200")

    def test_rounds_down_to_whole_shares(self) -> None:
        # 1000 / 5.5 = 181.81... -> 181
        shares = size_by_risk(Decimal("100000"), 1.0, Decimal("100"), Decimal("94.50"))
        assert shares == Decimal("181")

    def test_stop_above_entry_rejected(self) -> None:
        with pytest.raises(ValueError, match="below entry"):
            size_by_risk(Decimal("100000"), 1.0, Decimal("100"), Decimal("105"))

    def test_non_positive_equity_rejected(self) -> None:
        with pytest.raises(ValueError, match="equity"):
            size_by_risk(Decimal("0"), 1.0, Decimal("100"), Decimal("95"))


class TestSizeByFraction:
    def test_fraction_of_equity(self) -> None:
        # 10% of 50k = 5000; at $123 -> 40 shares
        assert size_by_fraction(Decimal("50000"), 10.0, Decimal("123")) == Decimal("40")


class TestMaxAffordable:
    def test_floor_division(self) -> None:
        assert max_affordable(Decimal("1000"), Decimal("333")) == Decimal("3")


class TestKellyFraction:
    def test_textbook_case(self) -> None:
        # W=0.6, R=2 -> 0.6 - 0.4/2 = 0.4
        assert kelly_fraction(0.6, 2.0) == pytest.approx(0.4)

    def test_negative_edge_clamps_to_zero(self) -> None:
        assert kelly_fraction(0.3, 1.0) == 0.0

    def test_invalid_inputs_rejected(self) -> None:
        with pytest.raises(ValueError, match="win_rate"):
            kelly_fraction(1.5, 2.0)
        with pytest.raises(ValueError, match="payoff_ratio"):
            kelly_fraction(0.5, 0.0)
