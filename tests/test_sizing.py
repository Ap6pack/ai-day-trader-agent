from __future__ import annotations

import pytest

from trader.config import Sizing
from trader.sizing import size_trade

RULES = Sizing(max_position_pct=0.10, max_symbol_pct=0.25, stop_loss_pct=3.0)


def test_buy_within_limits_with_stop_loss():
    decision = size_trade("aapl", "buy", 200.0, 10_000, quantity=4, rules=RULES)
    assert decision.approved
    assert decision.symbol == "AAPL"
    assert decision.max_quantity == 5  # $1,000 per-order limit / $200
    assert decision.suggested_stop_loss == 194.0


def test_max_quantity_without_proposed_quantity():
    decision = size_trade("AAPL", "buy", 200.0, 10_000, rules=RULES)
    assert decision.approved and decision.quantity is None
    assert decision.max_quantity == 5


def test_buy_over_per_order_limit():
    decision = size_trade("AAPL", "buy", 200.0, 10_000, quantity=6, rules=RULES)
    assert not decision.approved
    assert any("of equity" in v for v in decision.violations)


def test_buy_respects_existing_position_and_buying_power():
    decision = size_trade("AAPL", "buy", 200.0, 10_000, quantity=3,
                          current_position_value=2_200, buying_power=5_000, rules=RULES)
    assert not decision.approved
    assert decision.max_quantity == 1  # $300 of room under the $2,500 cap

    decision = size_trade("AAPL", "buy", 200.0, 10_000, quantity=3, buying_power=500, rules=RULES)
    assert not decision.approved
    assert decision.max_quantity == 2


def test_sell_cannot_exceed_shares_held():
    decision = size_trade("AAPL", "sell", 200.0, 10_000, quantity=10, shares_held=4, rules=RULES)
    assert not decision.approved
    assert decision.suggested_stop_loss is None
    assert decision.max_quantity == 4


@pytest.mark.parametrize("kwargs", [
    {"side": "hold"}, {"quantity": 0}, {"price": 0}, {"account_equity": 0},
])
def test_invalid_inputs_raise(kwargs):
    args = {"symbol": "AAPL", "side": "buy", "price": 10.0, "account_equity": 1_000,
            "quantity": 1, "rules": RULES, **kwargs}
    with pytest.raises(ValueError):
        size_trade(**args)
