from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.risk_gate import check_trade

CONFIG = SimpleNamespace(
    MAX_POSITION_PERCENTAGE=0.10,
    MAX_PORTFOLIO_ALLOCATION=0.25,
    MAX_DAILY_TRADES=3,
    STOP_LOSS_PERCENTAGE=3.0,
)


def test_buy_within_limits_is_approved_with_stop_loss():
    decision = check_trade("aapl", "buy", 4, 200.0, account_equity=10_000, config=CONFIG)
    assert decision.approved
    assert decision.symbol == "AAPL"
    assert decision.order_value == 800.0
    assert decision.max_quantity == 5  # $1,000 per-order limit / $200
    assert decision.suggested_stop_loss == 194.0


def test_buy_over_per_order_limit_is_rejected():
    decision = check_trade("AAPL", "BUY", 6, 200.0, account_equity=10_000, config=CONFIG)
    assert not decision.approved
    assert any("per-order limit" in v for v in decision.violations)
    assert decision.max_quantity == 5


def test_buy_respects_existing_position_and_buying_power():
    decision = check_trade(
        "AAPL", "BUY", 3, 200.0, account_equity=10_000,
        current_position_value=2_200, buying_power=5_000, config=CONFIG,
    )
    assert not decision.approved
    assert any("would be" in v for v in decision.violations)
    assert decision.max_quantity == 1  # only $300 of room left under the $2,500 cap

    decision = check_trade("AAPL", "BUY", 3, 200.0, account_equity=10_000,
                           buying_power=500, config=CONFIG)
    assert not decision.approved
    assert decision.max_quantity == 2


def test_daily_trade_limit_blocks_everything():
    decision = check_trade("AAPL", "BUY", 1, 10.0, account_equity=10_000,
                           trades_today=3, config=CONFIG)
    assert not decision.approved
    assert decision.max_quantity == 0


def test_sell_cannot_exceed_shares_held():
    decision = check_trade("AAPL", "SELL", 10, 200.0, account_equity=10_000,
                           shares_held=4, config=CONFIG)
    assert not decision.approved
    assert decision.suggested_stop_loss is None
    assert decision.max_quantity == 4


@pytest.mark.parametrize("kwargs", [
    {"side": "HOLD"}, {"quantity": 0}, {"price": 0}, {"account_equity": 0},
])
def test_invalid_inputs_raise(kwargs):
    args = {"symbol": "AAPL", "side": "BUY", "quantity": 1, "price": 10.0,
            "account_equity": 1_000, "config": CONFIG, **kwargs}
    with pytest.raises(ValueError):
        check_trade(**args)
