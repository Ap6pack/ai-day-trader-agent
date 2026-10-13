"""Non-finite numeric configuration must be refused, not silently obeyed.

float() accepts "nan", "NaN", "inf", "infinity" and "-inf". A NaN limit does not
fail — it stops being enforced, because every comparison against NaN is False.
`notional > max_order_usd` is False for NaN, so the cap never trips and nothing
tells the operator. These tests pin that every money limit refuses a non-finite
value instead.
"""

from __future__ import annotations

import pytest

from trader import alpaca, autotrader, config, dividends, guard
from trader.journal import Journal

NON_FINITE = ["nan", "NaN", "NAN", "inf", "-inf", "Infinity", "infinity"]


@pytest.fixture
def journal(tmp_path):
    return Journal(tmp_path / "j.db")


# --- the validator itself ---------------------------------------------------


@pytest.mark.parametrize("raw", NON_FINITE)
def test_finite_rejects_non_finite_values(raw):
    with pytest.raises(ValueError, match="finite"):
        config.finite("SOME_LIMIT", float(raw))


@pytest.mark.parametrize("raw", NON_FINITE)
def test_env_float_rejects_non_finite_values(monkeypatch, raw):
    monkeypatch.setenv("SOME_LIMIT", raw)
    with pytest.raises(ValueError, match="SOME_LIMIT"):
        config.env_float("SOME_LIMIT", 1.0)


def test_env_float_still_falls_back_for_unparseable_values(monkeypatch):
    """Only the non-finite policy changes; the forgiving parse is preserved."""
    monkeypatch.setenv("SOME_LIMIT", "5OO")  # letter O
    assert config.env_float("SOME_LIMIT", 500.0) == 500.0
    with pytest.raises(ValueError):
        config.env_float("SOME_LIMIT", 500.0, strict=True)


def test_env_float_reads_normal_values(monkeypatch):
    monkeypatch.setenv("SOME_LIMIT", "250.5")
    assert config.env_float("SOME_LIMIT", 500.0) == 250.5
    monkeypatch.delenv("SOME_LIMIT")
    assert config.env_float("SOME_LIMIT", 500.0) == 500.0


# --- the Robinhood order guard ----------------------------------------------


@pytest.mark.parametrize("raw", NON_FINITE)
def test_robinhood_order_cap_refuses_a_non_finite_limit(monkeypatch, raw):
    """Before this, TRADER_MAX_ORDER_USD=nan let a $10,000 buy through."""
    monkeypatch.setenv("TRADER_MODE", "live")
    monkeypatch.setenv("TRADER_MAX_ORDER_USD", raw)
    with pytest.raises(ValueError, match="TRADER_MAX_ORDER_USD"):
        config.load_limits()


def test_guard_fails_closed_on_a_non_finite_cap(monkeypatch, journal):
    """The guard catches the load error and denies, rather than running uncapped."""
    monkeypatch.setenv("TRADER_MODE", "live")
    monkeypatch.setenv("TRADER_MAX_ORDER_USD", "nan")
    output, code = guard.handle(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "mcp__Robinhood__place_equity_order",
            "tool_input": {"symbol": "AAPL", "side": "buy", "quantity": 100, "limit_price": 100.0},
        },
        journal=journal,
    )
    assert code == 0
    assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "TRADER_MAX_ORDER_USD" in output["hookSpecificOutput"]["permissionDecisionReason"]


# --- the Alpaca real-money limits -------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["ALPACA_LIVE_MAX_ORDER_USD", "ALPACA_LIVE_MAX_DAILY_LOSS_USD"],
)
@pytest.mark.parametrize("raw", ["nan", "inf"])
def test_live_money_limits_refuse_a_non_finite_value(monkeypatch, name, raw):
    """ALPACA_LIVE_MAX_DAILY_LOSS_USD=nan silently disabled the real-money
    daily loss stop: day_pl <= -nan is always False."""
    monkeypatch.setenv(name, raw)
    with pytest.raises(ValueError, match=name):
        alpaca.LiveLimits.from_env()


def test_max_live_qty_refuses_a_non_finite_order_cap(monkeypatch):
    monkeypatch.setenv("ALPACA_LIVE_MAX_ORDER_USD", "nan")
    with pytest.raises(ValueError, match="ALPACA_LIVE_MAX_ORDER_USD"):
        alpaca.max_live_qty(100.0)


# --- the desk autopilot limits ----------------------------------------------


@pytest.mark.parametrize("name", ["AUTOPILOT_MIN_CONFIDENCE", "AUTOPILOT_MIN_PRICE"])
def test_autopilot_limits_refuse_a_non_finite_value(monkeypatch, name):
    """confidence < nan is always False, so a NaN floor admits every signal."""
    monkeypatch.setenv(name, "nan")
    with pytest.raises(ValueError, match=name):
        autotrader.Limits.from_env()


# --- strategy thresholds ----------------------------------------------------


def test_dividend_threshold_refuses_a_non_finite_value(monkeypatch):
    monkeypatch.setenv("MIN_DIVIDEND_YIELD", "nan")
    with pytest.raises(ValueError, match="MIN_DIVIDEND_YIELD"):
        dividends.dividend_signal("KO", 50.0, rows=[])


# --- the shipped defaults still load ----------------------------------------


def test_shipped_defaults_are_unaffected(monkeypatch):
    for name in (
        "TRADER_MAX_ORDER_USD", "ALPACA_LIVE_MAX_ORDER_USD", "ALPACA_LIVE_MAX_DAILY_LOSS_USD",
        "AUTOPILOT_MIN_CONFIDENCE", "AUTOPILOT_MIN_PRICE", "MIN_DIVIDEND_YIELD", "TRADER_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    assert config.load_limits().max_order_usd == 500.0
    assert alpaca.LiveLimits.from_env() == alpaca.LiveLimits(500.0, 3, 200.0)
    assert autotrader.Limits.from_env().min_confidence == 0.5
