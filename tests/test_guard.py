from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trader import guard
from trader.config import Limits
from trader.journal import Journal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 28, 15, 0, tzinfo=timezone.utc)  # 11:00 ET
LIVE = Limits(mode="live", max_order_usd=500, max_orders_per_day=2, review_window_minutes=15)
PLACE = "mcp__robinhood-trading__place_equity_order"
REVIEW = "mcp__robinhood-trading__review_equity_order"


@pytest.fixture
def journal(tmp_path):
    return Journal(tmp_path / "journal.db")


def order(**overrides):
    return {"symbol": "AAPL", "side": "buy", "type": "limit", "quantity": "2",
            "limit_price": "200", **overrides}


def pre(tool_input, tool=PLACE):
    return {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": tool_input}


def post(tool_input, tool=REVIEW):
    return {"hook_event_name": "PostToolUse", "tool_name": tool, "tool_input": tool_input,
            "tool_response": {"ok": True}}


def run(event, journal, limits=LIVE, now=NOW):
    output, code = guard.handle(event, journal=journal, limits=limits, now=now)
    assert code == 0
    if output is None:
        return None
    decision = output["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    return decision["permissionDecisionReason"]


def reviewed(journal, minutes_ago=5, **overrides):
    guard.handle(post(order(**overrides)), journal=journal, now=NOW - timedelta(minutes=minutes_ago))


def test_review_mode_blocks_every_order(journal):
    reviewed(journal)
    reason = run(pre(order()), journal, limits=Limits(mode="review"))
    assert "Review-only mode" in reason
    assert journal.events()[0]["kind"] == "order_blocked"


def test_reviewed_order_within_limits_passes_to_permission_prompt(journal):
    reviewed(journal)
    assert run(pre(order()), journal) is None
    assert journal.events()[0]["kind"] == "order_allowed"
    assert journal.events()[0]["notional"] == 400


def test_order_without_recent_review_is_blocked(journal):
    assert "No review_equity_order" in run(pre(order()), journal)
    reviewed(journal, minutes_ago=30)
    assert "No review_equity_order" in run(pre(order()), journal)
    reviewed(journal, side="sell")
    assert "No review_equity_order" in run(pre(order()), journal)


def test_buy_over_dollar_cap_is_blocked_but_sell_is_not(journal):
    reviewed(journal, quantity="3")
    assert "exceeds TRADER_MAX_ORDER_USD" in run(pre(order(quantity="3")), journal)
    reviewed(journal, side="sell", quantity="10")
    assert run(pre(order(side="sell", quantity="10")), journal) is None


def test_dollar_amount_orders_are_sized(journal):
    reviewed(journal)
    assert run(pre({"symbol": "AAPL", "side": "buy", "type": "market", "dollar_amount": "450"}), journal) is None
    assert "exceeds" in run(pre({"symbol": "AAPL", "side": "buy", "type": "market",
                                 "dollar_amount": "900"}), journal)


def test_market_order_by_quantity_cannot_be_sized(journal):
    reviewed(journal)
    reason = run(pre({"symbol": "AAPL", "side": "buy", "type": "market", "quantity": "1"}), journal)
    assert "Cannot size" in reason


def test_daily_cap_counts_orders_since_eastern_midnight(journal):
    for _ in range(2):
        reviewed(journal)
        assert run(pre(order(quantity="1")), journal) is None
    reviewed(journal)
    assert "Daily order limit" in run(pre(order(quantity="1")), journal)
    # Next trading day resets the count.
    tomorrow = NOW + timedelta(days=1)
    reviewed(journal, minutes_ago=-(24 * 60 - 5))
    assert run(pre(order(quantity="1")), journal, now=tomorrow) is None


def test_symbol_allowlist(journal):
    limits = Limits(mode="live", allowed_symbols=frozenset({"MSFT"}))
    reviewed(journal)
    assert "not in TRADER_ALLOWED_SYMBOLS" in run(pre(order()), journal, limits=limits)


@pytest.mark.parametrize("tool", [
    "mcp__Robinhood__place_option_order",
    "mcp__Robinhood__exercise_option",
    "mcp__Robinhood__place_crypto_order",
])
def test_options_and_crypto_blocked_by_default(journal, tool):
    assert "disabled" in run(pre({"symbol": "AAPL"}, tool=tool), journal)


# --- enabling options/crypto must not disable every other limit -------------

OPTIONS_ON = Limits(mode="live", max_order_usd=500, max_orders_per_day=2,
                    review_window_minutes=15, allow_options=True)
CRYPTO_ON = Limits(mode="live", max_order_usd=500, max_orders_per_day=2,
                   review_window_minutes=15, allow_crypto=True)
OPTION_PLACE = "mcp__robinhood-trading__place_option_order"
CRYPTO_PLACE = "mcp__robinhood-trading__place_crypto_order"


def test_option_buy_respects_the_dollar_cap(journal):
    """1000 contracts at $50 is $5,000,000 of exposure, not $50,000: a US
    equity option contract is 100 shares."""
    event = pre(order(quantity="1000", limit_price="50"), tool=OPTION_PLACE)
    reason = run(event, journal, limits=OPTIONS_ON)
    assert "exceeds TRADER_MAX_ORDER_USD" in reason
    assert "$5,000,000.00" in reason and "1000 contracts x 100 shares" in reason


def test_crypto_buy_respects_the_dollar_cap(journal):
    event = pre(order(symbol="BTC", quantity="10", limit_price="90000"), tool=CRYPTO_PLACE)
    reason = run(event, journal, limits=CRYPTO_ON)
    assert "exceeds TRADER_MAX_ORDER_USD" in reason and "$900,000.00" in reason


@pytest.mark.parametrize("tool, limits", [(OPTION_PLACE, OPTIONS_ON), (CRYPTO_PLACE, CRYPTO_ON)])
def test_non_equity_respects_the_symbol_allowlist(journal, tool, limits):
    limits = Limits(**{**limits.__dict__, "allowed_symbols": frozenset({"MSFT"})})
    reason = run(pre(order(quantity="1", limit_price="1"), tool=tool), journal, limits=limits)
    assert "not in TRADER_ALLOWED_SYMBOLS" in reason


@pytest.mark.parametrize("tool, limits", [(OPTION_PLACE, OPTIONS_ON), (CRYPTO_PLACE, CRYPTO_ON)])
def test_non_equity_market_orders_are_unsizeable_and_denied(journal, tool, limits):
    """Same rule as equity: no price in the payload means no cap can be applied."""
    event = pre({"symbol": "AAPL", "side": "buy", "type": "market", "quantity": "5"}, tool=tool)
    assert "Cannot size this order" in run(event, journal, limits=limits)


@pytest.mark.parametrize("tool, limits", [(OPTION_PLACE, OPTIONS_ON), (CRYPTO_PLACE, CRYPTO_ON)])
def test_non_equity_within_the_cap_is_allowed(journal, tool, limits):
    qty, price = ("1", "2") if tool == OPTION_PLACE else ("1", "200")
    event = pre(order(quantity=qty, limit_price=price), tool=tool)
    assert run(event, journal, limits=limits) is None  # no deny


def test_exercise_still_honours_the_allowlist_and_the_daily_cap(journal):
    """Exercise cannot be priced from the request, so it keeps the checks that
    do not need a price rather than skipping all of them."""
    limits = Limits(**{**OPTIONS_ON.__dict__, "allowed_symbols": frozenset({"MSFT"})})
    tool = "mcp__robinhood-trading__exercise_option"
    assert "not in TRADER_ALLOWED_SYMBOLS" in run(pre({"symbol": "AAPL"}, tool=tool), journal, limits=limits)
    assert run(pre({"symbol": "MSFT"}, tool=tool), journal, limits=limits) is None


def test_a_zero_limit_price_is_not_a_price(journal):
    """limit_price=0 must not size the order at $0 and sail past the cap."""
    event = pre(order(quantity="1000", limit_price="0"))
    assert "Cannot size this order" in run(event, journal)


@pytest.mark.parametrize("amount", ["nan", "inf", "-100", "0"])
def test_an_unusable_dollar_amount_does_not_short_circuit_the_cap(journal, amount):
    """dollar_amount bypasses the quantity x price path, so it must be usable
    on its own. NaN passes every comparison and a zero or negative amount reads
    as free, either of which would wave through the 10,000 shares at $900 that
    sit beside it in the same payload."""
    reviewed(journal)
    event = pre(order(dollar_amount=amount, quantity="10000", limit_price="900"))
    assert "Cannot size this order" in run(event, journal)


def test_a_usable_dollar_amount_is_still_honoured(journal):
    reviewed(journal)
    assert run(pre(order(dollar_amount="400", quantity="10000", limit_price="900")),
               journal) is None                              # within the $500 cap
    assert "exceeds TRADER_MAX_ORDER_USD" in run(
        pre(order(dollar_amount="600", quantity="1", limit_price="1")), journal)


def test_option_review_and_placement_record_the_same_notional(journal):
    """The contract multiplier belongs to the instrument, not the action.

    Deriving it from the order tools alone journalled a review of 2 contracts
    at $3 as $6 while the matching placement journalled $600 - a 100x
    disagreement between two records of one order.
    """
    payload = {"symbol": "NVDA", "side": "buy", "quantity": "2", "limit_price": "3"}
    for tool in ("review_option_order", "place_option_order"):
        guard.handle(post(payload, tool=f"mcp__robinhood-trading__{tool}"), journal=journal)

    notionals = {row["tool"]: row["notional"] for row in journal.events(5)}
    assert notionals["review_option_order"] == notionals["place_option_order"] == 600.0, (
        f"review and placement disagree on the same order: {notionals}"
    )


def test_equity_review_is_not_given_a_contract_multiplier(journal):
    """Only option tools carry the 100x; equity reviews must stay at face value."""
    payload = {"symbol": "AAPL", "side": "buy", "quantity": "2", "limit_price": "3"}
    guard.handle(post(payload), journal=journal)
    assert journal.events(1)[0]["notional"] == 6.0


def test_non_order_tools_are_ignored(journal):
    assert run(pre({"symbols": ["AAPL"]}, tool="mcp__Robinhood__get_equity_quotes"), journal) is None
    assert journal.events() == []


def test_guard_errors_fail_closed(journal):
    broken = Limits(mode="live")
    reviewed(journal)
    reason = run(pre(order(quantity="not-a-number")), journal, limits=broken)
    assert "guard error" in reason


def _run_hook(payload: str, tmp_path: Path):
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(PROJECT_ROOT),
           "TRADER_JOURNAL_PATH": str(tmp_path / "hook.db")}
    return subprocess.run(
        [str(PROJECT_ROOT / ".claude/hooks/order_guard.sh")],
        input=payload, capture_output=True, text=True, env=env, timeout=30,
    )


def test_hook_script_denies_in_review_mode(tmp_path, monkeypatch):
    result = _run_hook(json.dumps(pre(order())), tmp_path)
    assert result.returncode in (0, 2)
    if result.returncode == 0:
        # .env may set TRADER_MODE=live on a developer machine; either way no silent pass
        # without a review on record.
        decision = json.loads(result.stdout)["hookSpecificOutput"]
        assert decision["permissionDecision"] == "deny"


def test_hook_script_blocks_on_malformed_input(tmp_path):
    result = _run_hook("not json", tmp_path)
    assert result.returncode == 2
    assert "blocking" in result.stderr


def test_project_settings_register_the_guard():
    """The hook only runs if .claude/settings.json exists and matches the order tools."""
    import re

    settings_path = PROJECT_ROOT / ".claude/settings.json"
    assert settings_path.exists(), ".claude/settings.json is missing: the order guard is not registered"
    settings = json.loads(settings_path.read_text())

    pre = settings["hooks"]["PreToolUse"]
    post = settings["hooks"]["PostToolUse"]
    commands = [h["command"] for entry in pre + post for h in entry["hooks"]]
    assert all(c.endswith("/.claude/hooks/order_guard.sh") for c in commands)

    pre_matchers = [re.compile(entry["matcher"]) for entry in pre]
    post_matchers = [re.compile(entry["matcher"]) for entry in post]
    for server in ("robinhood-trading", "claude_ai_Robinhood", "Robinhood"):
        for tool in ("place_equity_order", "place_option_order", "place_crypto_order", "exercise_option"):
            assert any(m.search(f"mcp__{server}__{tool}") for m in pre_matchers), (server, tool)
        assert any(m.search(f"mcp__{server}__review_equity_order") for m in post_matchers), server
        assert not any(m.search(f"mcp__{server}__get_equity_quotes") for m in pre_matchers)

    deny = settings["permissions"]["deny"]
    assert "Read(/.env)" in deny
