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
