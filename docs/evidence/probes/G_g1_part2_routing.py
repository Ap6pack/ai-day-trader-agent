"""G1 part 2: separate the two worlds.
 (A) tool ERRORED -> Claude Code routes it to PostToolUseFailure, which settings.json does
     NOT register -> guard never sees it -> is the gate still closed?
 (B) tool SUCCEEDED but the review payload says "do not place" -> PostToolUse DOES fire.
"""
import json, tempfile
from datetime import datetime, timezone
from pathlib import Path
from trader import config, guard
from trader.journal import Journal

NOW = datetime(2026, 10, 13, 15, 0, 0, tzinfo=timezone.utc)
LIMITS = config.Limits(mode="live", max_order_usd=500.0, max_orders_per_day=3,
                       review_window_minutes=15, allowed_symbols=frozenset())
ORDER = {"symbol": "AAPL", "side": "buy", "quantity": 2, "limit_price": 100.0}

def place(j):
    out, _ = guard.handle({"hook_event_name": "PreToolUse",
                           "tool_name": "mcp__Robinhood__place_equity_order",
                           "tool_input": ORDER}, journal=j, limits=LIMITS, now=NOW)
    return "ALLOWED" if out is None else "DENIED: " + out["hookSpecificOutput"]["permissionDecisionReason"]

# --- (A) control: the review tool ERRORED, so only PostToolUseFailure would fire.
with tempfile.TemporaryDirectory() as td:
    j = Journal(Path(td) / "j.db")
    out, code = guard.handle({"hook_event_name": "PostToolUseFailure",
                              "tool_name": "mcp__Robinhood__review_equity_order",
                              "tool_input": ORDER,
                              "error": "Robinhood API 403: account restricted"},
                             journal=j, limits=LIMITS, now=NOW)
    print("(A) PostToolUseFailure event fed to guard.handle():")
    print("    guard returned:", out, code)
    print("    events logged :", [dict(r)["kind"] for r in j.events(10)])
    print("    then place_equity_order ->", place(j))

# --- (B) the REAL path: review succeeded, payload is a negative verdict / hard alert.
SUCCESSFUL_BUT_NEGATIVE = {
 "would_be_rejected": {
   "alerts": [{"type": "error", "severity": "blocking",
               "title": "Order cannot be placed",
               "message": "Insufficient buying power. This order would be rejected."}],
   "can_place": False, "estimated_cost": "200.00"},
 "pdt_block": {
   "alerts": [{"type": "pattern_day_trade", "severity": "blocking",
               "message": "Placing this order would flag your account as a pattern day trader."}],
   "can_place": False},
 "collateral_warning": {
   "alerts": [{"type": "warning", "message": "Low liquidity; marketable limit may not fill."}],
   "can_place": True},
}
for name, resp in SUCCESSFUL_BUT_NEGATIVE.items():
    with tempfile.TemporaryDirectory() as td:
        j = Journal(Path(td) / "j.db")
        guard.handle({"hook_event_name": "PostToolUse",
                      "tool_name": "mcp__Robinhood__review_equity_order",
                      "tool_input": ORDER, "tool_response": resp},
                     journal=j, limits=LIMITS, now=NOW)
        row = j.latest_review("AAPL", "buy", 15, now=NOW)
        print(f"(B) [{name}] review row stored = {row is not None};"
              f" can_place in payload = {resp.get('can_place')}")
        print("    stored payload['response'] =", json.loads(dict(row)['payload'])['response'])
        print("    place_equity_order ->", place(j))
