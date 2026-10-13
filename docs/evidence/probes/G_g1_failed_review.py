"""G1: does a FAILED review_equity_order response still satisfy the guard's review gate?"""
import json, tempfile, os
from datetime import datetime, timezone
from pathlib import Path

from trader import config, guard
from trader.journal import Journal

NOW = datetime(2026, 10, 13, 15, 0, 0, tzinfo=timezone.utc)

LIMITS = config.Limits(
    mode="live",
    max_order_usd=500.0,
    max_orders_per_day=3,
    review_window_minutes=15,
    allowed_symbols=frozenset(),
)

FAILURE_SHAPES = {
  "mcp_isError_true": {
      "isError": True,
      "content": [{"type": "text", "text": "Error: insufficient buying power to review this order"}],
  },
  "plain_error_object": {"error": "ORDER_REVIEW_FAILED", "message": "symbol not tradable"},
  "rejection_verdict": {"status": "rejected", "reason": "pattern day trader restriction",
                        "can_place": False},
  "mcp_error_result": {"content": [{"type": "text",
      "text": '{"error":{"code":-32000,"message":"Robinhood API 403: account restricted"}}'}],
      "isError": True},
  "empty_response": None,
  "string_error": "Error executing tool review_equity_order: upstream 500",
}

ORDER_INPUT = {"symbol": "AAPL", "side": "buy", "quantity": 2, "limit_price": 100.0,
               "type": "limit", "time_in_force": "gfd"}

for name, resp in FAILURE_SHAPES.items():
    with tempfile.TemporaryDirectory() as td:
        j = Journal(Path(td) / "j.db")
        post = {
            "hook_event_name": "PostToolUse",
            "tool_name": "mcp__Robinhood__review_equity_order",
            "tool_input": {"symbol": "AAPL", "side": "buy", "quantity": 2, "limit_price": 100.0},
            "tool_response": resp,
        }
        out, code = guard.handle(post, journal=j, limits=LIMITS, now=NOW)
        rows = [dict(r) for r in j.events(10)]
        logged_kind = rows[0]["kind"] if rows else None

        pre = {
            "hook_event_name": "PreToolUse",
            "tool_name": "mcp__Robinhood__place_equity_order",
            "tool_input": ORDER_INPUT,
        }
        out2, code2 = guard.handle(pre, journal=j, limits=LIMITS, now=NOW)
        allowed = out2 is None
        rev = j.latest_review("AAPL", "buy", 15, now=NOW)
        print(f"[{name}]")
        print(f"   PostToolUse logged kind = {logged_kind!r}")
        print(f"   latest_review() -> {'ROW FOUND' if rev is not None else 'None'}")
        print(f"   PreToolUse place_equity_order -> {'ALLOWED (no deny output)' if allowed else 'DENIED: ' + json.dumps(out2)}")
        last = [dict(r) for r in j.events(1)][0]
        print(f"   journal last event: kind={last['kind']!r} detail={last['detail']!r}")
        print()
