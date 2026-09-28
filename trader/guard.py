"""
Order guard: a Claude Code hook that enforces hard limits on Robinhood orders.

Runs as a PreToolUse hook on the Robinhood order-placing tools and as a
PostToolUse hook on the review tools. Standard library only. Any error while
evaluating an order results in a deny: the guard fails closed.

Rules for placing an order:
  * TRADER_MODE=review (the default) blocks every order. Claude reviews the
    order and journals the decision instead.
  * Options and crypto orders are blocked unless explicitly enabled.
  * The symbol must be in TRADER_ALLOWED_SYMBOLS when that list is set.
  * The order must be sizeable from its own fields: a limit/stop price with a
    quantity, or a dollar_amount. Plain market orders by share count are
    blocked because their cost is unknown.
  * Buys may not exceed TRADER_MAX_ORDER_USD. Sells are not capped so an
    exit is never blocked by size.
  * At most TRADER_MAX_ORDERS_PER_DAY orders pass the guard per trading day.
  * The same symbol and side must have been reviewed with review_equity_order
    within TRADER_REVIEW_WINDOW_MINUTES.

Passing the guard does not approve the order: Claude Code's normal permission
prompt still asks the user.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from trader import config
from trader.journal import Journal

EQUITY_ORDER_TOOLS = {"place_equity_order"}
OPTION_ORDER_TOOLS = {"place_option_order", "exercise_option"}
CRYPTO_ORDER_TOOLS = {"place_crypto_order"}
ORDER_TOOLS = EQUITY_ORDER_TOOLS | OPTION_ORDER_TOOLS | CRYPTO_ORDER_TOOLS
REVIEW_TOOLS = {"review_equity_order"}


@dataclass
class Verdict:
    allowed: bool
    reason: str
    notional: Optional[float] = None


def tool_suffix(tool_name: str) -> str:
    return tool_name.rsplit("__", 1)[-1]


def _number(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def order_notional(tool_input: Dict[str, Any]) -> Optional[float]:
    """Dollar value of an equity order computed only from its own fields."""
    dollar_amount = _number(tool_input.get("dollar_amount"))
    if dollar_amount is not None:
        return dollar_amount
    quantity = _number(tool_input.get("quantity"))
    price = _number(tool_input.get("limit_price")) or _number(tool_input.get("stop_price"))
    if quantity is not None and price is not None:
        return quantity * price
    return None


def evaluate_order(
    tool: str,
    tool_input: Dict[str, Any],
    limits: config.Limits,
    journal: Journal,
    now: Optional[datetime] = None,
) -> Verdict:
    if limits.mode != "live":
        return Verdict(False, (
            "Review-only mode (TRADER_MODE=review): orders are never placed. "
            "Use review_equity_order to check the order, then record the decision with "
            "`python -m trader.journal decide ...`."
        ))
    if tool in OPTION_ORDER_TOOLS and not limits.allow_options:
        return Verdict(False, "Options orders are disabled (set TRADER_ALLOW_OPTIONS=true to enable).")
    if tool in CRYPTO_ORDER_TOOLS and not limits.allow_crypto:
        return Verdict(False, "Crypto orders are disabled (set TRADER_ALLOW_CRYPTO=true to enable).")
    if tool not in EQUITY_ORDER_TOOLS:
        # Options/crypto were explicitly enabled; only the daily cap applies.
        return _check_daily_cap(limits, journal, now) or Verdict(True, "Allowed by guard.")

    symbol = str(tool_input.get("symbol") or "").upper()
    side = str(tool_input.get("side") or "").lower()
    if not symbol or side not in ("buy", "sell"):
        return Verdict(False, "Order is missing a symbol or a buy/sell side.")
    if limits.allowed_symbols and symbol not in limits.allowed_symbols:
        return Verdict(False, f"{symbol} is not in TRADER_ALLOWED_SYMBOLS.")

    notional = order_notional(tool_input)
    if notional is None:
        return Verdict(False, (
            "Cannot size this order: a market order by share quantity has no price. "
            "Use a limit order (a marketable limit at the current ask) or dollar_amount."
        ))
    if side == "buy" and notional > limits.max_order_usd:
        return Verdict(False, (
            f"Buy of ${notional:,.2f} exceeds TRADER_MAX_ORDER_USD (${limits.max_order_usd:,.2f})."
        ), notional)

    capped = _check_daily_cap(limits, journal, now)
    if capped:
        capped.notional = notional
        return capped

    if journal.latest_review(symbol, side, limits.review_window_minutes, now=now) is None:
        return Verdict(False, (
            f"No review_equity_order for {side} {symbol} in the last "
            f"{limits.review_window_minutes} minutes. Review the order first."
        ), notional)

    return Verdict(True, f"Within limits: {side} {symbol} ${notional:,.2f}.", notional)


def _check_daily_cap(limits: config.Limits, journal: Journal, now: Optional[datetime]) -> Optional[Verdict]:
    placed = journal.orders_allowed_today(now=now)
    if placed >= limits.max_orders_per_day:
        return Verdict(False, (
            f"Daily order limit reached ({placed}/{limits.max_orders_per_day}, "
            "TRADER_MAX_ORDERS_PER_DAY)."
        ))
    return None


def deny_output(reason: str) -> Dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"Order guard: {reason}",
        }
    }


def handle(event: Dict[str, Any], journal: Optional[Journal] = None,
           limits: Optional[config.Limits] = None, now: Optional[datetime] = None) -> Tuple[Optional[Dict[str, Any]], int]:
    """Process one hook event. Returns (stdout JSON or None, exit code)."""
    hook_event = event.get("hook_event_name")
    tool = tool_suffix(str(event.get("tool_name") or ""))
    tool_input = event.get("tool_input") or {}

    if hook_event == "PostToolUse":
        if tool in REVIEW_TOOLS or tool in ORDER_TOOLS:
            journal = journal or Journal()
            journal.log_event(
                "review" if tool in REVIEW_TOOLS else "order_submitted",
                tool=tool,
                symbol=tool_input.get("symbol"),
                side=tool_input.get("side"),
                quantity=_number(tool_input.get("quantity")),
                price=_number(tool_input.get("limit_price")),
                notional=order_notional(tool_input) if tool in EQUITY_ORDER_TOOLS | REVIEW_TOOLS else None,
                payload={"input": tool_input, "response": event.get("tool_response")},
                now=now,
            )
        return None, 0

    if hook_event != "PreToolUse" or tool not in ORDER_TOOLS:
        return None, 0

    try:
        journal = journal or Journal()
        limits = limits or config.load_limits()
        verdict = evaluate_order(tool, tool_input, limits, journal, now=now)
    except Exception as exc:  # fail closed
        return deny_output(f"guard error, order blocked: {exc}"), 0

    journal.log_event(
        "order_allowed" if verdict.allowed else "order_blocked",
        tool=tool,
        symbol=tool_input.get("symbol"),
        side=tool_input.get("side"),
        quantity=_number(tool_input.get("quantity")),
        notional=verdict.notional,
        detail=verdict.reason,
        payload={"input": tool_input},
        now=now,
    )
    if verdict.allowed:
        # No decision: Claude Code's normal permission prompt still applies.
        return None, 0
    return deny_output(verdict.reason), 0


def main() -> int:
    try:
        config.load_env()
        event = json.load(sys.stdin)
        output, code = handle(event)
    except Exception as exc:
        # Unknown state: block if this could have been an order.
        print(f"Order guard failed: {exc}", file=sys.stderr)
        return 2
    if output is not None:
        print(json.dumps(output))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
