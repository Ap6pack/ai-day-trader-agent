"""
Small Alpaca REST client for the news bot, the desk autopilot and the desk:
market clock, latest price, account views and order entry.

Paper is the default. Real money is opt-in and needs both
    ALPACA_TRADING_BASE_URL=https://api.alpaca.markets
    ALPACA_LIVE_TRADING=true
Any other host, or the live host without the flag, makes every account and
order function raise (fail closed).

On the live account every buy is checked here, whichever agent sends it,
against ALPACA_LIVE_MAX_ORDER_USD, ALPACA_LIVE_MAX_ORDERS_PER_DAY and
ALPACA_LIVE_MAX_DAILY_LOSS_USD. Sells are never blocked, so positions can
always be exited. close_all_positions() is paper only: on a live account the
agents close only what they opened (flatten_owned), never your own holdings.
"""

from __future__ import annotations

import math
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests

from trader.config import secret
from trader.news_sources import alpaca_data_url, alpaca_headers

PAPER_HOST = "paper-api.alpaca.markets"
LIVE_HOST = "api.alpaca.markets"
DEFAULT_TRADING_URL = f"https://{PAPER_HOST}"
LIVE_TRADING_URL = f"https://{LIVE_HOST}"
MARKET_TZ = ZoneInfo("America/New_York")
OPEN_ORDER_STATUSES = ("new", "accepted", "pending_new", "partially_filled", "held", "accepted_for_bidding")
DEFAULT_DATA_FEED = "iex"
TIMEOUT_SECONDS = 10
CLIENT_ORDER_ID_NAMESPACE = uuid.UUID("5b0f3a3e-2f6c-4a4e-9a51-6f1f3e2b7c10")


class AccountError(RuntimeError):
    """The trading host is not a usable Alpaca account (unknown host, or live without opt-in)."""


# Older name, kept for callers that catch it.
NotPaperError = AccountError


class LiveLimitError(RuntimeError):
    """A live buy was refused by the ALPACA_LIVE_* limits."""


def trading_base_url() -> str:
    base = (secret("ALPACA_TRADING_BASE_URL") or DEFAULT_TRADING_URL).rstrip("/")
    if base.endswith("/v2"):
        base = base[: -len("/v2")]
    return base


def live_opt_in() -> bool:
    return (secret("ALPACA_LIVE_TRADING") or "").strip().lower() == "true"


def account_mode() -> str:
    """'paper' or 'live' (real money). Raises AccountError for anything else."""
    parsed = urlparse(trading_base_url())
    if parsed.scheme == "https" and parsed.hostname == PAPER_HOST and not parsed.port:
        return "paper"
    if parsed.scheme == "https" and parsed.hostname == LIVE_HOST and not parsed.port:
        if live_opt_in():
            return "live"
        raise AccountError(
            f"Refusing to trade: ALPACA_TRADING_BASE_URL is the LIVE (real money) host but "
            f"ALPACA_LIVE_TRADING is not true. Set it to true to trade real money, or use {DEFAULT_TRADING_URL}.")
    raise AccountError(
        f"Refusing to trade: ALPACA_TRADING_BASE_URL is {trading_base_url()!r}; "
        f"use {DEFAULT_TRADING_URL} (paper) or {LIVE_TRADING_URL} (live, with ALPACA_LIVE_TRADING=true).")


def _mode_or_none() -> Optional[str]:
    try:
        return account_mode()
    except AccountError:
        return None


def is_paper() -> bool:
    return _mode_or_none() == "paper"


def is_live() -> bool:
    return _mode_or_none() == "live"


def require_mode(expected: str, who: str = "This agent") -> None:
    """Fail closed unless the configured account is the one the caller was set up for."""
    mode = account_mode()
    if mode != expected:
        raise AccountError(f"{who} is set to trade {expected.upper()} but the Alpaca account is "
                           f"{mode.upper()} (ALPACA_TRADING_BASE_URL / ALPACA_LIVE_TRADING).")


def _require_account() -> None:
    account_mode()  # any usable account; order entry adds the live limits


# --- real-money limits ---------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class LiveLimits:
    max_order_usd: float = 500.0
    max_orders_per_day: int = 3
    max_daily_loss_usd: float = 200.0

    @classmethod
    def from_env(cls) -> "LiveLimits":
        return cls(
            max_order_usd=_env_float("ALPACA_LIVE_MAX_ORDER_USD", 500.0),
            max_orders_per_day=int(_env_float("ALPACA_LIVE_MAX_ORDERS_PER_DAY", 3)),
            max_daily_loss_usd=_env_float("ALPACA_LIVE_MAX_DAILY_LOSS_USD", 200.0),
        )


def market_day_start(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    local = now.astimezone(MARKET_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    return local.isoformat()


def max_live_qty(price: float, limits: Optional[LiveLimits] = None) -> int:
    """Whole shares that fit under ALPACA_LIVE_MAX_ORDER_USD at this price."""
    limits = limits or LiveLimits.from_env()
    return math.floor(limits.max_order_usd / price) if price > 0 else 0


def check_live_buy(payload: Dict[str, Any], limits: Optional[LiveLimits] = None,
                   now: Optional[datetime] = None) -> None:
    """Raise LiveLimitError if this live buy breaks a limit. Sells always pass."""
    if str(payload.get("side", "")).lower() != "buy":
        return
    limits = limits or LiveLimits.from_env()
    symbol, qty = str(payload["symbol"]).upper(), float(payload["qty"])
    price = latest_price(symbol)
    notional = qty * price
    if notional > limits.max_order_usd + 1e-6:
        raise LiveLimitError(f"LIVE buy {qty:g} {symbol} ~${notional:,.2f} exceeds ALPACA_LIVE_MAX_ORDER_USD "
                             f"${limits.max_order_usd:,.2f}")
    acct = account()
    day_pl = float(acct.get("equity") or 0) - float(acct.get("last_equity") or 0)
    if day_pl <= -limits.max_daily_loss_usd:
        raise LiveLimitError(f"LIVE buys stopped for today: day P&L ${day_pl:,.2f} hit ALPACA_LIVE_MAX_DAILY_LOSS_USD "
                             f"-${limits.max_daily_loss_usd:,.2f}")
    todays = orders(limit=500, status="all", after=market_day_start(now))
    buys = [o for o in todays if o.get("side") == "buy" and o.get("status") not in ("rejected", "canceled")]
    if len(buys) >= limits.max_orders_per_day:
        raise LiveLimitError(f"LIVE daily buy limit reached ({len(buys)}/{limits.max_orders_per_day}, "
                             f"ALPACA_LIVE_MAX_ORDERS_PER_DAY)")


def _request(method: str, url: str, **kwargs: Any) -> Any:
    resp = requests.request(method, url, headers=alpaca_headers(), timeout=TIMEOUT_SECONDS, **kwargs)
    resp.raise_for_status()
    return resp.json() if resp.content else None


def market_clock() -> Dict[str, Any]:
    """Alpaca's market clock: {is_open, next_open, next_close, timestamp}. Raises on failure."""
    return _request("GET", trading_base_url() + "/v2/clock")


def latest_price(symbol: str) -> float:
    feed = secret("ALPACA_DATA_FEED") or DEFAULT_DATA_FEED
    data = _request("GET", alpaca_data_url(f"/v2/stocks/{symbol}/trades/latest"), params={"feed": feed})
    return float(data["trade"]["p"])


def _price(value: float) -> float:
    # Alpaca rejects sub-penny prices at or above $1 and more than 4 decimals below.
    return round(value, 2 if value >= 1 else 4)


def client_order_id(key: Optional[str] = None) -> str:
    """newsbot-<uuid>. With a key (news id and symbol) the id is deterministic, so a
    re-delivered headline cannot open a second order: Alpaca rejects duplicate ids."""
    value = uuid.uuid5(CLIENT_ORDER_ID_NAMESPACE, key) if key else uuid.uuid4()
    return f"newsbot-{value}"


def bracket_order_payload(
    symbol: str,
    qty: int,
    entry_price: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    key: Optional[str] = None,
) -> Dict[str, Any]:
    """Market buy with a take-profit limit and a stop-loss, both relative to entry_price."""
    if qty < 1 or int(qty) != qty:
        raise ValueError("bracket orders need a whole number of shares, at least 1")
    if entry_price <= 0 or take_profit_pct <= 0 or stop_loss_pct <= 0:
        raise ValueError("entry price, take-profit and stop-loss must be positive")
    return {
        "symbol": symbol.upper(),
        "qty": str(int(qty)),
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": str(_price(entry_price * (1 + take_profit_pct / 100)))},
        "stop_loss": {"stop_price": str(_price(entry_price * (1 - stop_loss_pct / 100)))},
        "client_order_id": client_order_id(key),
    }


def submit_order(payload: Dict[str, Any]) -> Dict[str, Any]:
    if account_mode() == "live":
        check_live_buy(payload)
    return _request("POST", trading_base_url() + "/v2/orders", json=payload)


def close_all_positions() -> List[Dict[str, Any]]:
    """Paper only: cancel open orders (including bracket legs) and close every position."""
    if account_mode() != "paper":
        raise AccountError("close_all_positions is paper only; on a live account agents close "
                           "only the positions they opened (flatten_owned)")
    return _request("DELETE", trading_base_url() + "/v2/positions", params={"cancel_orders": "true"}) or []


# --- positions an agent opened today (live-safe exits) ----------------------------------


def owned_positions(prefix: str, now: Optional[datetime] = None) -> Dict[str, Dict[str, Any]]:
    """Shares still held from today's orders whose client_order_id starts with prefix
    ("newsbot-", "autopilot-"), with their open exit orders. Your own holdings of the
    same symbol are not counted."""
    owned: Dict[str, Dict[str, Any]] = {}
    for o in orders(limit=500, status="all", after=market_day_start(now), nested=True):
        if not str(o.get("client_order_id") or "").startswith(prefix):
            continue
        entry = owned.setdefault(o["symbol"], {"qty": 0.0, "open_order_ids": []})
        sign = 1 if o.get("side") == "buy" else -1
        entry["qty"] += sign * float(o.get("filled_qty") or 0)
        if o.get("side") == "sell" and o.get("status") in OPEN_ORDER_STATUSES:
            entry["open_order_ids"].append(o["id"])
        for leg in o.get("legs") or []:
            entry["qty"] -= float(leg.get("filled_qty") or 0)
            if leg.get("status") in OPEN_ORDER_STATUSES:
                entry["open_order_ids"].append(leg["id"])
    return {sym: e for sym, e in owned.items() if e["qty"] > 1e-9 or e["open_order_ids"]}


def owned_qty(prefix: str, symbol: str) -> float:
    return max(0.0, owned_positions(prefix).get(symbol.upper(), {}).get("qty", 0.0))


def flatten_owned(prefix: str, wait_seconds: float = 5.0) -> List[Dict[str, Any]]:
    """Cancel the agent's open exit orders, then sell what it still holds at market."""
    results = []
    for symbol, entry in owned_positions(prefix).items():
        for order_id in entry["open_order_ids"]:
            try:
                cancel_order(order_id)
            except Exception:
                pass  # already filled or canceled
        qty = math.floor(entry["qty"] + 1e-9)
        if qty < 1:
            continue
        deadline = time.monotonic() + wait_seconds
        while True:  # shares held by the canceled legs are released asynchronously
            payload = market_order_payload(symbol, "sell", qty)
            payload["client_order_id"] = f"{prefix}flat-{uuid.uuid4()}"[:48]
            try:
                results.append(submit_order(payload))
                break
            except requests.HTTPError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)
    return results


def positions() -> List[Dict[str, Any]]:
    _require_account()
    return _request("GET", trading_base_url() + "/v2/positions") or []



# --- account views for the desk -------------------------------------------------


def account() -> Dict[str, Any]:
    _require_account()
    return _request("GET", trading_base_url() + "/v2/account")


def orders(limit: int = 25, status: str = "all", after: Optional[str] = None,
           nested: bool = False) -> List[Dict[str, Any]]:
    _require_account()
    params: Dict[str, Any] = {"status": status, "limit": max(1, min(int(limit), 500)), "direction": "desc"}
    if after:
        params["after"] = after
    if nested:
        params["nested"] = "true"
    return _request("GET", trading_base_url() + "/v2/orders", params=params) or []


# --- the desk's manual ticket -------------------------------------------------


def cancel_order(order_id: str) -> None:
    _require_account()
    _request("DELETE", trading_base_url() + f"/v2/orders/{order_id}")


def market_order_payload(symbol: str, side: str, qty: int) -> Dict[str, Any]:
    """Plain day market order for the desk's manual paper ticket."""
    side = side.lower()
    if side not in ("buy", "sell"):
        raise ValueError("side must be buy or sell")
    if qty < 1 or int(qty) != qty:
        raise ValueError("orders need a whole number of shares, at least 1")
    return {
        "symbol": symbol.upper(),
        "qty": str(int(qty)),
        "side": side,
        "type": "market",
        "time_in_force": "day",
        "client_order_id": f"desk-{uuid.uuid4()}",
    }
