"""
Small Alpaca REST client for the news bot: market clock, latest price and
paper-account order entry.

Every function that can change the account (orders, closing positions) raises
unless ALPACA_TRADING_BASE_URL points at Alpaca's paper host. Live execution is
out of scope (docs/PLAN.md, Phase 4).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

from trader.config import secret
from trader.news_sources import alpaca_data_url, alpaca_headers

PAPER_HOST = "paper-api.alpaca.markets"
DEFAULT_TRADING_URL = f"https://{PAPER_HOST}"
DEFAULT_DATA_FEED = "iex"
TIMEOUT_SECONDS = 10
CLIENT_ORDER_ID_NAMESPACE = uuid.UUID("5b0f3a3e-2f6c-4a4e-9a51-6f1f3e2b7c10")


class NotPaperError(RuntimeError):
    """Raised when an order function is called against a non-paper base URL."""


def trading_base_url() -> str:
    base = (secret("ALPACA_TRADING_BASE_URL") or DEFAULT_TRADING_URL).rstrip("/")
    if base.endswith("/v2"):
        base = base[: -len("/v2")]
    return base


def is_paper() -> bool:
    parsed = urlparse(trading_base_url())
    return parsed.scheme == "https" and parsed.hostname == PAPER_HOST


def _require_paper() -> None:
    if not is_paper():
        raise NotPaperError(
            f"Refusing to trade: ALPACA_TRADING_BASE_URL is {trading_base_url()!r}, "
            f"not {DEFAULT_TRADING_URL}. The news bot trades Alpaca paper only."
        )


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
    _require_paper()
    return _request("POST", trading_base_url() + "/v2/orders", json=payload)


def close_all_positions() -> List[Dict[str, Any]]:
    """Cancel open orders (including bracket legs) and close every position."""
    _require_paper()
    return _request("DELETE", trading_base_url() + "/v2/positions", params={"cancel_orders": "true"}) or []


def positions() -> List[Dict[str, Any]]:
    _require_paper()
    return _request("GET", trading_base_url() + "/v2/positions") or []



# --- read-only account views for the desk (paper only) ------------------------


def account() -> Dict[str, Any]:
    _require_paper()
    return _request("GET", trading_base_url() + "/v2/account")


def orders(limit: int = 25, status: str = "all") -> List[Dict[str, Any]]:
    _require_paper()
    return _request("GET", trading_base_url() + "/v2/orders",
                    params={"status": status, "limit": max(1, min(int(limit), 500)), "direction": "desc"}) or []


# --- the desk's manual paper ticket ------------------------------------------


def cancel_order(order_id: str) -> None:
    _require_paper()
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
